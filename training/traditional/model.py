import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Advanced Loss Function: Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Model Components ---

class SincConv_fast(nn.Module):
    """SincNet-based first layer to capture raw audio waveforms."""
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        if kernel_size % 2 == 0: kernel_size += 1
        self.conv = nn.Conv1d(1, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for Channel Attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels) # Added Attention
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply Attention
        out = self.se(out)

        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.leaky_relu(out)
        return out

class RawNet(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        
        # Frontend: SincNet
        self.sinc_layer = SincConv_fast(out_channels=128, kernel_size=251)
        self.pool_sinc = nn.MaxPool1d(3) 

        # Backbone: Residual Blocks with SE-Attention
        self.block0 = ResidualBlock(128, 128)
        self.pool0 = nn.MaxPool1d(3) 

        self.block1 = ResidualBlock(128, 256)
        self.pool1 = nn.MaxPool1d(3) 

        self.block2 = ResidualBlock(256, 512)
        self.pool2 = nn.MaxPool1d(3) 
        
        # Additional block for deeper feature extraction
        self.block3 = ResidualBlock(512, 512)
        self.pool3 = nn.MaxPool1d(3)

        # Classification Head
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.act_fc = nn.LeakyReLU(0.2)
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)

        x = self.pool_sinc(self.sinc_layer(x))
        x = self.pool0(self.block0(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))

        x = self.avg_pool(x).flatten(1)
        x = self.act_fc(self.bn_fc(self.fc1(x)))
        x = self.out(x)
        return x