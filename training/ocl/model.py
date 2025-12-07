import torch
import torch.nn as nn
import torch.nn.functional as F

# --- One-Class Learning Loss: OC-Softmax ---
class OCSoftmax(nn.Module):
    """
    One-Class Softmax Loss (from ASVspoof 2019 baseline strategies).
    Encourages real speech (class 0) to be compact, and spoof (class 1) to be distant.
    """
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        
        # Center for the "Real" class in the embedding space
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, embeddings, labels):
        """
        embeddings: (Batch, feat_dim) - Output from the bottleneck layer
        labels: (Batch,) - 0 for Real, 1 for Spoof
        """
        # Normalize embeddings and center to hypersphere
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(embeddings, p=2, dim=1)

        # Cosine similarity between embeddings and center
        scores = x.mm(w.t()).squeeze() # (Batch,)
        
        # Bias the scores: 
        # For Real (0): we want score > r_real
        # For Spoof (1): we want score < r_fake
        # We construct a margin-based loss
        
        # Target scores based on label
        # If label=0 (Real), margin = r_real. If label=1 (Spoof), margin = r_fake
        margins = torch.where(labels == 0, self.r_real, self.r_fake)
        
        # Logit calculation for OC-Softmax
        # Real: alpha * (r_real - score) -> Minimize this (make score large)
        # Fake: alpha * (score - r_fake) -> Minimize this (make score small)
        
        # Note: Original OC-Softmax formulation varies. This is a simplified metric learning version.
        # Ideally, we return the Cross Entropy of the modified logits.
        
        # Standard implementation creates 2-class logits from the similarity score
        # Class 0 Logit (Realness): -|score - center|
        # Class 1 Logit (Spoofness): |score - center|
        
        # Let's stick to the simplest effective implementation for this timeframe:
        # 1-Class Objective: Minimize distance to center for Real, Maximize for Fake
        
        dist = 1.0 - scores # Cosine distance (0 to 2)
        
        # Hinge Loss equivalent for OCL
        loss_real = self.softplus(self.alpha * (dist - (1 - self.r_real))) # Penalize if dist > (1-r_real)
        loss_fake = self.softplus(self.alpha * ((1 - self.r_fake) - dist)) # Penalize if dist < (1-r_fake)
        
        loss = torch.where(labels == 0, loss_real, loss_fake).mean()
        
        return loss, scores

# --- Model Components ---

class SincConv_fast(nn.Module):
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
        self.se = SEBlock(out_channels) 
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.leaky_relu(out)
        return out

class RawNetOCL(nn.Module):
    """
    RawNet modified to accept Extra Features and output Embedding for OCL.
    """
    def __init__(self, d_args):
        super().__init__()
        
        # Raw Audio Branch
        self.sinc_layer = SincConv_fast(out_channels=128, kernel_size=251)
        self.pool_sinc = nn.MaxPool1d(3) 
        self.block0 = ResidualBlock(128, 128)
        self.pool0 = nn.MaxPool1d(3) 
        self.block1 = ResidualBlock(128, 256)
        self.pool1 = nn.MaxPool1d(3) 
        self.block2 = ResidualBlock(256, 512)
        self.pool2 = nn.MaxPool1d(3) 
        self.block3 = ResidualBlock(512, 512)
        self.pool3 = nn.MaxPool1d(3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature Branch (Shimmer, ZCR, Centroid)
        # Simple MLP to upscale features to match embedding space
        self.feature_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU()
        )

        # Fusion & Bottleneck
        # 256 (from RawNet fc1) + 128 (from Features) = 384
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.act_fc = nn.LeakyReLU(0.2)
        
        # Final Embedding Layer (dimension 64 for OCL)
        self.bottleneck = nn.Linear(256 + 128, 64) 

    def forward(self, x_raw, x_feat):
        # 1. Raw Audio Path
        if x_raw.dim() == 2: x_raw = x_raw.unsqueeze(1)
        x = self.pool_sinc(self.sinc_layer(x_raw))
        x = self.pool0(self.block0(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.avg_pool(x).flatten(1)
        x_raw_emb = self.act_fc(self.bn_fc(self.fc1(x))) # 256 dim

        # 2. Handcrafted Feature Path
        x_feat_emb = self.feature_mlp(x_feat) # 128 dim
        
        # 3. Concatenate
        combined = torch.cat((x_raw_emb, x_feat_emb), dim=1) # 384 dim
        
        # 4. Bottleneck (Embedding for OCL)
        embedding = self.bottleneck(combined) # 64 dim
        
        return embedding