import React, { useState, useRef, useEffect } from 'react';
import { Mic, Upload, ShieldCheck, ShieldAlert, Lock, Unlock, Play, Square, FileAudio, AlertTriangle, Settings } from 'lucide-react';

const VoiceAuthApp = () => {
  const [activeTab, setActiveTab] = useState('record'); // 'record' or 'upload'
  const [recording, setRecording] = useState(false);
  const [timeLeft, setTimeLeft] = useState(5);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [demoMode, setDemoMode] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const timerIntervalRef = useRef(null);
  const chunksRef = useRef([]);

  // Cleanup URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const startRecording = async () => {
    setResult(null);
    setError(null);
    setAudioBlob(null);
    setTimeLeft(5);
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setAudioUrl(url);
        
        // Stop all tracks to release mic
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setRecording(true);

      // Start Countdown
      timerIntervalRef.current = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            stopRecording();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

    } catch (err) {
      setError("Could not access microphone. Please ensure permissions are granted.");
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      clearInterval(timerIntervalRef.current);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setAudioBlob(file);
      setAudioUrl(url);
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!audioBlob) return;

    setLoading(true);
    setError(null);

    // --- DEMO MODE LOGIC (For Preview Purposes) ---
    if (demoMode) {
      setTimeout(() => {
        setLoading(false);
        // Randomly succeed or fail for demo
        const isBonafide = Math.random() > 0.5;
        setResult({
          access_granted: isBonafide,
          label: isBonafide ? "Bonafide" : "Spoof",
          confidence: isBonafide ? 0.98 : 0.12,
          spoof_probability: isBonafide ? 0.02 : 0.88
        });
      }, 2000);
      return;
    }

    // --- REAL BACKEND LOGIC ---
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Failed to connect to backend. Make sure 'server.py' is running on port 5000.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setAudioBlob(null);
    setResult(null);
    setError(null);
    setTimeLeft(5);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center p-4 font-sans">
      
      {/* Main Card */}
      <div className="w-full max-w-md bg-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-slate-700 relative">
        
        {/* Header */}
        <div className="bg-slate-900/50 p-6 text-center border-b border-slate-700">
          <h1 className="text-xl font-bold tracking-wider text-blue-400 uppercase flex items-center justify-center gap-2">
            <ShieldCheck className="w-6 h-6" />
            Secure Voice Gate
          </h1>
          <p className="text-xs text-slate-400 mt-1">Deepfake Detection & Anti-Spoofing</p>
        </div>

        {/* Demo Mode Toggle (For Preview Convenience) */}
        <div className="absolute top-4 right-4 group">
            <button 
                onClick={() => setDemoMode(!demoMode)}
                className={`p-1.5 rounded-full transition-colors ${demoMode ? 'bg-blue-500 text-white' : 'bg-slate-700 text-slate-400'}`}
                title="Toggle Demo Mode (Simulate Backend)"
            >
                <Settings className="w-4 h-4" />
            </button>
        </div>

        {/* Content Area */}
        <div className="p-6">
          
          {/* Tabs */}
          {!result && (
            <div className="flex bg-slate-700/50 p-1 rounded-lg mb-6">
              <button
                onClick={() => setActiveTab('record')}
                className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm font-medium transition-all ${
                  activeTab === 'record' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'
                }`}
              >
                <Mic className="w-4 h-4" /> Record
              </button>
              <button
                onClick={() => setActiveTab('upload')}
                className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm font-medium transition-all ${
                  activeTab === 'upload' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'
                }`}
              >
                <Upload className="w-4 h-4" /> Upload
              </button>
            </div>
          )}

          {/* Result View */}
          {result ? (
            <div className="text-center animate-in fade-in zoom-in duration-300">
              <div className={`mx-auto w-24 h-24 rounded-full flex items-center justify-center mb-4 ${
                result.access_granted ? 'bg-green-500/20 text-green-400 ring-4 ring-green-500/10' : 'bg-red-500/20 text-red-400 ring-4 ring-red-500/10'
              }`}>
                {result.access_granted ? <Unlock className="w-12 h-12" /> : <Lock className="w-12 h-12" />}
              </div>
              
              <h2 className={`text-2xl font-bold mb-1 ${result.access_granted ? 'text-green-400' : 'text-red-400'}`}>
                {result.access_granted ? 'ACCESS GRANTED' : 'ACCESS DENIED'}
              </h2>
              <p className="text-slate-400 mb-6">
                Identity Verification: <span className="text-white font-medium">{result.label}</span>
              </p>

              <div className="bg-slate-900/50 rounded-lg p-4 mb-6 text-sm">
                <div className="flex justify-between mb-2">
                  <span className="text-slate-400">Bonafide Score:</span>
                  <span className="text-blue-400 font-mono">{(result.confidence * 100).toFixed(2)}%</span>
                </div>
                <div className="w-full bg-slate-700 h-2 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${result.access_granted ? 'bg-green-500' : 'bg-red-500'}`} 
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-2 text-left">
                  Threshold: 50.00% | Latency: 45ms
                </p>
              </div>

              <button 
                onClick={reset}
                className="w-full py-3 bg-slate-700 hover:bg-slate-600 rounded-lg font-medium transition-colors"
              >
                Verify Another User
              </button>
            </div>
          ) : (
            // Input Views
            <div className="space-y-6">
              
              {activeTab === 'record' ? (
                <div className="text-center py-4">
                    <div className="relative inline-block">
                        {recording && (
                            <span className="absolute inset-0 rounded-full animate-ping bg-red-500/50"></span>
                        )}
                        <button
                            onClick={recording ? stopRecording : startRecording}
                            className={`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-all duration-300 ${
                            recording ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-600 hover:bg-blue-500'
                            } shadow-xl border-4 border-slate-700`}
                        >
                            {recording ? <Square className="w-10 h-10 fill-current" /> : <Mic className="w-10 h-10" />}
                        </button>
                    </div>
                    
                    <p className={`mt-4 font-mono text-lg ${recording ? 'text-red-400 animate-pulse' : 'text-slate-400'}`}>
                        {recording ? `00:0${timeLeft} / 00:05` : 'Press to Record'}
                    </p>
                </div>
              ) : (
                <div className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center hover:border-blue-500 transition-colors bg-slate-900/20">
                  <input 
                    type="file" 
                    accept="audio/*" 
                    onChange={handleFileUpload} 
                    className="hidden" 
                    id="audio-upload"
                  />
                  <label htmlFor="audio-upload" className="cursor-pointer flex flex-col items-center gap-2">
                    <FileAudio className="w-12 h-12 text-slate-500" />
                    <span className="text-blue-400 font-medium">Click to Upload Audio</span>
                    <span className="text-xs text-slate-500">Supports WAV, FLAC, MP3</span>
                  </label>
                </div>
              )}

              {/* Audio Preview */}
              {audioBlob && (
                <div className="bg-slate-700/30 rounded-lg p-3 flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-500/20 rounded-full flex items-center justify-center text-blue-400">
                    <Play className="w-5 h-5 ml-1" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {activeTab === 'record' ? 'Recorded_Voice.wav' : audioBlob.name}
                    </p>
                    <p className="text-xs text-slate-500">{(audioBlob.size / 1024).toFixed(1)} KB</p>
                  </div>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-start gap-2 text-red-400 text-sm">
                  <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                  <p>{error}</p>
                </div>
              )}

              {/* Action Button */}
              <button
                onClick={handleSubmit}
                disabled={!audioBlob || loading}
                className={`w-full py-3 rounded-lg font-bold text-lg transition-all ${
                  !audioBlob || loading
                    ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20'
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing Biometrics...
                  </span>
                ) : (
                  'Verify Access'
                )}
              </button>
              
              <p className="text-center text-xs text-slate-600">
                 Server Status: {demoMode ? 'Simulated' : 'Waiting for connection...'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VoiceAuthApp;