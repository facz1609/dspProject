import librosa
import numpy as np

# Load audio file
file_path = 'aa.wav'
y, sr = librosa.load(file_path, sr=None)

# Calculate autocorrelation
autocorr = np.correlate(y, y, mode='full')
autocorr = autocorr[autocorr.size // 2:]

# Normalize autocorrelation
autocorr /= autocorr.max()
# In MATLAB, you can use the audioread function to load audio files and the xcorr function to calculate the autocorrelation. Here's a code snippet for MATLAB:

# % Load audio file
file_path = 'path/to/your/audio/file.wav';
[y, Fs] = audioread(file_path);

# % Calculate autocorrelation
autocorr = xcorr(y, 'coeff');
autocorr = autocorr(length(y):end);