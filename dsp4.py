import wave
import struct
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
try:
    file = wave.open("speech_audio.wav", "r")
except:
    print("Error: Unable to open audio file")
    exit()

n_frames = file.getnframes()
signal = np.zeros(n_frames)

for i in range(n_frames):
    # Read the audio samples and convert to float
    try:
        frame = file.readframes(1)
        sample = struct.unpack("<h", frame)[0]
        signal[i] = sample / 32768.0
    except:
        print("Error: Unable to read audio frame at index ", i)
        exit()

file.close()

# Preprocess the audio signal
signal = signal - np.mean(signal)
signal = signal / np.max(np.abs(signal))
frame_len = int(0.02 * file.getframerate())
hop_len = int(0.01 * file.getframerate())

# Compute the autocorrelation function and extract the autocorrelation feature
autocorr = np.zeros(signal.shape)
for i in range(frame_len, len(signal)):
    autocorr[i] = np.sum(signal[i-frame_len:i] * signal[i:i-frame_len:-1])

autocorr = autocorr[frame_len//2:]
autocorr = autocorr[:len(signal)//hop_len]

# Apply a threshold to detect applause events
threshold = np.mean(autocorr) + 3*np.std(autocorr)
is_applause = autocorr > threshold

# Plot the detected applause events
plt.figure(figsize=(12, 6))
plt.plot(signal)
plt.vlines(np.where(is_applause)[0]*hop_len, -1, 1, color='r')
plt.title("Speech Signal with Detected Applause Events")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()
