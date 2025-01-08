import math
import wave
import numpy as np
import librosa
from numba import jit  
from matplotlib import pyplot as plt
import IPython.display as ipd


FRAMES_PER_SECOND = 44100
def beat(frequency1, frequency2, num_seconds):
    for frame in range(round(num_seconds * FRAMES_PER_SECOND)):
        time = frame / FRAMES_PER_SECOND
        amplitude1 = math.sin(2 * math.pi * frequency1 * time)
        amplitude2 = math.sin(2 * math.pi * frequency2 * time)
        amplitude = max(-1, min(amplitude1 + amplitude2, 1))
        yield round((amplitude + 1) / 2 * 255)
with wave.open("msound.wav", mode="wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(1)
    wav_file.setframerate(FRAMES_PER_SECOND)
    wav_file.writeframes(bytes(beat(5000, 10000, 3)))


msound = 'msound.wav'
mmusic, sampling_rate = librosa.load(msound, sr= None, dtype= 'float32')
print(f"Length of signal: {mmusic.size} samples and the data type {mmusic.dtype}")
print('sampling rate:', sampling_rate)


def stft_basic(x, w, H=8, only_positive_frequencies=True):
    N = len(w)
    L = len(x)
    M = np.floor((L - N) / H).astype(int)
    X = np.zeros((N, M), dtype='complex')
    for m in range(M):
        H_int = int(H) 
        x_win = x[m * H_int:m * H_int + N] * w
        X_win = np.fft.fft(x_win)
        X[:, m] = X_win

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]
        return X
x = mmusic[:1024]
N = 32
H = 16
w = np.ones(N)
X = stft_basic(x, w, H, only_positive_frequencies=True)
Y = np.abs(X) ** 2
plt.figure(figsize=(18, 5))
librosa.display.specshow(Y, sr=44100, hop_length=16, cmap='Greys',x_axis="time", y_axis= 'linear')
plt.colorbar()