import numpy as np
import librosa
from matplotlib import pyplot as plt

msound = 'stft.wav'
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