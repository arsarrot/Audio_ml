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
    wav_file.writeframes(bytes(beat(500, 5000, 3)))


msound = 'msound.wav'
mmusic, sampling_rate = librosa.load(msound, sr= None, dtype= 'float32')
print(f"Length of signal: {mmusic.size} samples and the data type {mmusic.dtype}")
print('sampling rate:', sampling_rate)
ipd.Audio('msound.wav')


@jit(nopython=True)
def generate_matrix_dft(N:int, K:int):
    dft = np.zeros((K, N), dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            dft[k, n] = np.exp(-2j * np.pi * k * n / N)
    return dft


@jit(nopython=True)
def dft(x:np.array):
    x = x.astype(np.complex128)
    N = len(x)
    dft_mat = generate_matrix_dft(N, k)
    return np.dot(dft_mat, x)


@jit(nopython=True)
def twiddle(N:int):

    k = np.arange(N // 2)
    sigma = np.exp(-2j * np.pi * k / N)
    return sigma


@jit(nopython=True)
def fft(x:np.array):
    x = x.astype(np.complex128)
    N = len(x)
    log2N = np.log2(N)
    assert log2N == int(log2N)
    X = np.zeros(N, dtype=np.complex128)

    if N == 1:
        return x
    else:
        this_range = np.arange(N)
        A = fft(x[this_range % 2 == 0])
        B = fft(x[this_range % 2 == 1])
        C = twiddle(N) * B
        X[:N//2] = A + C
        X[N//2:] = A - C
        return X


N = 4096
n = np.arange(N)
x = mmusic[:4096]
k = 4096
X = fft(x)
X = np.abs(X)
f = np.linspace(0, 44100, len(X))
f_bins = int(len(X)*0.5)  
plt.figure(figsize=(18, 5))
plt.subplot(1, 2, 1)
plt.plot(x, 'k')
plt.tick_params(axis='both', which='major', labelsize=13) 
plt.xlabel('Time(n)', fontsize=14)
plt.subplot(1, 2, 2)
plt.plot(f[:f_bins], X[:f_bins],'k')
plt.xlabel('Frequency(Hz)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=13) 
plt.tight_layout()