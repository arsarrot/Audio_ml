import numpy as np
import librosa
from numba import jit  
from matplotlib import pyplot as plt
import scipy.io.wavfile as sci_wav
import librosa.display as libdisplay

def framing_signal(signal: np.array, fs: float, frame_size: float, frame_step: float):
    frame_length = np.round(frame_size * fs).astype(int)
    frame_step = np.round(frame_step * fs).astype(int)
    signal_length = signal.shape[0]
    n_frames = np.ceil(abs(signal_length - frame_length) / frame_step).astype(int)
    pad_signal_length = int(n_frames * frame_step + frame_length)
    zeros_pad = np.zeros((1, pad_signal_length - signal_length))
    pad_signal = np.concatenate((signal.reshape((1, -1)), zeros_pad), axis=1).reshape(-1)
    frames = np.zeros((n_frames, frame_length))
    indices = np.arange(0, frame_length)
    for i in np.arange(0, n_frames):
        offset = i * frame_step
        frames[i] = pad_signal[(indices + offset)]
    return frames
def hamming_window(frames: np.array):
    window_length = frames.shape[1]
    n = np.arange(0, window_length)
    h = 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_length - 1))
    frames *= h
    return frames
def stft_spec(y: np.array, fs: float, n_fft: int, frame_size: float, frame_step: float):
    frames = framing_signal(y, fs, frame_size, frame_step)
    frames = hamming_window(frames)
    spec_frames = np.fft.fft(frames, n=n_fft, axis=1)
    spec_frames = spec_frames[:, 0: int(n_fft / 2 + 1)]
    return spec_frames
def mel_filtering(frames, f_min, f_max, n_mels, fs):
    n_fft = frames.shape[1] - 1
    mel_lf = 2595 * np.log10(1 + f_min / 700)
    mel_hf = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_lf, mel_hf, n_mels + 2)
    hz_points = 700 * (np.power(10, mel_points / 2595) - 1)
    fft_bank_bin = np.floor((n_fft + 1) * hz_points / (fs / 2))
    fft_bank_bin[-1] = n_fft
    f_bank = np.zeros((n_mels, n_fft + 1))
    for i in np.arange(1, n_mels + 1):
        left_f = int(fft_bank_bin[i - 1])
        center_f = int(fft_bank_bin[i])
        right_f = int(fft_bank_bin[i + 1])
        for k in np.arange(left_f, center_f + 1):
            f_bank[i - 1, k] = (k - left_f) / (center_f - left_f)
        for k in np.arange(center_f, right_f + 1):
            f_bank[i - 1, k] = (-k + right_f) / (-center_f + right_f)
        f_bank[i - 1] /= (hz_points[i] - hz_points[i-1])
    filtered_frames = np.dot(frames, f_bank.T)
    filtered_frames += np.finfo(float).eps
    return filtered_frames, hz_points
def signal_power_to_db(power_frames, min_amp=1e-10, top_db=80):
    log_spec = 10.0 * np.log10(np.maximum(min_amp, power_frames))
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec
def dct(frames: np.array):
    rows, cols = frames.shape
    N = cols
    n = np.arange(1, N + 1)
    weights = np.zeros((N, N))
    for k in np.arange(0, N):
        weights[:, k] = np.cos(np.pi * (n - 1 / 2) * k / N)
    dct_signal = np.sqrt(2 / N) * np.dot(frames, weights)
    return dct_signal
def sinusoidal_liftering(mfcc: np.array):
    mfcc_lift = np.zeros(mfcc.shape)
    n = np.arange(1, mfcc_lift.shape[1] + 1)
    D = 22
    w = 1 + (D / 2) * np.sin(np.pi * n / D)
    mfcc_lift = mfcc * w
    return mfcc_lift
def mfcc_features(y: np.array, fs: float, n_fft: int = 512, frame_size: float = 0.025, frame_step: float = 0.01,
                         n_mels: int = 40, n_mfcc: int = 13):
    mag_spec_frames = np.abs(stft_spec(y, fs, n_fft, frame_size, frame_step))
    pow_spec_frames = (mag_spec_frames**2) / mag_spec_frames.shape[1]
    mel_power_spec_frames, hz_freq = mel_filtering(pow_spec_frames, 0, fs/2, n_mels, fs)
    log_spec_frames = signal_power_to_db(mel_power_spec_frames)
    mfcc = dct(log_spec_frames)
    mfcc = sinusoidal_liftering(mfcc)
    mfcc = mfcc[:, 1:n_mfcc]
    return mfcc
def spec_visual(y_spec: np.array, y_spec_lib: np.array, parameters: dict, title: str = "", title_lib: str = ""):
      plt.figure(figsize=(18, 5))
      plt.subplot(1, 2, 1)
      plt.title(title, fontsize=14)
      libdisplay.specshow(y_spec, y_axis='linear', sr=parameters['fs'], cmap='copper', x_axis='time', hop_length=parameters['window_step'])
      plt.tick_params(axis='both', which='major', labelsize=13) 
      plt.subplot(1, 2, 2)
      plt.title(title_lib, fontsize=14)
      libdisplay.specshow(y_spec_lib, y_axis='linear', sr=parameters['fs'], cmap='copper', x_axis='time', hop_length=parameters['window_step'])
      plt.tight_layout()
      plt.tick_params(axis='both', which='major', labelsize=13) 
def mfcc_visual(y_spec: np.array, y_spec_lib: np.array, parameters: dict, title: str = "", title_lib: str='' ):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 2, 1)
    plt.title(title, fontsize=14)
    libdisplay.specshow(y_spec, y_axis='frames', sr=parameters['fs'], x_axis='time', hop_length=parameters['window_step'], cmap = 'copper')
    plt.colorbar()
    plt.ylabel('MFCC')
    plt.subplot(1, 2, 2)
    plt.title(title_lib, fontsize=14)
    libdisplay.specshow(y_spec_lib, y_axis='frames', sr=parameters['fs'], x_axis='time', hop_length=parameters['window_step'], cmap = 'copper')
    plt.colorbar()
    plt.ylabel('MFCC')
def spectrogram_creation(signal: np.array, parameters: dict):
    n_fft = parameters['n_fft']
    window_length = parameters['window_length']
    window_step = parameters['window_step']
    spec_lib = librosa.amplitude_to_db(np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=window_step, win_length=window_length)))
    frame_size = window_length / fs
    frame_step = window_step / fs
    power_spectrum_frames = np.abs(stft_spec(y=signal, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step))**2
    spec = signal_power_to_db(power_spectrum_frames)
    return spec_lib, spec.T
def drawing_mfccs(signal: np.array, parameters: dict):
    n_fft = parameters['n_fft']
    window_length = parameters['window_length']
    window_step = parameters['window_step']
    n_mels = 40
    n_mfcc = 13
    mfcc_lib = librosa.feature.mfcc(sr=fs, y=signal, n_mfcc=n_mfcc, n_mels=n_mels, win_length=window_length, hop_length=window_step, lifter=20)[1:n_mfcc, :]
    frame_size = window_length / fs
    frame_step = window_step / fs
    mfcc = mfcc_features(y=signal, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step)
    return mfcc_lib, mfcc.T
    
audio_input = 'svoice.wav'
fs, y = sci_wav.read(audio_input)
y = 1.0 * y
t = np.linspace(0, y.shape[0] / fs, y.shape[0])
plt.figure(figsize=(6, 4))
plt.title("Time signal")
plt.plot(t, y, color = 'firebrick')
n_fft = 512
params = {
    'fs': fs,
    'n_fft': n_fft,
    'window_length': n_fft,
    'window_step': 441
}
spec_lib, spec = spectrogram_creation(y, params)
spec_visual(spec, spec_lib, params, "Generated Spectrogram", "Librosa Spectrogram")
mffc_lib, mfcc = drawing_mfccs(y, params)
mfcc_visual(mfcc, mffc_lib, params, "Calculated MFCCs", 'MFCCs with Librosa')
plt.show()