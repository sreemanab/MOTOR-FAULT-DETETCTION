import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq

np.random.seed(42)

# -----------------------------
# Safe Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
real_data_path = os.path.join(BASE_DIR, "real_data")
output_path = os.path.join(BASE_DIR, "data")

# -----------------------------
# Load MAT File
# -----------------------------
def load_signal(file_path):
    mat_data = loadmat(file_path)

    print("Keys in file:", mat_data.keys())

    for key in mat_data.keys():
        if "DE_time" in key:
            return mat_data[key].flatten()

    raise ValueError("Drive End signal not found in MAT file")

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(signal, fs=12000):

    rms = np.sqrt(np.mean(signal**2))
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    kurt = kurtosis(signal)
    skewness = skew(signal)
    crest_factor = np.max(np.abs(signal)) / rms

    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)

    magnitude = np.abs(yf[:N//2])
    freq = xf[:N//2]

    peak_freq = freq[np.argmax(magnitude)]
    spectral_centroid = np.sum(freq * magnitude) / np.sum(magnitude)
    band_energy = np.sum(magnitude**2)

    return [
        rms, mean_val, std_val,
        kurt, skewness, crest_factor,
        peak_freq, spectral_centroid, band_energy
    ]

# -----------------------------
# Segment Signal
# -----------------------------
def segment_signal(signal, window_size=2048):
    segments = []
    for i in range(0, len(signal) - window_size, window_size):
        segments.append(signal[i:i+window_size])
    return segments

# -----------------------------
# Build Dataset
# -----------------------------
def build_dataset():

    normal_files = ["97.mat", "98.mat", "99.mat", "100.mat"]

    ir_files = ["105.mat", "108.mat"]
    or_files = ["171.mat", "211.mat"]

    feature_list = []
    labels = []
    sources = []

    # -------- NORMAL --------
    for file in normal_files:
        file_path = os.path.join(real_data_path, file)
        signal = load_signal(file_path)
        segments = segment_signal(signal)

        for seg in segments:
            feature_list.append(extract_features(seg))
            labels.append(0)  # Normal
            sources.append(file)

    # -------- INNER RACE --------
    for file in ir_files:
        file_path = os.path.join(real_data_path, file)
        signal = load_signal(file_path)
        segments = segment_signal(signal)

        for seg in segments:
            feature_list.append(extract_features(seg))
            labels.append(1)  # IR
            sources.append(file)

    # -------- OUTER RACE --------
    for file in or_files:
        file_path = os.path.join(real_data_path, file)
        signal = load_signal(file_path)
        segments = segment_signal(signal)

        for seg in segments:
            feature_list.append(extract_features(seg))
            labels.append(2)  # OR
            sources.append(file)

    columns = [
        "rms", "mean", "std",
        "kurtosis", "skewness", "crest_factor",
        "peak_freq", "spectral_centroid", "band_energy"
    ]

    df = pd.DataFrame(feature_list, columns=columns)
    df["label"] = labels
    df["source"] = sources

    df.to_csv(os.path.join(output_path, "real_vibration_dataset.csv"), index=False)

    print("Multi-class dataset created!")
    print("Dataset shape:", df.shape)



if __name__ == "__main__":
    build_dataset()
