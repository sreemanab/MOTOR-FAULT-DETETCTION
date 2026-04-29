import numpy as np
import os
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
import numpy as np


np.random.seed(42)



import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -----------------------------
# 1. Signal Simulation
# -----------------------------
def simulate_signal(fault=False, duration=2, fs=1000):
    t = np.linspace(0, duration, duration * fs)
    
    # Base healthy vibration (50 Hz motor)
    signal = 0.5 * np.sin(2 * np.pi * 50 * t)

    if fault:
     signal += 0.15 * np.sin(2 * np.pi * 120 * t)  # reduced amplitude
     signal += 0.2 * np.random.normal(0, 1, len(t))  # moderate noise

    return signal, fs


# -----------------------------
# 2. Feature Extraction
# -----------------------------
def extract_features(signal, fs):
    
    # Time domain
    rms = np.sqrt(np.mean(signal**2))
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    kurt = kurtosis(signal)
    skewness = skew(signal)
    crest_factor = np.max(np.abs(signal)) / rms

    # Frequency domain
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
# 3. Dataset Generation
# -----------------------------
def generate_dataset(samples=200):
    
    feature_list = []
    labels = []

    for _ in range(samples):
        signal, fs = simulate_signal(fault=False)
        features = extract_features(signal, fs)
        feature_list.append(features)
        labels.append(0)

    for _ in range(samples):
        signal, fs = simulate_signal(fault=True)
        features = extract_features(signal, fs)
        feature_list.append(features)
        labels.append(1)

    columns = [
        "rms", "mean", "std",
        "kurtosis", "skewness", "crest_factor",
        "peak_freq", "spectral_centroid", "band_energy"
    ]

    df = pd.DataFrame(feature_list, columns=columns)
    df["label"] = labels

    return df


# -----------------------------
# 4. Save Dataset
# -----------------------------
if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv("vibration_data.csv", index=False)
    print("Dataset generated successfully!")

   
