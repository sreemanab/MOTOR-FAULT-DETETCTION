import numpy as np
from scipy.stats import kurtosis    

def extract_features(signal):
    return {
        "rms": np.sqrt(np.mean(signal**2)),
        "peak": np.max(np.abs(signal)),
        "std": np.std(signal),
        "kurtosis": kurtosis(signal)
    }
