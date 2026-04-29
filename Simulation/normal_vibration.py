import numpy as np

def normal_vibration(sample_rate=1000, duration=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = 0.5 * np.sin(2 * np.pi * 50 * t)
    noise = np.random.normal(0, 0.05, len(t))
    return signal + noise   # ✅ ONLY signal
