import numpy as np

def imbalance_vibration(sample_rate=1000, duration=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = 1.2 * np.sin(2 * np.pi * 50 * t)
    noise = np.random.normal(0, 0.1, len(t))
    return signal + noise


def bearing_fault_vibration(sample_rate=1000, duration=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration))
    base = 0.5 * np.sin(2 * np.pi * 50 * t)

    impulses = np.zeros_like(t)
    idx = np.random.choice(len(t), size=int(0.02 * len(t)), replace=False)
    impulses[idx] = np.random.uniform(2, 3, len(idx))

    noise = np.random.normal(0, 0.1, len(t))
    return base + impulses + noise
