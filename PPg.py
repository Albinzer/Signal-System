import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Generate a synthetic PPG signal
def generate_ppg_signal(t, heart_rate=60, noise_level=0.1):
    clean_ppg = np.sin(2 * np.pi * heart_rate / 60 * t)  # Clean PPG signal
    noise = noise_level * np.random.normal(size=len(t))  # Noise component
    noisy_ppg = clean_ppg + noise  # Noisy PPG signal
    return clean_ppg, noisy_ppg, noise

# Bandpass filter function
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Parameters
fs = 100  # Sampling frequency (Hz)
t = np.arange(0, 10, 1/fs)  # Time vector (10 seconds)
clean_ppg, noisy_ppg, noise_signal = generate_ppg_signal(t, heart_rate=75, noise_level=0.2)  # Generate signals

# Step 1: Filtering
lowcut = 0.5  # Low cutoff frequency (Hz)
highcut = 5  # High cutoff frequency (Hz)
filtered_ppg = bandpass_filter(noisy_ppg, lowcut, highcut, fs)

# Step 2: Extract noise signal (difference between noisy and filtered signal)
extracted_noise = noisy_ppg - filtered_ppg

# Step 3: Remove noise (approximate original clean signal)
reconstructed_ppg = noisy_ppg - extracted_noise

# Plotting
plt.figure(figsize=(12, 12))

# 1. Plot Original Clean PPG Signal
plt.subplot(5, 1, 1)
plt.plot(t, clean_ppg, label="Original Clean PPG Signal", color="green")
plt.title("1. Original Clean PPG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# 2. Plot Filtered PPG Signal (Denoised)
plt.subplot(5, 1, 2)
plt.plot(t, filtered_ppg, label="Filtered PPG Signal (Denoised)", color="orange")
plt.title("2. Filtered PPG Signal (Denoised)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# 3. Plot Filtered PPG Signal with Noise (Original Noisy Signal)
plt.subplot(5, 1, 3)
plt.plot(t, noisy_ppg, label="Filtered PPG Signal with Noise (Original Noisy Signal)", color="gray")
plt.title("3. Filtered PPG Signal with Noise (Original Noisy Signal)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# 4. Plot Extracted Noise with PPG Signal
plt.subplot(5, 1, 4)
plt.plot(t, extracted_noise, label="Extracted Noise", color="blue")
plt.plot(t, clean_ppg, label="Original Clean PPG Signal", color="green", linestyle="dashed")
plt.title("4. Extracted Noise with PPG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# 5. Plot Reconstructed PPG Signal (Noise Removed)
plt.subplot(5, 1, 5)
plt.plot(t, reconstructed_ppg, label="Reconstructed PPG Signal (Noise Removed)", color="purple")
plt.plot(t, clean_ppg, label="Original Clean PPG Signal (Reference)", color="green", linestyle="dashed")
plt.title("5. Reconstructed PPG Signal (Noise Removed)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
