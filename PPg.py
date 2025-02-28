#ppg-filtering,feature extraction,peak detection
#Deepseek

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Generate a synthetic PPG signal
def generate_ppg_signal(t, heart_rate=60, noise_level=0.1):
    ppg = np.sin(2 * np.pi * heart_rate / 60 * t)  # PPG signal
    noise = noise_level * np.random.normal(size=len(t))  # Add noise
    return ppg + noise

# Bandpass filter
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Feature extraction: Heart rate calculation
def calculate_heart_rate(peaks, fs):
    rr_intervals = np.diff(peaks) / fs  # RR intervals in seconds
    heart_rate = 60 / np.mean(rr_intervals)  # Heart rate in BPM
    return heart_rate

# Parameters
fs = 100  # Sampling frequency (Hz)
t = np.arange(0, 10, 1/fs)  # Time vector (10 seconds)
ppg_signal = generate_ppg_signal(t, heart_rate=75, noise_level=0.2)  # Synthetic PPG signal

# Step 1: Filtering
lowcut = 0.5  # Low cutoff frequency (Hz)
highcut = 5  # High cutoff frequency (Hz)
filtered_ppg = bandpass_filter(ppg_signal, lowcut, highcut, fs)

# Step 2: Peak detection
peaks, _ = find_peaks(filtered_ppg, height=0.5, distance=fs//2)

# Step 3: Feature extraction
heart_rate = calculate_heart_rate(peaks, fs)

# Output
print("Detected Peaks:", peaks)
print("Calculated Heart Rate:", heart_rate, "BPM")

# Plotting
plt.figure(figsize=(12, 8))

# Plot original PPG signal
plt.subplot(3, 1, 1)
plt.plot(t, ppg_signal, label="Original PPG Signal")
plt.title("Original PPG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Plot filtered PPG signal
plt.subplot(3, 1, 2)
plt.plot(t, filtered_ppg, label="Filtered PPG Signal", color="orange")
plt.title("Filtered PPG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Plot filtered PPG signal with detected peaks
plt.subplot(3, 1, 3)
plt.plot(t, filtered_ppg, label="Filtered PPG Signal", color="orange")
plt.plot(t[peaks], filtered_ppg[peaks], "x", label="Detected Peaks", color="red")
plt.title("Filtered PPG Signal with Detected Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

#chatgpt

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Simulated PPG signal (replace with real data if available)
Fs = 100  # Sampling frequency (Hz)
t = np.linspace(0, 10, Fs * 10)  # 10 seconds of data
ppg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.5 * t)  # Synthetic PPG

# Add noise
np.random.seed(42)
ppg_signal += 0.1 * np.random.randn(len(t))

# Bandpass Filter (0.5 - 5 Hz) for PPG
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

filtered_ppg = bandpass_filter(ppg_signal, 0.5, 5, Fs)

# Peak Detection
peaks, _ = find_peaks(filtered_ppg, height=0.3, distance=Fs/2)  # Adjust height & distance as needed

# Heart Rate Calculation
peak_intervals = np.diff(peaks) / Fs  # Convert to time (seconds)
heart_rate = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0  # BPM calculation

# Plot the Results
plt.figure(figsize=(12, 6))

# Raw PPG Signal
plt.subplot(2, 1, 1)
plt.plot(t, ppg_signal, label="Raw PPG Signal", color='gray')
plt.title("Raw PPG Signal with Noise")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Filtered PPG with Peaks
plt.subplot(2, 1, 2)
plt.plot(t, filtered_ppg, label="Filtered PPG Signal", color='blue')
plt.plot(t[peaks], filtered_ppg[peaks], "ro", label="Detected Peaks")  # Mark peaks
plt.title(f"Filtered PPG Signal & Detected Peaks (Heart Rate: {heart_rate:.2f} BPM)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
