import numpy as np
import matplotlib.pyplot as plt

# Generate a sample signal (sum of two sinusoids)
Fs = 1000  # Sampling frequency (Hz)
T = 1  # Duration (seconds)
t = np.linspace(0, T, Fs, endpoint=False)  # Time vector

f1, f2 = 50, 150  # Frequencies in Hz
signal = 2 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Compute DFT using FFT
dft_result = np.fft.fft(signal)  # Corrected function call
frequencies = np.fft.fftfreq(len(signal), d=1/Fs)  # Frequency axis

# Keep only positive frequencies
positive_frequencies = frequencies[:len(frequencies)//2]
positive_magnitude = np.abs(dft_result[:len(dft_result)//2])

# Plot the original signal
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, signal, label="Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time Domain Signal")
plt.legend()
plt.grid()

# Plot the DFT Magnitude Spectrum (Only Positive Frequencies)
plt.subplot(1, 2, 2)
plt.plot(positive_frequencies, positive_magnitude, color='red', label="DFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Domain (DFT)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
