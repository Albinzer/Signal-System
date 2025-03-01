
#Rafi Dft

import numpy as np
import matplotlib.pyplot as plt

# Function to compute DFT
def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):  # Loop over output frequencies
        for n in range(N):  # Loop over input samples
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Generate a sample signal (sum of two sinusoids)
Fs = 500 # Sampling frequency (Hz)
T = 1  # Duration (seconds)
t = np.linspace(0, T, Fs, endpoint=False)  # Time vector

f1, f2 = 50, 150  # Frequencies in Hz
signal = 2*np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Compute DFT
dft_result = DFT(signal)
frequencies = np.fft.fftfreq(len(signal), d=1/Fs)

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
plt.plot(frequencies[:Fs//2], np.abs(dft_result)[:Fs//2])

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Domain (DFT)")
plt.grid()

plt.tight_layout()
plt.show()


