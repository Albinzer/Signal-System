import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Define two signals
x = np.array([1, 2, 3, 4])  # Input signal
h = np.array([0, 1, 0.5])  # Impulse response

# Perform Linear Convolution
linear_conv = np.convolve(x, h, mode='full')

# Perform Circular Convolution
N = max(len(x), len(h))  # Define the period for circular convolution
circular_conv = np.fft.ifft(np.fft.fft(x, N) * np.fft.fft(h, N)).real  # Using FFT for circular convolution

# Plot Signals
plt.figure(figsize=(12, 6))

# Input Signal
plt.subplot(2, 2, 1)
plt.stem(x, basefmt="k", use_line_collection=True)
plt.title("Input Signal x(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")

# Impulse Response
plt.subplot(2, 2, 2)
plt.stem(h, basefmt="k", use_line_collection=True)
plt.title("Impulse Response h(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")

# Linear Convolution Result
plt.subplot(2, 2, 3)
plt.stem(linear_conv, basefmt="k", use_line_collection=True)
plt.title("Linear Convolution Result")
plt.xlabel("n")
plt.ylabel("Amplitude")

# Circular Convolution Result
plt.subplot(2, 2, 4)
plt.stem(circular_conv, basefmt="k", use_line_collection=True)
plt.title("Circular Convolution Result")
plt.xlabel("n")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
