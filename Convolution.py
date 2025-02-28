#convolution
#deepseek

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Generate a signal and a kernel
def generate_signal_and_kernel():
    t = np.linspace(0, 1, 100)  # Time vector
    signal = np.sin(2 * np.pi * 5 * t)  # Signal: Sine wave (5 Hz)
    kernel = np.array([1, 0.5, 0.25])  # Kernel: [1, 0.5, 0.25]
    return t, signal, kernel

# Perform convolution
def perform_convolution(signal, kernel):
    return convolve(signal, kernel, mode='same')  # 'same' mode preserves input size

# Parameters
t, signal, kernel = generate_signal_and_kernel()

# Perform convolution
convolved_signal = perform_convolution(signal, kernel)

# Output
print("Input Signal:", signal)
print("Kernel:", kernel)
print("Convolved Signal:", convolved_signal)

# Plotting
plt.figure(figsize=(12, 8))

# Plot input signal
plt.subplot(3, 1, 1)
plt.plot(t, signal, label="Input Signal (5 Hz Sine Wave)")
plt.title("Input Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Plot kernel
plt.subplot(3, 1, 2)
plt.stem(kernel, use_line_collection=True, label="Kernel")
plt.title("Kernel")
plt.xlabel("Index")
plt.ylabel("Amplitude")
plt.legend()

# Plot convolved signal
plt.subplot(3, 1, 3)
plt.plot(t, convolved_signal, label="Convolved Signal", color="red")
plt.title("Convolved Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

#chatgpt
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

#blacbox
import numpy as np
import matplotlib.pyplot as plt

# Function to perform convolution
def convolution(x, h):
    # Length of the input signals
    len_x = len(x)
    len_h = len(h)
    
    # Length of the output signal
    len_y = len_x + len_h - 1
    
    # Initialize the output signal
    y = np.zeros(len_y)
    
    # Perform convolution
    for n in range(len_y):
        for m in range(len_h):
            if n - m >= 0 and n - m < len_x:
                y[n] += x[n - m] * h[m]
    
    return y

# Main function
if __name__ == "__main__":
    # Define input signals
    x = np.array([1, 2, 3, 4])  # Input signal
    h = np.array([0.25, 0.5, 0.25])  # Filter (kernel)

    # Perform convolution
    y = convolution(x, h)

    # Plotting the results
    plt.figure(figsize=(10, 6))

    # Plot input signal
    plt.subplot(3, 1, 1)
    plt.stem(x, use_line_collection=True)
    plt.title('Input Signal (x[n])')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid()

    # Plot filter (kernel)
    plt.subplot(3, 1, 2)
    plt.stem(h, use_line_collection=True)
    plt.title('Filter (h[n])')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid()

    # Plot output signal
    plt.subplot(3, 1, 3)
    plt.stem(y, use_line_collection=True)
    plt.title('Output Signal (y[n] = x[n] * h[n])')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.tight_layout()
    plt.show()
