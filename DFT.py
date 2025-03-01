#discrete fourier transform 
#Deepseek

import numpy as np
import matplotlib.pyplot as plt

# Input: Time-domain signal (e.g., a sine wave)
N = 8  # Number of samples
x = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # Example input signal

# DFT function
def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)  # Output frequency-domain signal
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Compute DFT
X = DFT(x)

# Output: Frequency-domain representation
print("Input Signal (x):", x)
print("DFT Output (X):", X)

# Plotting
plt.figure(figsize=(12, 6))

# Plot input signal
plt.subplot(2, 1, 1)
plt.stem(x, use_line_collection=True)
plt.title("Input Signal (Time Domain)")
plt.xlabel("Time (n)")
plt.ylabel("Amplitude")

# Plot DFT output (Magnitude)
plt.subplot(2, 1, 2)
plt.stem(np.abs(X), use_line_collection=True)
plt.title("DFT Output (Frequency Domain)")
plt.xlabel("Frequency (k)")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()

#Chatgpt

import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of a sequence x[n]
    :param x: Input signal (list or numpy array)
    :return: Frequency domain representation X[k]
    """
    N = len(x)  # Number of samples
    X = np.zeros(N, dtype=complex)  # Initialize output array
    
    # Compute DFT using the formula
    for k in range(N):  # Loop for each frequency component
        for n in range(N):  # Loop for each input sample
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)  # Apply DFT formula
    
    return X

# Example Usage
x = [1, 2, 3, 4, 3, 2, 1, 0]  # Input signal
X = dft(x)  # Compute DFT

# Plot Input Signal (Time Domain)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.stem(x, use_line_collection=True)
plt.xlabel('Time (n)')
plt.ylabel('Amplitude')
plt.title('Input Signal (Time Domain)')

# Plot Magnitude Spectrum (Frequency Domain)
plt.subplot(1, 2, 2)
plt.stem(np.abs(X), use_line_collection=True)
plt.xlabel('Frequency Index (k)')
plt.ylabel('Magnitude')
plt.title('DFT Magnitude Spectrum')

plt.tight_layout()
plt.show()



#Blacbox
import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)  # Initialize the output array
    for k in range(N):  # For each output frequency
        for n in range(N):  # For each input sample
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)  # DFT formula
    return X

# Example usage
if __name__ == "__main__":
    # Input signal (e.g., a simple sine wave)
    t = np.linspace(0, 1, 100, endpoint=False)  # Time variable
    freq = 5  # Frequency of the sine wave
    x = np.sin(2 * np.pi * freq * t)  # Input signal

    # Compute DFT
    X = dft(x)

    # Frequency axis
    freq_axis = np.fft.fftfreq(len(x), d=t[1]-t[0])

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title('Input Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.stem(freq_axis, np.abs(X), use_line_collection=True)
    plt.title('DFT Magnitude Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xlim(-10, 10)  # Limit x-axis for better visibility
    plt.tight_layout()
    plt.show()

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


