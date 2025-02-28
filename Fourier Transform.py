#fourier transform
#Deepseek
import numpy as np
import matplotlib.pyplot as plt

# Input: Time-domain signal (e.g., a sine wave)
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector (1 second)
f1 = 50  # Frequency of the sine wave (50 Hz)
x = np.sin(2 * np.pi * f1 * t)  # Input signal (sine wave)

# Compute Fourier Transform using FFT
X = np.fft.fft(x)  # FFT of the signal
frequencies = np.fft.fftfreq(len(x), d=1/fs)  # Frequency bins

# Output: Frequency-domain representation
print("Input Signal (x):", x)
print("FFT Output (X):", X)

# Plotting
plt.figure(figsize=(12, 8))

# Plot input signal
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title("Input Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot FFT output (Magnitude)
plt.subplot(3, 1, 2)
plt.stem(frequencies, np.abs(X), use_line_collection=True)
plt.title("FFT Output (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

# Plot FFT output (Phase)
plt.subplot(3, 1, 3)
plt.stem(frequencies, np.angle(X), use_line_collection=True)
plt.title("FFT Output (Phase)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")

plt.tight_layout()
plt.show()

#chatgpt
import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of a sequence x[n].
    :param x: Input signal (list or numpy array)
    :return: Frequency domain representation X[k]
    """
    N = len(x)  # Number of samples
    X = np.zeros(N, dtype=complex)  # Initialize output array
    
    for k in range(N):  # Loop for each frequency component
        for n in range(N):  # Loop for each input sample
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)  # Apply DFT formula
    
    return X

# Generate a sample signal (sum of 2 sine waves)
Fs = 100  # Sampling Frequency
T = 1  # Total Time
t = np.linspace(0, T, Fs, endpoint=False)  # Time array
freq1, freq2 = 5, 20  # Two frequencies (5Hz and 20Hz)
x = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)  # Signal

# Compute DFT and FFT
X_dft = dft(x)  # Using manual DFT function
X_fft = np.fft.fft(x)  # Using NumPy's optimized FFT

# Frequency axis
freqs = np.fft.fftfreq(len(x), 1/Fs)

# Plot the Input Signal
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Input Signal (Time Domain)')

# Plot Magnitude Spectrum (DFT)
plt.subplot(1, 2, 2)
plt.stem(freqs[:Fs//2], np.abs(X_fft)[:Fs//2], use_line_collection=True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Fourier Transform (Frequency Domain)')

plt.tight_layout()
plt.show()


#blacbox
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(signal, dt):
    N = len(signal)
    # Frequency axis
    freq = np.fft.fftfreq(N, d=dt)
    # Compute the Fourier Transform using FFT
    F = np.fft.fft(signal)
    return freq, F

# Example usage
if __name__ == "__main__":
    # Time variable
    dt = 0.01  # Time step
    t = np.arange(0, 1, dt)  # Time vector

    # Input signal: a combination of two sine waves
    f1 = 5  # Frequency of first sine wave
    f2 = 20  # Frequency of second sine wave
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

    # Compute Fourier Transform
    freq, F = fourier_transform(signal, dt)

    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Plot the input signal
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Input Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Plot the magnitude of the Fourier Transform
    plt.subplot(2, 1, 2)
    plt.plot(freq, np.abs(F))
    plt.title('Fourier Transform Magnitude Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xlim(0, 30)  # Limit x-axis for better visibility
    plt.tight_layout()
    plt.show()

