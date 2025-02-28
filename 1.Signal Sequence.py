#signal sequence
#Deepseek
import numpy as np
import matplotlib.pyplot as plt

# Generate a signal sequence
def generate_signal_sequence(frequency, sampling_rate, duration):
    t = np.arange(0, duration, 1/sampling_rate)  # Time vector
    signal = np.sin(2 * np.pi * frequency * t)  # Sine wave signal
    return t, signal

# Parameters
frequency = 5  # Frequency of the signal (Hz)
sampling_rate = 100  # Sampling rate (samples per second)
duration = 1  # Duration of the signal (seconds)

# Generate signal sequence
t, signal = generate_signal_sequence(frequency, sampling_rate, duration)

# Output
print("Time Vector (t):", t)
print("Signal Sequence (x[n]):", signal)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, signal, marker='o', label="Signal Sequence")
plt.title("Signal Sequence (Discrete Sine Wave)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

#chatgpt
import numpy as np
import matplotlib.pyplot as plt

# Define the range for discrete-time signals
n = np.arange(-10, 20)

# 1. Unit Impulse Signal δ(n)
impulse_signal = np.where(n == 0, 1, 0)

# 2. Unit Step Signal u(n)
step_signal = np.where(n >= 0, 1, 0)

# 3. Ramp Signal r(n)
ramp_signal = np.where(n >= 0, n, 0)

# 4. Sinusoidal Signal x(n) = A sin(ωn + φ)
A = 1  # Amplitude
omega = np.pi / 8  # Frequency
phi = 0  # Phase shift
sinusoidal_signal = A * np.sin(omega * n + phi)

# 5. Exponential Signal x(n) = A e^(αn)
alpha = 0.2
exponential_signal = np.exp(alpha * n)

# Plotting the signals
plt.figure(figsize=(12, 8))

# Impulse Signal
plt.subplot(3, 2, 1)
plt.stem(n, impulse_signal, basefmt="k", use_line_collection=True)
plt.title("Unit Impulse Signal δ(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

# Step Signal
plt.subplot(3, 2, 2)
plt.stem(n, step_signal, basefmt="k", use_line_collection=True)
plt.title("Unit Step Signal u(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

# Ramp Signal
plt.subplot(3, 2, 3)
plt.stem(n, ramp_signal, basefmt="k", use_line_collection=True)
plt.title("Ramp Signal r(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

# Sinusoidal Signal
plt.subplot(3, 2, 4)
plt.stem(n, sinusoidal_signal, basefmt="k", use_line_collection=True)
plt.title("Sinusoidal Signal x(n) = A sin(ωn + φ)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

# Exponential Signal
plt.subplot(3, 2, 5)
plt.stem(n, exponential_signal, basefmt="k", use_line_collection=True)
plt.title("Exponential Signal x(n) = A e^(αn)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

#blacbox
import numpy as np
import matplotlib.pyplot as plt

# Function to generate a sinusoidal signal
def generate_sinusoidal(frequency, amplitude, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, signal

# Function to generate a square wave signal
def generate_square_wave(frequency, amplitude, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    return t, signal

# Function to generate a triangular wave signal
def generate_triangular_wave(frequency, amplitude, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = amplitude * (2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1)
    return t, signal

# Function to generate random noise
def generate_random_noise(duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    noise = np.random.normal(0, 0.5, t.shape)  # Mean 0, std deviation 0.5
    return t, noise

# Main function
if __name__ == "__main__":
    fs = 1000  # Sampling frequency
    duration = 2  # Duration in seconds
    frequency = 5  # Frequency for sinusoidal and square wave
    amplitude = 1  # Amplitude

    # Generate signals
    t_sin, sin_signal = generate_sinusoidal(frequency, amplitude, duration, fs)
    t_sq, sq_signal = generate_square_wave(frequency, amplitude, duration, fs)
    t_tri, tri_signal = generate_triangular_wave(frequency, amplitude, duration, fs)
    t_noise, noise_signal = generate_random_noise(duration, fs)

    # Plotting the results
    plt.figure(figsize=(12, 10))

    # Sinusoidal Signal
    plt.subplot(4, 1, 1)
    plt.plot(t_sin, sin_signal, label='Sinusoidal Signal', color='blue')
    plt.title('Sinusoidal Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()

    # Square Wave Signal
    plt.subplot(4, 1, 2)
    plt.plot(t_sq, sq_signal, label='Square Wave Signal', color='green')
    plt.title('Square Wave Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()

    # Triangular Wave Signal
    plt.subplot(4, 1, 3)
    plt.plot(t_tri, tri_signal, label='Triangular Wave Signal', color='orange')
    plt.title('Triangular Wave Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()

    # Random Noise Signal
    plt.subplot(4, 1, 4
