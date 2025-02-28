#signal operation -shifting,scaling,folding and arithmatic 
#deepseek

import numpy as np
import matplotlib.pyplot as plt

# Generate a signal
def generate_signal():
    t = np.linspace(0, 1, 100)  # Time vector
    signal = np.sin(2 * np.pi * 5 * t)  # Signal: Sine wave (5 Hz)
    return t, signal

# Signal operations
def shift_signal(signal, shift):
    return np.roll(signal, shift)  # Shift the signal

def scale_signal(signal, scale_factor):
    return signal * scale_factor  # Amplitude scaling

def fold_signal(signal):
    return signal[::-1]  # Time reversal (folding)

def add_signals(signal1, signal2):
    return signal1 + signal2  # Addition of two signals

# Parameters
t, signal = generate_signal()
shift = 20  # Shift by 20 samples
scale_factor = 2  # Amplitude scaling factor
signal2 = np.cos(2 * np.pi * 5 * t)  # Second signal for arithmetic operations

# Perform operations
shifted_signal = shift_signal(signal, shift)
scaled_signal = scale_signal(signal, scale_factor)
folded_signal = fold_signal(signal)
added_signal = add_signals(signal, signal2)

# Output
print("Original Signal:", signal)
print("Shifted Signal:", shifted_signal)
print("Scaled Signal:", scaled_signal)
print("Folded Signal:", folded_signal)
print("Added Signal:", added_signal)

# Plotting
plt.figure(figsize=(12, 10))

# Plot original signal
plt.subplot(4, 1, 1)
plt.plot(t, signal, label="Original Signal (5 Hz Sine Wave)")
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Plot shifted signal
plt.subplot(4, 1, 2)
plt.plot(t, shifted_signal, label="Shifted Signal", color="orange")
plt.title("Shifted Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Plot scaled signal
plt.subplot(4, 1, 3)
plt.plot(t, scaled_signal, label="Scaled Signal", color="green")
plt.title("Scaled Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Plot folded signal
plt.subplot(4, 1, 4)
plt.plot(t, folded_signal, label="Folded Signal", color="red")
plt.title("Folded Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

# Plot arithmetic operation (addition)
plt.figure(figsize=(12, 4))
plt.plot(t, signal, label="Signal 1 (Sine Wave)")
plt.plot(t, signal2, label="Signal 2 (Cosine Wave)")
plt.plot(t, added_signal, label="Added Signal", color="purple")
plt.title("Arithmetic Operation: Addition of Two Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()

#chatgpt
import numpy as np
import matplotlib.pyplot as plt

# Define the original signal
n = np.arange(-5, 6)  # Time indices
x = np.array([2, 3, 1, 5, 7, 6, 4, 2, 1, 3, 2])  # Signal values

# Time Shifting
shift_right = np.roll(x, 2)  # Right shift (delay)
shift_left = np.roll(x, -2)  # Left shift (advance)

# Time Scaling (Compression & Expansion)
scale_compress = x[::2]  # Every second sample (Compression)
scale_expand = np.repeat(x, 2)  # Duplicate samples (Expansion)

# Time Folding (Reversal)
folded_signal = x[::-1]

# Arithmetic Operations
y = np.array([1, 2, 0, 1, 4, 3, 5, 2, 1, 0, 1])  # Another signal
add_signal = x + y  # Addition
sub_signal = x - y  # Subtraction
mul_signal = x * y  # Multiplication
div_signal = np.divide(x, y, out=np.zeros_like(x, dtype=float), where=y!=0)  # Division (Avoid divide by zero)

# Plot Signals
plt.figure(figsize=(12, 8))

# Original Signal
plt.subplot(3, 3, 1)
plt.stem(n, x, basefmt="k", use_line_collection=True)
plt.title("Original Signal x(n)")

# Time Shifting
plt.subplot(3, 3, 2)
plt.stem(n, shift_right, basefmt="k", use_line_collection=True)
plt.title("Right Shift (x(n-2))")

plt.subplot(3, 3, 3)
plt.stem(n, shift_left, basefmt="k", use_line_collection=True)
plt.title("Left Shift (x(n+2))")

# Time Folding
plt.subplot(3, 3, 4)
plt.stem(n, folded_signal, basefmt="k", use_line_collection=True)
plt.title("Folded Signal x(-n)")

# Arithmetic Operations
plt.subplot(3, 3, 5)
plt.stem(n, add_signal, basefmt="k", use_line_collection=True)
plt.title("Addition: x(n) + y(n)")

plt.subplot(3, 3, 6)
plt.stem(n, sub_signal, basefmt="k", use_line_collection=True)
plt.title("Subtraction: x(n) - y(n)")

plt.subplot(3, 3, 7)
plt.stem(n, mul_signal, basefmt="k", use_line_collection=True)
plt.title("Multiplication: x(n) * y(n)")

plt.subplot(3, 3, 8)
plt.stem(n, div_signal, basefmt="k", use_line_collection=True)
plt.title("Division: x(n) / y(n)")

plt.tight_layout()
plt.show()


#blacbox
import numpy as np
import matplotlib.pyplot as plt

# Generate a simple signal (e.g., a unit step function)
def generate_signal(n):
    return np.where(n >= 0, 1, 0)

# Shifting operation
def shift_signal(x, n0):
    return np.roll(x, n0)

# Scaling operation
def scale_signal(x, a):
    return a * x

# Folding operation
def fold_signal(x):
    return x[::-1]

# Arithmetic operation
def add_signals(x, h):
    return x + h

# Main function
if __name__ == "__main__":
    # Define the time vector
    n = np.arange(-5, 10, 1)  # Time index

    # Generate the original signal
    x = generate_signal(n)

    # Perform operations
    shifted_x = shift_signal(x, 2)  # Shift right by 2
    scaled_x = scale_signal(x, 2)    # Scale by a factor of 2
    folded_x = fold_signal(x)         # Fold the signal
    added_x = add_signals(x, shifted_x)  # Add original and shifted signals

    # Plotting the results
    plt.figure(figsize=(12, 10))

    # Original Signal
    plt.subplot(5, 1, 1)
    plt.stem(n, x, use_line_collection=True)
    plt.title('Original Signal (x[n])')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid()

    # Shifted Signal
    plt.subplot(5, 1, 2)
    plt.stem(n, shifted_x, use_line_collection=True)
    plt.title('Shifted Signal (x[n-2])')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid()

    # Scaled Signal
    plt.subplot(5, 1, 3)
    plt.stem(n, scaled_x, use_line_collection=True)
    plt.title('Scaled Signal (2 * x[n])')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid()

    # Folded Signal
    plt.subplot(5, 1, 4)
    plt.stem(n, folded_x, use_line_collection=True)
    plt.title('Folded Signal (x[-n])')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid()

    # Added Signal
    plt.subplot(5, 1, 5)
    plt.stem(n, added_x, use_line_collection=True)
    plt.title('Added Signal (x[n] + x[n-2])')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid()
