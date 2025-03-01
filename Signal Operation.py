import numpy as np
import matplotlib.pyplot as plt

# Define the original signal
n = np.arange(-7, 8)  # Time indices
x = np.array([2, 3, 1, 5, 7, 6, 4, 2, 1, 3, 2, 6, 3, 4, 6])  # Signal values

# Time Shifting
shift_right = np.roll(x, 2)  # Right shift (delay)
shift_left = np.roll(x, -2)  # Left shift (advance)

# Time Scaling (Compression & Expansion)
scale_compress = x[::2]  # Every second sample (Compression)
n_compress = n[::2]  # Adjusted time index

scale_expand = np.repeat(x, 2)  # Duplicate samples (Expansion)
n_expand = np.linspace(n[0], n[-1], len(scale_expand))  # Adjusted time index

# Time Folding (Reversal)
folded_signal = x[::-1]

# Arithmetic Operations
y = np.array([1, 2, 0, 1, 4, 3, 5, 2, 1, 0, 1, 5, 6, 7, 3])  # Another signal
add_signal = x + y  # Addition
sub_signal = x - y  # Subtraction
mul_signal = x * y  # Multiplication
div_signal = np.divide(x, y, out=np.zeros_like(x, dtype=float), where=y != 0)  # Avoid divide by zero

# Plot Signals
plt.figure(figsize=(12, 10))  # Increased figure size to accommodate 10 subplots

# Original Signal
plt.subplot(4, 3, 1)
plt.stem(n, x, basefmt="k", use_line_collection=True)
plt.title("Original Signal x(n)")

# Time Shifting
plt.subplot(4, 3, 2)
plt.stem(n, shift_right, basefmt="k", use_line_collection=True)
plt.title("Right Shift (x(n-2))")

plt.subplot(4, 3, 3)
plt.stem(n, shift_left, basefmt="k", use_line_collection=True)
plt.title("Left Shift (x(n+2))")

# Time Scaling
plt.subplot(4, 3, 4)
plt.stem(n_compress, scale_compress, basefmt="k", use_line_collection=True)
plt.title("Compressed Signal (x(n) downsampled)")

plt.subplot(4, 3, 5)
plt.stem(n_expand, scale_expand, basefmt="k", use_line_collection=True)
plt.title("Expanded Signal (x(n) upsampled)")

# Time Folding
plt.subplot(4, 3, 6)
plt.stem(n, folded_signal, basefmt="k", use_line_collection=True)
plt.title("Folded Signal x(-n)")

# Arithmetic Operations
plt.subplot(4, 3, 7)
plt.stem(n, add_signal, basefmt="k", use_line_collection=True)
plt.title("Addition: x(n) + y(n)")

plt.subplot(4, 3, 8)
plt.stem(n, sub_signal, basefmt="k", use_line_collection=True)
plt.title("Subtraction: x(n) - y(n)")

plt.subplot(4, 3, 9)
plt.stem(n, mul_signal, basefmt="k", use_line_collection=True)
plt.title("Multiplication: x(n) * y(n)")

# Division plot (added the 10th subplot)
plt.subplot(4, 3, 10)
plt.stem(n, div_signal, basefmt="k", use_line_collection=True)
plt.title("Division: x(n) / y(n)")

plt.tight_layout()
plt.show()
