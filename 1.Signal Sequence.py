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
