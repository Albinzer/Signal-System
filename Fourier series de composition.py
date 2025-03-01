#fourier series de composition 

import numpy as np
import matplotlib.pyplot as plt

# Input: Define a periodic signal (e.g., square wave)
def square_wave(t, T):
    return np.where(np.sin(2 * np.pi * t / T) > 0, 1, -1)

# Parameters
T = 2 * np.pi  # Period of the signal
t = np.linspace(0, 2 * T, 1000)  # Time vector
x = square_wave(t, T)  # Input signal (square wave)

# Fourier Series Decomposition
def fourier_series(x, t, T, n_terms):
    a0 = (2 / T) * np.trapz(x, t)  # DC component
    a = []  # Cosine coefficients
    b = []  # Sine coefficients
    for n in range(1, n_terms + 1):
        an = (2 / T) * np.trapz(x * np.cos(2 * np.pi * n * t / T), t)
        bn = (2 / T) * np.trapz(x * np.sin(2 * np.pi * n * t / T), t)
        a.append(an)
        b.append(bn)
    return a0, a, b

# Reconstruct the signal using Fourier Series
def reconstruct_signal(t, T, a0, a, b):
    x_reconstructed = a0 / 2
    for n in range(1, len(a) + 1):
        x_reconstructed += a[n - 1] * np.cos(2 * np.pi * n * t / T) + b[n - 1] * np.sin(2 * np.pi * n * t / T)
    return x_reconstructed

# Number of terms in the Fourier Series
n_terms = 10

# Compute Fourier Series coefficients
a0, a, b = fourier_series(x, t, T, n_terms)

# Reconstruct the signal
x_reconstructed = reconstruct_signal(t, T, a0, a, b)

# Output: Fourier Series coefficients
print("DC Component (a0):", a0)
print("Cosine Coefficients (a_n):", a)
print("Sine Coefficients (b_n):", b)

# Plotting
plt.figure(figsize=(12, 8))

# Plot original signal
plt.subplot(2, 1, 1)
plt.plot(t, x, label="Original Signal")
plt.title("Original Signal (Square Wave)")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.legend()

# Plot reconstructed signal
plt.subplot(2, 1, 2)
plt.plot(t, x_reconstructed, label="Reconstructed Signal", color="red")
plt.title(f"Reconstructed Signal (Fourier Series, {n_terms} Terms)")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

#chatgpt

import numpy as np
import matplotlib.pyplot as plt

# Define the periodic function: Square wave
def square_wave(x, T):
    return np.where(np.sin(2 * np.pi * x / T) >= 0, 1, -1)

# Compute Fourier Series approximation
def fourier_series_approx(x, T, N):
    a0 = 0  # DC component for odd function
    f_approx = np.full_like(x, a0)  # Initialize with DC component
    
    for n in range(1, N + 1, 2):  # Only odd harmonics for square wave
        bn = (4 / (n * np.pi))  # Fourier coefficient for sine terms
        f_approx += bn * np.sin(2 * np.pi * n * x / T)  # Add sine terms
    
    return f_approx

# Parameters
T = 2 * np.pi  # Period
x = np.linspace(-2 * T, 2 * T, 1000)  # x values

# Compute original function and Fourier approximations
y_original = square_wave(x, T)
N_terms = [1, 3, 5, 10, 50]  # Number of terms in Fourier series

# Plot results
plt.figure(figsize=(12, 8))

for i, N in enumerate(N_terms):
    plt.subplot(3, 2, i + 1)
    y_approx = fourier_series_approx(x, T, N)
    plt.plot(x, y_original, 'k--', label="Original Function")
    plt.plot(x, y_approx, 'r', label=f"N={N}")
    plt.title(f"Fourier Approximation with {N} Terms")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

#blacbox
import numpy as np
import matplotlib.pyplot as plt

def fourier_series(f, T, N, t):
    # Calculate a0
    a0 = (1/T) * np.trapz(f(t), t)
    
    # Initialize coefficients
    a_n = np.zeros(N)
    b_n = np.zeros(N)
    
    # Calculate an and bn coefficients
    for n in range(1, N + 1):
        a_n[n - 1] = (2/T) * np.trapz(f(t) * np.cos(2 * np.pi * n * t / T), t)
        b_n[n - 1] = (2/T) * np.trapz(f(t) * np.sin(2 * np.pi * n * t / T), t)
    
    # Construct the Fourier series
    f_approx = a0 / 2  # Start with a0/2
    for n in range(1, N + 1):
        f_approx += a_n[n - 1] * np.cos(2 * np.pi * n * t / T) + b_n[n - 1] * np.sin(2 * np.pi * n * t / T)
    
    return f_approx

# Example usage
if __name__ == "__main__":
    # Define the periodic function (square wave)
    T = 2 * np.pi  # Period
    t = np.linspace(0, T, 1000)  # Time vector
    f = lambda t: 1 if (t % T) < (T / 2) else -1  # Square wave function

    # Number of terms in the Fourier series
    N = 10

    # Compute Fourier series approximation
    f_approx = fourier_series(f, T, N, t)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(t, f(t), label='Original Function (Square Wave)', color='blue')
    plt.plot(t, f_approx, label='Fourier Series Approximation', color='red', linestyle='--')
    plt.title('Fourier Series Decomposition')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.xlim(0, T)
    plt.tight_layout()
    plt.show()

#rafi

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the periodic function
def f(x):
    return np.sign(np.sin(x))  # Change this to any periodic function

# Define the period
L = np.pi  # Half-period

# Compute Fourier coefficients
def a0():
    return (1 / L) * quad(f, -L, L)[0]

def an(n):
    return (1 / L) * quad(lambda x: f(x) * np.cos(n * np.pi * x / L), -L, L)[0]

def bn(n):
    return (1 / L) * quad(lambda x: f(x) * np.sin(n * np.pi * x / L), -L, L)[0]

# Fourier series approximation
def fourier_series(x, N):
    sum_series = a0() / 2
    for n in range(1, N + 1):
        sum_series += an(n) * np.cos(n * np.pi * x / L) + bn(n) * np.sin(n * np.pi * x / L)
    return sum_series

# Increase number of points for smoothness
x_vals = np.linspace(-L, L, 1000)  
y_original = np.vectorize(f)(x_vals)

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_original, label="Original Function", color='black', linewidth=2)

# Define strong contrast colors
colors = ['r', 'b', 'g', 'm']
linestyles = ['-', '--', '-.', ':']
alphas = [0.9, 0.7, 0.5, 0.4]  # Different transparency levels

# Fourier approximations for different N
N_values = [1, 5, 10, 20]
for i, N in enumerate(N_values):
    y_fourier = np.vectorize(lambda x: fourier_series(x, N))(x_vals)
    plt.plot(x_vals, y_fourier, label=f"Fourier Approx (N={N})", 
             color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], alpha=alphas[i])

plt.legend()
plt.title("Improved Fourier Series Decomposition")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

