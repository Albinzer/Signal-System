#correlation 
#deepseek

import numpy as np
import matplotlib.pyplot as plt

# Generate two signals
def generate_signals():
    t = np.linspace(0, 1, 100)  # Time vector
    signal1 = np.sin(2 * np.pi * 5 * t)  # Signal 1: Sine wave (5 Hz)
    signal2 = np.sin(2 * np.pi * 5 * t + np.pi / 4)  # Signal 2: Phase-shifted sine wave
    return t, signal1, signal2

# Compute Pearson correlation
def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

# Compute cross-correlation
def cross_correlation(x, y):
    return np.correlate(x, y, mode='full')

# Parameters
t, signal1, signal2 = generate_signals()

# Compute Pearson correlation
pearson_corr = pearson_correlation(signal1, signal2)

# Compute cross-correlation
cross_corr = cross_correlation(signal1, signal2)
lags = np.arange(-len(signal1) + 1, len(signal1))  # Time lags

# Output
print("Pearson Correlation Coefficient:", pearson_corr)
print("Cross-Correlation:", cross_corr)

# Plotting
plt.figure(figsize=(12, 8))

# Plot signals
plt.subplot(3, 1, 1)
plt.plot(t, signal1, label="Signal 1 (5 Hz Sine Wave)")
plt.plot(t, signal2, label="Signal 2 (Phase-Shifted Sine Wave)")
plt.title("Input Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Plot Pearson correlation
plt.subplot(3, 1, 2)
plt.bar(['Pearson Correlation'], [pearson_corr], color='blue')
plt.title("Pearson Correlation Coefficient")
plt.ylabel("Correlation Value")

# Plot cross-correlation
plt.subplot(3, 1, 3)
plt.plot(lags, cross_corr, label="Cross-Correlation", color='green')
plt.title("Cross-Correlation")
plt.xlabel("Time Lag")
plt.ylabel("Correlation Value")
plt.legend()

plt.tight_layout()
plt.show()

#chatgpt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.signal import correlate

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)  # Independent variable
y = 2 * x + np.random.normal(0, 2, 100)  # Correlated variable with noise
z = np.sin(x) + np.random.normal(0, 0.2, 100)  # Non-linearly related variable

# Pearson Correlation
pearson_corr, _ = pearsonr(x, y)

# Spearman Correlation
spearman_corr, _ = spearmanr(x, z)

# Kendall Correlation
kendall_corr, _ = kendalltau(x, z)

# Cross-Correlation (between y and z)
cross_corr = correlate(y - np.mean(y), z - np.mean(z), mode='full')

# Print correlation values
print(f"Pearson Correlation (x, y): {pearson_corr:.2f}")
print(f"Spearman Correlation (x, z): {spearman_corr:.2f}")
print(f"Kendall Correlation (x, z): {kendall_corr:.2f}")

# Plot Correlation Matrix
data = pd.DataFrame({'x': x, 'y': y, 'z': z})
corr_matrix = data.corr()

plt.figure(figsize=(12, 5))

# Scatter Plot of x vs y
plt.subplot(1, 3, 1)
plt.scatter(x, y, color='blue', alpha=0.6)
plt.title(f"Scatter Plot (Pearson: {pearson_corr:.2f})")
plt.xlabel("x")
plt.ylabel("y")

# Scatter Plot of x vs z
plt.subplot(1, 3, 2)
plt.scatter(x, z, color='red', alpha=0.6)
plt.title(f"Scatter Plot (Spearman: {spearman_corr:.2f})")
plt.xlabel("x")
plt.ylabel("z")

# Cross-Correlation Plot
plt.subplot(1, 3, 3)
plt.plot(cross_corr, color='green')
plt.title("Cross-Correlation of y & z")
plt.xlabel("Lag")
plt.ylabel("Correlation")

plt.tight_layout()
plt.show()

# Heatmap of Correlation Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

#blacbox
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
def generate_data(n=100):
    np.random.seed(0)
    x = np.random.rand(n)
    y = 2 * x + np.random.normal(0, 0.1, n)  # Positive correlation
    return x, y

# Calculate correlation coefficient
def calculate_correlation(x, y):
    correlation_matrix = np.corrcoef(x, y)
    return correlation_matrix[0, 1]  # Return the correlation coefficient

# Main function
if __name__ == "__main__":
    # Generate synthetic data
    x, y = generate_data(n=100)

    # Calculate correlation
    correlation_coefficient = calculate_correlation(x, y)
    print(f"Correlation Coefficient: {correlation_coefficient:.2f}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', alpha=0.6)
    plt.title('Scatter Plot of X and Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()

    # Add a regression line
    sns.regplot(x=x, y=y, scatter=False, color='red', label='Regression Line')
    plt.legend()
    plt.show()

