import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Load data
data = pd.read_csv("ft_linear_regression/data.csv")
X = data["km"].values
y = data["price"].values

# Feature scaling (Normalization)
X_mean, X_std = np.mean(X), np.std(X)
y_mean, y_std = np.mean(y), np.std(y)

X = (X - X_mean) / X_std  # Normalize mileage
y = (y - y_mean) / y_std  # Normalize price

# Initialize parameters
theta0 = 0
theta1 = 0
learning_rate = 0.001
iterations = 10000
m = len(y)

# Gradient Descent with convergence check
for i in range(iterations):
    predictions = theta0 + theta1 * X
    error = predictions - y

    # Compute gradients
    grad_theta0 = (1/m) * np.sum(error)
    grad_theta1 = (1/m) * np.sum(error * X)

    # Update theta
    theta0 -= learning_rate * grad_theta0
    theta1 -= learning_rate * grad_theta1

    # Check for NaN or divergence
    if np.isnan(theta0) or np.isnan(theta1):
        print(f"Overflow at iteration {i}, reducing learning rate...")
        learning_rate /= 10  # Reduce learning rate
        theta0, theta1 = 0, 0  # Reset
        i = 0  # Restart training

# Convert theta0 and theta1 back to original scale
theta1 = theta1 * (y_std / X_std)
theta0 = (y_mean - theta1 * X_mean)

# Save model
with open("model.json", "w") as f:
    json.dump({"theta0": theta0, "theta1": theta1}, f)
    

print(f"Training complete: theta0 = {theta0}, theta1 = {theta1}")

# Plot Data
plt.scatter(data["km"], data["price"], color="blue", label="Actual Data")
plt.plot(data["km"], theta0 + theta1 * data["km"], color="red", label="Regression Line")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.legend()
plt.show()
