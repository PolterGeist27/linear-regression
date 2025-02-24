import numpy as np
import json
import os
import pandas as pd

# Function to calculate MAE (Mean Absolute Error)
def mean_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)

# Function to calculate MSE (Mean Squared Error)
def mean_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)

# Function to calculate RMSE (Root Mean Squared Error)
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to calculate R² Score (Coefficient of Determination)
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


def loadModel(filename="model.json"):
    if not os.path.exists(filename):
        print("Model not found, using default values.")
        return (0, 0)

    try:
        with open("model.json", "r") as file:
            model = json.load(file)

        theta0 = model.get("theta0", 0);
        theta1 = model.get("theta1", 0);
    
        return (theta0, theta1)
    except (json.JSONDecodeError, ValueError):
        print("Error reading model, using default values.")
        return (0, 0)


# Load data
data = pd.read_csv("ft_linear_regression/data.csv")

# Get the values from the model
theta0, theta1 = loadModel()

y_values = data["price"].values
predictions = theta0 + theta1 * data["km"]
 
mae = mean_absolute_error(y_values, predictions) # Average error
mse = mean_squared_error(y_values, predictions) # Average error squared (larger errors are more penalized than)
rmse = root_mean_squared_error(y_values, predictions) # The error in the same units as the price
r2 = r2_score(y_values, predictions) # Closer to one the better the model

print(f"📊 Model Evaluation Metrics (Manual Calculation):")
print(f"MAE  = {mae:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R² Score = {r2:.4f}")
