import json
import os

# Function to load model parameters safely
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

# Get the values from the model
theta0, theta1 = loadModel()

# Calculate the predicted price based on the model's values
def estimatePrice(mileage):
    return (theta0 + theta1 * mileage)

try:
    mileage = float(input("Enter mileage: "))
    print(f"Estimaded price: {estimatePrice(mileage)}")
except ValueError:
    print("Invalid input, please enter a numerical value.")

