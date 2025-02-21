import pandas as pd
import matplotlib.pyplot as plt
import json

"""
Como minimizar o erro:

    1- Escolher um ponto inicial.

    2- Calcular a derivada da função no ponto para obter o declive. O declive indica a direção de
        maior subida.

    3- Mover na direção oposta ao declive (a "descer" no gráfico). O tamanho desse passo é determinado
        pela learning rate do modelo.

    4- Repetir até convergir.

"""

# Function that manually calculates the error of the model
def calculateError(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].km
        y = points.iloc[i].price
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def linearRegression(theta0, theta1, learning_rate, points):
    tmp0 = 0
    tmp1 = 0
    m = len(points)

    for i in range(m):
        mileage = points.iloc[i].km
        price = points.iloc[i].price

        # Calculate the error between the predicted value and the actual value
        error = theta0 + (theta1 * mileage) - price
        tmp0 += error # bias
        tmp1 += error * mileage # gradient

    new_theta0 = theta0 - learning_rate * (tmp0/m)
    new_theta1 = theta1 - learning_rate * (tmp1/m)

    return (new_theta0, new_theta1)



def main():
    data = pd.read_csv('ft_linear_regression/data.csv')

    # Train the model
    theta0, theta1 = linearRegression(theta0, theta1, 0.0001, data)

    # Save the computed values to a json file
    with open("model.json", "w") as file:
        json.dump({"theta0": theta0, "theta1": theta1}, file)

    # Display the data on a graph
    plt.scatter(data.km, data.price)

    # Plot the regression line
    x_values = [min(data.km), max(data.km)]
    y_values = [theta1 * x + theta0 for x in x_values]
    plt.plot(x_values, y_values, color="red")

    plt.show()



if __name__ == "__main__":
    main()

