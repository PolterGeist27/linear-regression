import pandas as pd
from utils import *
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, learning_rate = 0.001, n_iters=1000):

        # Data: file data | x: original km column | y: original price column | X: x normalized | Y: y normalized 
        self.data = pd.read_csv('ft_linear_regression/data.csv')
        self.x = self.data["km"].values
        self.y = self.data["price"].values
        self.X = normalize(self.x)
        self.Y = normalize(self.y)

        # length of data
        self.M = len(self.x)

        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Theta normalized / _Theta denormalized
        self._T0 = 0
        self._T1 = 0
        self.T0 = 1
        self.T1 = 1


    def estimatePrice(self, t0, t1, mileage):
        return (t0 + (t1 * float(mileage)))
    

    def gradient_descent(self):
        print("\033[33m{:s}\033[0m".format('TRAINING MODEL :'))

        for _ in range(self.n_iters):
            sum1 = 0
            sum2 = 0
            for i in range(self.M):
                T = self.T0 + self.T1 * self.X[i] - self.Y[i]
                sum1 += T
                sum2 += T * self.X[i]

            self.T0 = self.T0 - self.learning_rate * (sum1 / self.M)
            self.T1 = self.T1 - self.learning_rate * (sum2 / self.M)

        self.display_graph()

        
    def display_graph(self):
        plt.plot(self.x, self.y, 'ro', label='data')
        x_estim = self.x
        y_estim = [denormalizeElem(self.y, self.estimatePrice(self.T0, self.T1, normalizeElem(self.x, _))) for _ in
                   x_estim]
        plt.plot(x_estim, y_estim, 'g-', label='Estimation')
        plt.ylabel('Price (€)')
        plt.xlabel('Mileage (km)')
        plt.title('Price = f(Mileage)')
        plt.show()


def main():
    ftlr = LinearRegression()
    ftlr.gradient_descent()


if __name__ == "__main__":
    main()