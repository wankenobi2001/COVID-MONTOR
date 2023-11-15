import numpy as np
import matplotlib.pyplot as plt


class SquareLine:
    alpha = False
    iteration = False
    x = []
    y = []
    w = [0, 0]

    error = []
    MSE = np.array([])

    def __init__(self, points, alpha=0.01, iteration=100):
        self.x = [points[x][0] for x in range(len(points))]
        self.y = [points[x][1] for x in range(len(points))]

        self.alpha = alpha
        self.iteration = iteration

    def fit(self):
        for i in range(0, self.iteration):
            y_pred = np.array([])
            error = np.array([])
            error_x = np.array([])

            for x in range(len(self.x)):
                y_pred = np.append(y_pred, self.w[0] + self.w[1] * self.x[x])

            error = np.append(error, y_pred - self.y)
            error_x = np.append(error_x, error * x)
            self.MSE = np.append(self.MSE, (error ** 2).mean())
            self.w[0] = self.w[0] - self.alpha * np.sum(error)
            self.w[1] = self.w[1] - self.alpha * np.sum(error_x)

            if self.MSE[i] > self.MSE[i - 1]:
                break

        return self.w


fit_line = SquareLine(points=[[2, 4], [4, 2]])
fit_line.fit()

print("W values = {}".format(fit_line.w))

pred = np.array([fit_line.w[0] + fit_line.w[1]*fit_line.x[i] for i in range(len(fit_line.y))])
plt.scatter(fit_line.x, fit_line.y)
plt.plot(fit_line.x, pred)
plt.title("Linear Fit")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()


