import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def linear_regression(x_list, y_list):

    # solve least square linear regression:
    #   for x1, ..., xn
    #   linear regression is y = mx + b
    #   where
    #     m = { n Sigma(xiyi) - Sigma(xi)Sigma(yi) }
    #           / { n Sigma(xi^2) - (Sigma(xi))^2 }

    #     b = { Sigma(xi^2) Sigma(yi) - Sigma(xiyi)Sigma(xi) / }
    #           / { n Sigma(xi^2) - (Sigma(xi))^2 }
    assert len(x_list) == len(y_list)
    _N = len(x_list)
    sigma_xi = sum(x_list)
    sigma_yi = sum(y_list)
    sigma_xiyi = sum(x_list[i] * y_list[i] for i in range(_N))
    sigma_xi_square = sum(x * x for x in x_list)

    m = (_N*sigma_xiyi - sigma_xi*sigma_yi) \
        / (_N*sigma_xi_square - sigma_xi**2)
    b = (sigma_xi_square*sigma_yi - sigma_xiyi*sigma_xi) \
        / (_N*sigma_xi_square - sigma_xi**2)

    return (m, b)


X_LABEL = "Apperant Temperature Difference (C)"
Y_LABEL = "Wind Speed (km/h)"


def read_weather_history():
    # "Temperature (C)": x
    # "Wind Speed (km/h)": y
    frame = pd.read_csv("weatherHistory.csv")
    # x_data = frame[X_LABEL]
    x_data = frame["Apparent Temperature (C)"] - frame["Temperature (C)"]
    y_data = frame[Y_LABEL]
    return (x_data, y_data)


# x_coord = [n*2 for n in range(N)]
# y_coord = [random.randint(1, 10) for n in range(N)]
x_coord, y_coord = read_weather_history()
assert len(x_coord) == len(y_coord)
N = len(x_coord)
m, b = linear_regression(x_coord, y_coord)
y_pred = [m*x_coord[i]+b for i in range(N)]

error = sum((y_pred[i] - y_coord[i]) ** 2 for i in range(N))
# print("Dataset:", list(zip(x_coord, y_coord)))
print("Linear Regression:", "y = %fx + %f" % (m, b))
print("Error:", error)
plt.scatter(x_coord, y_coord, s=1, alpha=0.1)
plt.plot(x_coord, y_pred, color='red')
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.title("Linear Regression: %s vs. %s" % (X_LABEL, Y_LABEL))
plt.show()
