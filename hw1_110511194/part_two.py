import numpy as np
import matplotlib.pyplot as plt

from part_one import read_csv
from part_one import generate_Phi
from part_one import  generate_phi
from part_one import generate_w
from part_one import generate_y


def generate_std(x, Sn):
    std = np.zeros(len(x))
    for i in range(len(x)):
        phi = generate_phi(x[i], 10)
        std[i] = np.sqrt(1 + np.matmul(np.matmul(phi.T, Sn), phi))
    return std

if __name__ == '__main__':
    data = read_csv(1, 50, 'HW1.csv', delimiter=',')
    x = np.linspace(0, 3, 100)
    Phi = generate_Phi(data, 10)
    w = generate_w(Phi, data[:, 1])
    y = generate_y(x, w, 10)

    n = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    for i in n:
        data_bay = data[:i, :2]
        I = np.eye(10)
        Phi = generate_Phi(data_bay, 10)
        Sn = np.linalg.inv(1e-6 * I + np.dot(np.transpose(Phi),Phi))
        Mn = np.dot(Sn, np.dot(np.transpose(Phi), data_bay[:i, 1]))
        y_bay = generate_y(x, Mn, 10)
        std = generate_std(x, Sn)
        lower_bound = y_bay - std
        upper_bound = y_bay + std
        plt.plot(data_bay[:, 0], data_bay[:, 1], 'bo', markersize=3)
        plt.fill_between(x, lower_bound, upper_bound, color='pink')
        plt.plot(x, y, color='green', label = 'fitting curve')
        plt.plot(x, y_bay, color='red', label = 'fitting curve with {} data'.format(i))
        plt.xlabel('x')
        plt.ylabel('y')
        if i <= 10:
            plt.ylim(-30, 30)
        plt.legend()
        plt.show()
