import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def read_csv(front, rows, filename, delimiter=','):
    data = np.genfromtxt(filename, delimiter=delimiter, skip_header=front, max_rows=rows)
    return data


def generate_Phi(data, M):
    phi = np.zeros((len(data), M))
    for i in range(0, len(data)):
        phi[i, 0] = 1
        for j in range(1, M):
            mu = 3 * j / M
            phi[:, j] = 1 / (1 + np.exp(-(data[:, 0] - mu) / 0.1))
    return phi


def generate_w(phi, t):
    w = np.dot(np.linalg.pinv(phi), t)
    return w


def generate_w_revised(Phi, t, lmbda):
    N, M = Phi.shape
    I = np.eye(M)
    w = np.linalg.inv(lmbda * I + np.transpose(Phi) @ Phi) @ np.transpose(Phi) @ t
    return w


def generate_phi(x, M):
    phi = np.zeros((M,))
    phi[0] = 1
    for j in range(1, M):
        mu = 3 * j / M
        phi[j] = 1 / (1 + np.exp(-(x - mu) / 0.1))
    return phi


def generate_y(x, w, M):
    y = np.zeros(len(x))
    for i in range(len(x)):
        phi = generate_phi(x[i], M)
        y[i] = np.dot(w, phi)
    return y


def plot(x, y, color, M):
    plt.plot(x, y, color=color, label='M = {}'.format(M))
    plt.xlabel('x')
    plt.ylabel('y')
    return


def generate_plot(data, x, M, color):
    Phi = generate_Phi(data, M)
    w = generate_w(Phi, data[:, 1])
    y = generate_y(x, w, M)
    plot(x, y, color, M)
    return


def generate_plot_revised(data, x, M, color, lmbda):
    Phi = generate_Phi(data, M)
    w = generate_w_revised(Phi, data[:, 1], lmbda)
    y = generate_y(x, w, M)
    plot(x, y, color, M)
    return


def cal_MSE_train(data, M, revised, lam):
    Phi = generate_Phi(data, M)
    if revised == True:
        w = generate_w_revised(Phi, data[:, 1], lam)
    else:
        w = generate_w(Phi, data[:, 1])
    y = generate_y(data[:, 0], w, M)
    e = y - data[:, 1]
    E = 0
    for i in range(len(e)):
        E += e[i] ** 2
    E /= len(data)
    return E


def cal_MSE_test(data, data_test, M, revised, lam):
    Phi = generate_Phi(data, M)
    if revised == True:
        w = generate_w_revised(Phi, data[:, 1], lam)
    else:
        w = generate_w(Phi, data[:, 1])
    y = generate_y(data_test[:, 0], w, M)
    e = y - data_test[:, 1]
    E = 0
    for i in range(len(e)):
        E += e[i] ** 2
    E /= len(data_test)
    return E


def evaluate_MSE_with_cross_validation(data, M_values, n_runs=10):
    kf = KFold(n_splits=5, shuffle=True)
    all_MSE_results = []
    for i in range(n_runs):
        MSE_results = []
        for M in M_values:
            MSE_M = 0
            for train_index, test_index in kf.split(data):
                train_data = data[train_index]
                test_data = data[test_index]
                Phi_train = generate_Phi(train_data, M)
                w = generate_w(Phi_train, train_data[:, 1])
                y_pred = generate_y(test_data[:, 0], w, M)
                MSE_fold = np.sum((y_pred - test_data[:, 1]) ** 2) / len(data)
                MSE_M += MSE_fold
            MSE_M /= 5  # average over 5 folds
            MSE_results.append(MSE_M)
        all_MSE_results.append(MSE_results)
    mean_MSE_results = np.mean(all_MSE_results, axis=0)
    return mean_MSE_results


if __name__ == '__main__':
    data = read_csv(1, 50, 'HW1.csv', delimiter=',')

    # question one
    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    x = np.linspace(0, 3, 100)
    generate_plot(data, x, 1, 'red')
    plt.title('Fitting curve with M = 1')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot(data, x, 3, 'green')
    plt.title('Fitting curve with M = 3')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot(data, x, 5, 'black')
    plt.title('Fitting curve with M = 5')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot(data, x, 10, 'purple')
    plt.title('Fitting curve with M = 10')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot(data, x, 20, 'gray')
    plt.title('Fitting curve with M = 20')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot(data, x, 30, 'gold')
    plt.title('Fitting curve with M = 30')
    plt.legend()
    plt.show()

    # question two
    x_values = range(1, 31)
    E = np.zeros(31)
    for i in range(1, 31):
        E[i] = cal_MSE_train(data, i, False, 0)

    plt.plot(x_values, E[1:], label='training data')

    data_test = read_csv(52, 20, 'HW1.csv', delimiter=',')
    Et = np.zeros(31)
    for i in range(1, 31):
        Et[i] = cal_MSE_test(data, data_test, i, False, 0)

    plt.plot(x_values, Et[1:], color='red', label='testing data')
    plt.title('Mean square error')
    plt.xlabel('M')
    plt.ylabel('Mean Square Error')
    x_ticks = range(1, 31, 5)
    x_ticks = list(x_ticks) + [30]
    plt.xticks(x_ticks)
    plt.legend()
    plt.show()
    
    # question three
    M_values = list(range(1, 31))
    MSE_results = evaluate_MSE_with_cross_validation(data[:, :2], M_values)
    x_values = range(1, 16)
    plt.plot(x_values, MSE_results[:15], label='MSE value')
    plt.title('Mean square error by different M after 5-fold cross-validation')
    plt.xlabel('M')
    plt.ylabel('Mean Square Error')
    x_ticks = range(1, 16, 5)
    x_ticks = list(x_ticks) +[15]
    plt.xticks(x_ticks)
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot(data, x, 10, 'purple')
    plt.title('Fitting curve with M = 10')
    plt.legend()
    plt.show()
    
    # question four
    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot_revised(data, x, 1, 'red', 0.1)
    plt.title('Fitting curve with M = 1 and regularization')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot_revised(data, x, 3, 'green', 0.1)
    plt.title('Fitting curve with M = 3 and regularization')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot_revised(data, x, 5, 'black', 0.1)
    plt.title('Fitting curve with M = 5 and regularization')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot_revised(data, x, 10, 'purple', 0.1)
    plt.title('Fitting curve with M = 10 and regularization')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot_revised(data, x, 20, 'gray', 0.1)
    plt.title('Fitting curve with M = 20 and regularization')
    plt.legend()
    plt.show()

    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=3)
    generate_plot_revised(data, x, 30, 'gold', 0.1)
    plt.title('Fitting curve with M = 30 and regularization')
    plt.legend()
    plt.show()

    x_values = range(1, 31)
    E4 = np.zeros(31)
    lam = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    for j in lam:
        for i in range(1, 31):
            E4[i] = cal_MSE_train(data, i, True, j)

        plt.plot(x_values, E4[1:], label='training data')

        data_test = read_csv(52, 20, 'HW1.csv', delimiter=',')
        Et4 = np.zeros(31)
        for i in range(1, 31):
            Et4[i] = cal_MSE_test(data, data_test, i, True, j)

        plt.plot(x_values, Et4[1:], color='red', label='testing data')
        plt.title('Mean square error after regularization with lambda = {}'.format(j))
        plt.xlabel('M')
        plt.ylabel('Mean Square Error')
        x_ticks = range(1, 31, 5)
        x_ticks = list(x_ticks) + [30]
        plt.xticks(x_ticks)
        plt.legend()
        plt.show()
