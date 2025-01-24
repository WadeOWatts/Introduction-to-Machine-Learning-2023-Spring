import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def import_and_classify_data(file_path):
    df = pd.read_excel(file_path, header=0, usecols=[1, 2, 3])

    np_array_1 = df[df.iloc[:, 2] == 1].iloc[:, :2].to_numpy()
    np_array_2 = df[df.iloc[:, 2] == 2].iloc[:, :2].to_numpy()
    np_array_3 = df[df.iloc[:, 2] == 3].iloc[:, :2].to_numpy()
    np_array_4 = df[df.iloc[:, 2] == 4].iloc[:, :2].to_numpy()

    plt.plot(np_array_1[:, 0], np_array_1[:, 1], 'o', markersize=3, color='blue')
    plt.plot(np_array_2[:, 0], np_array_2[:, 1], 'o', markersize=3, color='green')
    plt.plot(np_array_3[:, 0], np_array_3[:, 1], 'o', markersize=3, color='red')
    plt.plot(np_array_4[:, 0], np_array_4[:, 1], 'o', markersize=3, color='black')

    plt.show()
    return np_array_1, np_array_2, np_array_3, np_array_4


def calculate_mean_vector(C1, C2, C3, C4):
    u1 = np.mean(C1, axis=0)
    u2 = np.mean(C2, axis=0)
    u3 = np.mean(C3, axis=0)
    u4 = np.mean(C4, axis=0)
    return u1, u2, u3, u4


def generate_matrix(C1, C2, C3, C4, u1, u2, u3, u4):
    # Compute the number of rows in each C matrix
    n1 = C1.shape[0]
    n2 = C2.shape[0]
    n3 = C3.shape[0]
    n4 = C4.shape[0]

    # Compute the sum of the outer products of each row in each C matrix minus its corresponding u vector
    sum_matrix = np.zeros((2, 2))
    for i in range(n1):
        sum_matrix += np.outer(C1[i] - u1, C1[i] - u1)
    for i in range(n2):
        sum_matrix += np.outer(C2[i] - u2, C2[i] - u2)
    for i in range(n3):
        sum_matrix += np.outer(C3[i] - u3, C3[i] - u3)
    for i in range(n4):
        sum_matrix += np.outer(C4[i] - u4, C4[i] - u4)

    # Divide by the total number of rows in all C matrices
    total_rows = n1 + n2 + n3 + n4
    matrix = sum_matrix / total_rows

    return matrix


def generate_w(matrix, u1, u2, u3, u4):
    w1 = np.linalg.inv(matrix) @ u1
    w2 = np.linalg.inv(matrix) @ u2
    w3 = np.linalg.inv(matrix) @ u3
    w4 = np.linalg.inv(matrix) @ u4
    return w1, w2, w3, w4


def find_pob(C1, C2, C3, C4):
    n1 = C1.shape[0]
    n2 = C2.shape[0]
    n3 = C3.shape[0]
    n4 = C4.shape[0]
    total_rows = n1 + n2 + n3 + n4
    pc1 = n1 / total_rows
    pc2 = n2 / total_rows
    pc3 = n3 / total_rows
    pc4 = n4 / total_rows
    return pc1, pc2, pc3, pc4


def generate_wk0(matrix, u1, u2, u3, u4, pc1, pc2, pc3, pc4):
    w10 = (-0.5 * (u1.reshape(2, 1).T @ np.linalg.inv(matrix) @ u1.reshape(2, 1)) + np.log(pc1)).reshape(1, )
    w20 = (-0.5 * (u2.reshape(2, 1).T @ np.linalg.inv(matrix) @ u2.reshape(2, 1)) + np.log(pc2)).reshape(1, )
    w30 = (-0.5 * (u3.reshape(2, 1).T @ np.linalg.inv(matrix) @ u3.reshape(2, 1)) + np.log(pc3)).reshape(1, )
    w40 = (-0.5 * (u4.reshape(2, 1).T @ np.linalg.inv(matrix) @ u4.reshape(2, 1)) + np.log(pc4)).reshape(1, )
    return w10, w20, w30, w40


def generate_pt():
    x = np.arange(0, 101)
    y = np.arange(0, 101)
    X, Y = np.meshgrid(x, y)

    # Flatten X and Y into 1D arrays
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # Combine x_flat and y_flat into a single 2D array
    points = np.stack((x_flat, y_flat), axis=1)
    return points


def compute_a_k(x, w, t):
    return np.dot(w.T, x) + t


def generate_ak(points, w1, w2, w3, w4, w10, w20, w30, w40):
    a1 = (np.apply_along_axis(compute_a_k, axis=1, arr=points, w=w1, t=w10)).flatten()
    a2 = (np.apply_along_axis(compute_a_k, axis=1, arr=points, w=w2, t=w20)).flatten()
    a3 = (np.apply_along_axis(compute_a_k, axis=1, arr=points, w=w3, t=w30)).flatten()
    a4 = (np.apply_along_axis(compute_a_k, axis=1, arr=points, w=w4, t=w40)).flatten()

    exp_a1 = np.exp(a1)
    exp_a2 = np.exp(a2)
    exp_a3 = np.exp(a3)
    exp_a4 = np.exp(a4)
    exp_sum = exp_a1 + exp_a2 + exp_a3 + exp_a4

    pc1_x = exp_a1 / exp_sum
    pc2_x = exp_a2 / exp_sum
    pc3_x = exp_a3 / exp_sum
    pc4_x = exp_a4 / exp_sum

    return pc1_x, pc2_x, pc3_x, pc4_x


def compare_sets(A, B, C, D):
    # Create a numpy array of zeros with the same length as the input sets
    result = np.zeros_like(A)

    # Loop through each element in the sets and compare them
    for i in range(len(A)):
        max_value = A[i]
        max_index = 0

        # Find the maximum value among the four sets
        if B[i] > max_value:
            max_value = B[i]
            max_index = 1
        if C[i] > max_value:
            max_value = C[i]
            max_index = 2
        if D[i] > max_value:
            max_value = D[i]
            max_index = 3

        # Set the index of the maximum value to 1 in the result array
        result[i] = max_index

    return result


def generate_plot(points, result):
    colors = ['blue', 'green', 'red', 'black']
    fig, ax = plt.subplots()

    for i in range(len(colors)):
        idx = np.where(result == i)
        ax.scatter(points[idx, 0], points[idx, 1], c=colors[i], label=f'{i + 1}')

    ax.set_xlabel('x1: athleticism')
    ax.set_ylabel('x2: skill')
    ax.legend()
    plt.show()

    return


def generate_sig(data, c, M):
    a = data[:, c]
    phi = np.zeros((len(a), M))
    for i in range(0, len(a)):
        phi[i, 0] = 1
        for j in range(1, M):
            mu = 100 * j / M
            phi[:, j] = 1 / (1 + np.exp(-(a - mu) / 5))
    return phi


def generate_t():
    mat = np.zeros((1000, 4))
    mat[:400, 0] = 1
    mat[400:650, 1] = 1
    mat[650:850, 2] = 1
    mat[850:, 3] = 1
    return mat


def generate_w_dis():
    w1_x1 = np.ones(shape=(M, 1))
    w2_x1 = np.ones(shape=(M, 1))
    w3_x1 = np.ones(shape=(M, 1))
    w4_x1 = np.ones(shape=(M, 1))
    return w1_x1, w2_x1, w3_x1, w4_x1


def generate_y_dis(Phi_x1, w1_x1, w2_x1, w3_x1, w4_x1):
    a1 = Phi_x1 @ (w1_x1.reshape(5, 1))
    a2 = Phi_x1 @ (w2_x1.reshape(5, 1))
    a3 = Phi_x1 @ (w3_x1.reshape(5, 1))
    a4 = Phi_x1 @ (w4_x1.reshape(5, 1))

    e_a1 = np.exp(a1)
    e_a2 = np.exp(a2)
    e_a3 = np.exp(a3)
    e_a4 = np.exp(a4)

    e_a_sum = e_a1 + e_a2 + e_a3 + e_a4

    y1 = e_a1 / e_a_sum
    y2 = e_a2 / e_a_sum
    y3 = e_a3 / e_a_sum
    y4 = e_a4 / e_a_sum

    return y1, y2, y3, y4


def generate_grad(Phi_x1, y1, y2, y3, y4, t):
    grad_w1 = Phi_x1.T @ (y1 - (t[:, 0]).reshape(1000, 1))
    grad_w2 = Phi_x1.T @ (y2 - (t[:, 1]).reshape(1000, 1))
    grad_w3 = Phi_x1.T @ (y3 - (t[:, 2]).reshape(1000, 1))
    grad_w4 = Phi_x1.T @ (y4 - (t[:, 3]).reshape(1000, 1))
    return grad_w1, grad_w2, grad_w3, grad_w4


def generate_H(Phi_x1, y1, y2, y3, y4):
    one = np.ones(1000).reshape((1000, 1))
    y = np.hstack((y1, y2, y3, y4))

    for i in range(4):
        for j in range(4):
            if i == j:
                R = np.diag(np.diag(y[:, i] * (one - y[:, j])))
            else:
                R = np.diag((y[:, i] * (-y[:, j])).flatten())
            H_sub = Phi_x1.T @ R @ Phi_x1
            if j == 0:
                H_h = H_sub
            else:
                H_h = np.hstack((H_h, H_sub))
        if i == 0:
            H = H_h
        else:
            H = np.vstack((H, H_h))

    return H


def svd_inverse(A):
    U, s, Vt = np.linalg.svd(A)
    s_inv = np.zeros_like(s)
    mask = s > 1e-15  # Set a tolerance threshold for small singular values
    s_inv[mask] = 1 / s[mask]
    A_inv = Vt.T @ np.diag(s_inv) @ U.T

    return A_inv


def generate_new_w(w1_x1, w2_x1, w3_x1, w4_x1, H_inv, grad_w1, grad_w2, grad_w3, grad_w4):
    w_old = np.vstack((w1_x1, w2_x1, w3_x1, w4_x1))
    grad = np.vstack((grad_w1, grad_w2, grad_w3, grad_w4))
    w_new = w_old - H_inv @ grad
    return w_new[0:5, 0].reshape(5, 1), w_new[5:10, 0].reshape(5, 1), w_new[10:15, 0].reshape(5, 1), w_new[15:20,
                                                                                                     0].reshape(5, 1)


if __name__ == "__main__":

    # generative model
    imported_data = import_and_classify_data('HW2.xlsx')

    C1, C2, C3, C4 = imported_data

    u1, u2, u3, u4 = calculate_mean_vector(C1, C2, C3, C4)

    matrix = generate_matrix(C1, C2, C3, C4, u1, u2, u3, u4)

    w1, w2, w3, w4 = generate_w(matrix, u1, u2, u3, u4)

    pc1, pc2, pc3, pc4 = find_pob(C1, C2, C3, C4)

    w10, w20, w30, w40 = generate_wk0(matrix, u1, u2, u3, u4, pc1, pc2, pc3, pc4)

    points = generate_pt()

    pc1_x, pc2_x, pc3_x, pc4_x = generate_ak(points, w1, w2, w3, w4, w10, w20, w30, w40)

    result = compare_sets(pc1_x, pc2_x, pc3_x, pc4_x)

    generate_plot(points, result)

    # discriminitive model
    M = 5

    C = np.concatenate((C1, C2, C3, C4), axis=0)
    Phi_x1 = generate_sig(C, 0, M)
    w1_x1, w2_x1, w3_x1, w4_x1 = generate_w_dis()

    Phi_x2 = generate_sig(C, 1, M)
    w1_x2, w2_x2, w3_x2, w4_x2 = generate_w_dis()

    t = generate_t()

    result_dis_old = np.zeros((10201, 1))

    while True:
        for i in range(5):
            y1, y2, y3, y4 = generate_y_dis(Phi_x1, w1_x1, w2_x1, w3_x1, w4_x1)

            grad_w1, grad_w2, grad_w3, grad_w4 = generate_grad(Phi_x1, y1, y2, y3, y4, t)
            grad_mean = np.sum(np.abs(grad_w1) + np.abs(grad_w2) + np.abs(grad_w3) + np.abs(grad_w4))
            if grad_mean < 1:
                break

            H = generate_H(Phi_x1, y1, y2, y3, y4)
            H_inv = svd_inverse(H)

            w1_x1, w2_x1, w3_x1, w4_x1 = generate_new_w(w1_x1, w2_x1, w3_x1, w4_x1, H_inv, grad_w1, grad_w2, grad_w3,
                                                        grad_w4)

        for i in range(5):
            y1_2, y2_2, y3_2, y4_2 = generate_y_dis(Phi_x2, w1_x2, w2_x2, w3_x2, w4_x2)

            grad_w1_2, grad_w2_2, grad_w3_2, grad_w4_2 = generate_grad(Phi_x2, y1_2, y2_2, y3_2, y4_2, t)
            grad_mean_2 = np.sum(np.abs(grad_w1_2) + np.abs(grad_w2_2) + np.abs(grad_w3_2) + np.abs(grad_w4_2))
            if grad_mean_2 < 1:
                break

            H_2 = generate_H(Phi_x2, y1_2, y2_2, y3_2, y4_2)
            H_inv_2 = svd_inverse(H_2)

            w1_x2, w2_x2, w3_x2, w4_x2 = generate_new_w(w1_x2, w2_x2, w3_x2, w4_x2, H_inv_2, grad_w1_2, grad_w2_2,
                                                        grad_w3_2, grad_w4_2)

        Phi_data1 = generate_sig(points, 0, M)
        Phi_data2 = generate_sig(points, 1, M)

        y1_d1, y2_d1, y3_d1, y4_d1 = generate_y_dis(Phi_data1, w1_x1, w2_x1, w3_x1, w4_x1)
        y1_d2, y2_d2, y3_d2, y4_d2 = generate_y_dis(Phi_data2, w1_x2, w2_x2, w3_x2, w4_x2)

        result_dis = compare_sets(y1_d1 + y1_d2, y2_d1 + y2_d2, y3_d1 + y3_d2, y4_d1 + y4_d2)


        result_dis_old = result_dis
        generate_plot(points, result_dis)
        if np.array_equal(result_dis, result_dis_old):
            break
