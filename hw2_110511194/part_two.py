import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from part_one import calculate_mean_vector
from part_one import generate_matrix
from part_one import generate_w
from part_one import find_pob
from part_one import generate_wk0
from part_one import generate_pt
from part_one import compute_a_k
from part_one import generate_plot
from part_one import generate_sig
from part_one import svd_inverse


def import_and_classify_data(file_path):
    df = pd.read_excel(file_path, header=0, usecols=[1, 2, 3])

    np_array_1 = df[df.iloc[:, 2] == 1].iloc[:, :2].to_numpy()
    np_array_2 = df[df.iloc[:, 2] == 2].iloc[:, :2].to_numpy()
    np_array_3 = df[df.iloc[:, 2] == 3].iloc[:, :2].to_numpy()
    np_array_4 = df[df.iloc[:, 2] == 4].iloc[:, :2].to_numpy()

    plt.plot(np_array_1[:, 0], np_array_1[:, 1], 'o', markersize=3, color='blue')
    plt.plot(np_array_2[:, 0], np_array_2[:, 1], 'o', markersize=3, color='green')
    plt.plot(np_array_3[:, 0], np_array_3[:, 1], 'o', markersize=3, color='red')
    plt.plot(np_array_4[:, 0], np_array_4[:, 1], 'o', markersize=3, color='blue')
    plt.show()

    return np_array_1, np_array_2, np_array_3, np_array_4


def generate_ak(points, w1, w2, w3, w4, w10, w20, w30, w40):
    a1 = (np.apply_along_axis(compute_a_k, axis=1, arr=points, w=w1, t=w10)).flatten()
    a2 = (np.apply_along_axis(compute_a_k, axis=1, arr=points, w=w2, t=w20)).flatten()
    a3 = (np.apply_along_axis(compute_a_k, axis=1, arr=points, w=w3, t=w30)).flatten()
    a4 = (np.apply_along_axis(compute_a_k, axis=1, arr=points, w=w4, t=w40)).flatten()

    exp_a1 = (0.4 / 0.55) * np.exp(a1) + (0.15 / 0.55) * np.exp(a4)
    exp_a2 = np.exp(a2)
    exp_a3 = np.exp(a3)
    exp_sum = exp_a1 + exp_a2 + exp_a3

    pc1_x = exp_a1 / exp_sum
    pc2_x = exp_a2 / exp_sum
    pc3_x = exp_a3 / exp_sum

    return pc1_x, pc2_x, pc3_x


def compare_sets(A, B, C):
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

        # Set the index of the maximum value to 1 in the result array
        result[i] = max_index

    return result


def generate_t():
    mat = np.zeros((1000, 4))
    mat[:550, 0] = 1
    mat[550:800, 1] = 1
    mat[800:, 2] = 1
    return mat


def generate_w_dis():
    w1_x1 = np.ones(shape=(M, 1))
    w2_x1 = np.ones(shape=(M, 1))
    w3_x1 = np.ones(shape=(M, 1))
    return w1_x1, w2_x1, w3_x1


def generate_y_dis(Phi_x1, w1_x1, w2_x1, w3_x1):
    a1 = Phi_x1 @ (w1_x1.reshape(3, 1))
    a2 = Phi_x1 @ (w2_x1.reshape(3, 1))
    a3 = Phi_x1 @ (w3_x1.reshape(3, 1))

    e_a1 = np.exp(a1)
    e_a2 = np.exp(a2)
    e_a3 = np.exp(a3)
    e_a_sum = e_a1 + e_a2 + e_a3

    y1 = e_a1 / e_a_sum
    y2 = e_a2 / e_a_sum
    y3 = e_a3 / e_a_sum

    return y1, y2, y3


def generate_grad(Phi_x1, y1, y2, y3, t):
    grad_w1 = Phi_x1.T @ (y1 - (t[:, 0]).reshape(1000, 1))
    grad_w2 = Phi_x1.T @ (y2 - (t[:, 1]).reshape(1000, 1))
    grad_w3 = Phi_x1.T @ (y3 - (t[:, 2]).reshape(1000, 1))
    return grad_w1, grad_w2, grad_w3


def generate_H(Phi_x1, y1, y2, y3):
    one = np.ones(1000).reshape((1000, 1))
    y = np.hstack((y1, y2, y3))

    for i in range(3):
        for j in range(3):
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


def generate_new_w(w1_x1, w2_x1, w3_x1, H_inv, grad_w1, grad_w2, grad_w3):
    w_old = np.vstack((w1_x1, w2_x1, w3_x1))
    grad = np.vstack((grad_w1, grad_w2, grad_w3))
    w_new = w_old - H_inv @ grad
    # print(np.mean(w_new))
    return w_new[0:3, 0].reshape(3, 1), w_new[3:6, 0].reshape(3, 1), w_new[6:9, 0].reshape(3, 1)


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

    pc1_x, pc2_x, pc3_x = generate_ak(points, w1, w2, w3, w4, w10, w20, w30, w40)

    result = compare_sets(pc1_x, pc2_x, pc3_x)

    generate_plot(points, result)

    # discriminative model
    M = 3

    C = np.concatenate((C1, C4, C2, C3), axis=0)
    Phi_x1 = generate_sig(C, 0, M)
    w1_x1, w2_x1, w3_x1 = generate_w_dis()

    Phi_x2 = generate_sig(C, 1, M)
    w1_x2, w2_x2, w3_x2 = generate_w_dis()

    t = generate_t()

    b1 = b2 = False
    while True:
        for i in range(5):
            y1, y2, y3 = generate_y_dis(Phi_x1, w1_x1, w2_x1, w3_x1)

            grad_w1, grad_w2, grad_w3 = generate_grad(Phi_x1, y1, y2, y3, t)
            grad_mean = np.sum(np.abs(grad_w1) + np.abs(grad_w2) + np.abs(grad_w3))
            # print("x1", grad_mean)
            if grad_mean < 1:
                b1 = True
                break

            H = generate_H(Phi_x1, y1, y2, y3)
            H_inv = svd_inverse(H)

            w1_x1, w2_x1, w3_x1 = generate_new_w(w1_x1, w2_x1, w3_x1, H_inv, grad_w1, grad_w2, grad_w3)

        for i in range(5):
            y1_2, y2_2, y3_2 = generate_y_dis(Phi_x2, w1_x2, w2_x2, w3_x2)
            #print(np.mean(y1_2))
            grad_w1_2, grad_w2_2, grad_w3_2 = generate_grad(Phi_x2, y1_2, y2_2, y3_2, t)
            grad_mean_2 = np.sum(np.abs(grad_w1_2) + np.abs(grad_w2_2) + np.abs(grad_w3_2))
            #print("x2", grad_mean_2)
            if grad_mean_2 < 10:
                b2 = True
                break

            H_2 = generate_H(Phi_x2, y1_2, y2_2, y3_2)
            H_inv_2 = svd_inverse(H_2)

            w1_x2, w2_x2, w3_x2 = generate_new_w(w1_x2, w2_x2, w3_x2, H_inv_2, grad_w1_2, grad_w2_2, grad_w3_2)
        # print(w1_x2, w2_x2, w3_x2)

        Phi_data1 = generate_sig(points, 0, M)
        Phi_data2 = generate_sig(points, 1, M)

        y1_d1, y2_d1, y3_d1 = generate_y_dis(Phi_data1, w1_x1, w2_x1, w3_x1)
        y1_d2, y2_d2, y3_d2 = generate_y_dis(Phi_data2, w1_x2, w2_x2, w3_x2)

        result_dis = compare_sets(y1_d1 + y1_d2, y2_d1 + y2_d2, y3_d1 + y3_d2)

        generate_plot(points, result_dis)
        if b1 == b2 == True:
            break
        else:
            b1 = b2 = False
