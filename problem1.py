import numpy as np
import time
import copy
import math


def set_lowVal_zero(X):
    low_values_indices = abs(X) < 1e-12  # where values are low
    X[low_values_indices] = 0  # all low values set to 0
    return X


def householder(v, i):
    alpha = np.linalg.norm(v)
    v[i] = v[i] + np.sign(v[i]) * alpha
    v = v / np.linalg.norm(v)

    return np.eye(len(v)) - 2.0 * np.outer(v, v.T)


def Golub_Kahan(X):
    col = X.shape[1]
    row = X.shape[0]

    B = X.copy()
    P1 = np.eye(row)
    Q1 = np.eye(col)
    for i in range(min(col, row) - 1):
        # column
        h = np.zeros(len(B[:, i]))
        h[i:] = B[i:, i]
        P = householder(h, i)
        B = set_lowVal_zero(P @ B)
        P1 = P @ P1
        #     print(i, B)
        #     # row
        # for i in range(row - 1):
        h = np.zeros(len(B[i, :]))
        h[i + 1 :] = B[i, i + 1 :]
        Q = householder(h, i + 1)
        B = set_lowVal_zero(B @ Q)
        Q1 = Q1 @ Q

    for i in range(min(col, row) - 1, max(col, row) - 1):
        if col > row:
            h = np.zeros(len(B[i, :]))
            h[i + 1 :] = B[i, i + 1 :]

            Q = householder(h, i + 1)
            B = set_lowVal_zero(B @ Q)
            Q1 = Q1 @ Q

        elif col < row:
            h = np.zeros(len(B[:, i]))
            h[i:] = B[i:, i]
            P = householder(h, i)
            B = set_lowVal_zero(P @ B)
            P1 = P @ P1

    if row < col:
        B = B[0:row, 0:row]
    elif row > col:
        B = B[0:col, 0:col]

    return B, P1, Q1


def phase_2a(A, max_iterations=10000, tol=1e-12):
    n = A.shape[0]
    eigenvectors = np.eye(A.shape[0])
    # eigenvalues = np.eye(A.shape[0])
    eigenvalues_list = []
    Q_list = []
    X = A
    for i in range(max_iterations):
        # QR factorization
        bottom_right = X[-2:, -2:]
        eigenvalues1, eigenvectors1 = np.linalg.eig(bottom_right)
        bottom_right_element = X[-1, -1]
        if abs(bottom_right_element - eigenvalues1[0]) > abs(
            bottom_right_element - eigenvalues1[1]
        ):
            sigma = eigenvalues1[1]
        else:
            sigma = eigenvalues1[0]
        Q, R = np.linalg.qr(X - sigma * np.eye(n))

        U = np.eye(A.shape[0])
        U[0 : Q.shape[0], 0 : Q.shape[0]] = Q

        eigenvectors = eigenvectors @ U

        X = np.dot(R, Q) + sigma * np.eye(n)

        if np.linalg.norm(X[:-1, -1]) <= tol:
            eigenvalues_list.append(X[-1, -1])
            Q_list.append(Q)
            X = X[:-1, :-1]

            n = n - 1
        if X.shape == (1, 1):
            eigenvalues_list.append(X[-1, -1])
            Q_list.append(Q)
            break

    eigenvalues_list.reverse()

    return eigenvalues_list, eigenvectors


def phase2A(A):
    start_time = time.time()
    B, P, Q = Golub_Kahan(A)

    eigenvalues, eigenvectors = phase_2a(B @ B.T)
    sigma = np.zeros((A.shape[0], A.shape[1]))
    for i in range(min(A.shape[0], A.shape[1])):
        sigma[i, i] = eigenvalues[i]**(1/2)

    U = np.linalg.inv(P) @ eigenvectors
    Vt = np.linalg.inv(U @ sigma) @ A
    end_time = time.time()
    execution_time = end_time - start_time
    print("use phase Ⅱ-A, running time should be", execution_time)
    return U, sigma, Vt


def phase_2b(A, max_iterations=10000, tol=1e-12):
    t = A.shape[0]
    eigenvectors = np.eye(A.shape[0])
    eigenvalues_list = []
    for j in range(A.shape[0] - 1):
        n = A.shape[0]
        for i in range(max_iterations):
            Q, R = np.linalg.qr(A.T)
            U = np.eye(t)
            U[0 : Q.shape[0], 0 : Q.shape[0]] = Q

            eigenvectors = eigenvectors @ U
            L = np.linalg.cholesky(R @ R.T)
            A = L.T
            if A[0, 1] < tol or A[n - 2, n - 1] < tol:
                eigenvalues_list.append(A[-1, -1])
                A = A[:-1, :-1]
                break
    eigenvalues_list.append(A[-1, -1])
    eigenvalues_list.reverse()
    return eigenvalues_list, eigenvectors


def phase2B(A):
    start_time = time.time()
    B, P, Q = Golub_Kahan(A)

    eigenvalues, eigenvectors = phase_2b(B @ B.T)
    sigma = np.zeros((A.shape[0], A.shape[1]))
    for i in range(min(A.shape[0], A.shape[1])):
        sigma[i, i] = eigenvalues[i]**(1/2)

    U = np.linalg.inv(P) @ eigenvectors
    Vt = np.linalg.inv(U @ sigma) @ A
    end_time = time.time()
    execution_time = end_time - start_time
    print("use phase Ⅱ-B, running time should be", execution_time)
    return U, sigma, Vt


A = np.array([[4, 3, 0, 4], [2, 1, 2, 10], [4, 7, 0, 3], [5, 6, 1, 3]])
# B, P, Q = Golub_Kahan(A)
# # eigenvalues_list, eigenvectors = phase_2b(B.T @ B)
# # print(eigenvalues_list, eigenvectors)
# eigenvalues, eigenvectors = np.linalg.eig(B.T @ B)
# print(eigenvalues, np.linalg.inv(P) @ eigenvectors)
# U, sigma, Vt = phase2A(A)
# print(U @ sigma @ Vt)
# print(U)

# # print(np.linalg.inv(P) @ B @ np.linalg.inv(Q))


# # U, sigma, Vt = phase2A(A)
# # print(22222222, U @ sigma @ Vt)
# # U, sigma, Vt = phase2B(A)
# # U, S, V = phase2_A1(B)
# # # print(U, sigma, Vt)

# # print(22222222, U @ np.diag(S) @ V)
# U, sigma, VT = np.linalg.svd(A)
# print(U)
# # E = np.eye(len(sigma))
# for i in range(len(sigma)):
#     E[i, i] = sigma[i]
# print(1111111, U @ E @ VT, 111111)


# print(B)
# print(set_lowVal_zero(P @ A @ Q))
# eigenvalues, eigenvectors = phase_2a(np.dot(np.transpose(B), B))
# print(eigenvalues, eigenvectors)
# eigenvalues = phase_2b(np.dot(np.transpose(B), B))
# print(eigenvalues)

# eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.transpose(B), B))
# print(eigenvalues, eigenvectors)
