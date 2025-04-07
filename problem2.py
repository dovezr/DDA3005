import numpy as np
from PIL import Image
from numpy import asarray
from numpy import linalg
from matplotlib import image
from matplotlib import pyplot as plt
import time
import scipy
#import cv2 as cv
from scipy.linalg import lu, solve, norm
import math
import problem1 as SVD
import copy
import math


def creat_A(j, k, n):
    A = np.zeros((n, n))
    for t in range(0, n):
        for r in range(0, n):
            i = t - r + n
            if (i >= n + j - k + 1) and (i <= n + j):
                A[t, r] = (2 / (k * (k + 1))) * (k + i - n - j)
    return A


def inverse_matrix_using_lu(A):
    P, L, U = scipy.linalg.lu(A)
    identity = np.eye(len(A))
    _, _, I_U = scipy.linalg.lu(identity)
    A_inverse = np.linalg.solve(U, np.linalg.solve(I_U, np.linalg.solve(L, P.T)))

    return A_inverse


def singular_drawing(X, type):
    if type == "A":
        U, sigma, VT = SVD.phase2A(X)
        # print(sigma)
    else:
        U, sigma, VT = SVD.phase2B(X)

    # U, sigma, VT = np.linalg.svd(X)
    singular = []
    i_list = []

    for i in range(sigma.shape[0]):
        singular.append(sigma[i, i] ** (1 / 2))
        i_list.append(i)
    plt.plot(i_list, singular)
    plt.title("Singular values")
    # plt.show()


def qr_inverse(A):
    Q, R = np.linalg.qr(A)
    if not np.all(np.isfinite(R.diagonal())):
        raise ValueError("Matrix is not invertible.")
    R_inv = np.linalg.inv(R)
    Q_inv = Q.T

    A_inv = np.dot(R_inv, Q_inv)

    return A_inv


def cal_trunc(A, trunc=100000000, pictype="A"):
    if pictype == "A":
        U, sigma, VT = SVD.phase2A(A)
    else:
        U, sigma, VT = SVD.phase2B(A)

    rows, columns = A.shape
    A_pseudoinverse = np.zeros((rows, columns))
    for i in range(rows):
        uiT = np.transpose(U[:, i])
        vi = np.transpose(VT[i, :])
        if sigma[i, i] != 0 and i < trunc:
            A_pseudoinverse += np.outer(vi, uiT) / sigma[i, i]
    return A_pseudoinverse


def slice_and_combine(img, m, n, t, trunc=100000000, pictype="A"):
    X_final_shape = (m, n, t)
    X_final = np.zeros(X_final_shape)
    B_final = np.zeros(X_final_shape)
    for i in range(t):
        X_p = np.squeeze(img[:, :, i : i + 1])
        Al = creat_A(0, 12, n)
        Ar = creat_A(1, 30, n)
        B = np.dot(np.dot(Al, X_p), Ar)
        B_final[:, :, i : i + 1] = B[:, :, np.newaxis]
        Al_trunc = cal_trunc(Al, trunc, pictype)
        Ar_trunc = cal_trunc(Ar, trunc, pictype)
        X = np.dot(np.dot(Al_trunc, B), Ar_trunc)

        if i == 0:
            if pictype == "A":
                plt.figure(1)
                singular_drawing(Al, "A")
                plt.figure(2)
                singular_drawing(Ar, "A")
            else:
                plt.figure(1)
                singular_drawing(Al, "B")
                plt.figure(2)
                singular_drawing(Ar, "B")
        X_final[:, :, i : i + 1] = X[:, :, np.newaxis]
    return X_final, B_final


def cal_PSNR(X, X_trunc):
    m, n, t = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    X_trunc_flat = X_trunc.reshape(-1, X_trunc.shape[-1])

    i = np.linalg.norm(X_flat - X_trunc_flat, "fro") ** 2
    # i = np.linalg.norm(X - X_trunc, "fro") ** 2
    # np.sum(np.square(X - X_trunc))
    return 10 * (math.log10((n**2) / i))


"""def open_pic(name, pictype):
    im = Image.open("C:/Users/zld/Desktop/大三上/DDA 3005/Project/test_images-3/" + name + ".png")
    img = np.array(im)
    img = img.astype(np.float64) / 255
    m, n, t = img.shape
    img[42, 0 : n - 1] = 0
    X, B = slice_and_combine(img, m, n, t, 3000, pictype)
    X = (X * 255).astype(np.uint8)
    B = (B * 255).astype(np.uint8)
    image1 = Image.fromarray(X)
    if image1.mode == "RGBA":
        image1 = image1.convert("RGB")
    image1.save(name + "1.jpg")
    image2 = Image.fromarray(B)
    if image2.mode == "RGBA":
        image2 = image2.convert("RGB")
    image2.save(name + "2.jpg")
    PSNR = cal_PSNR(img, X)
    return PSNR"""


def open_pic(name, pictype):
    im = Image.open("/Users/ziyiou/study database/大三上/DDA3005/group project/test_images-3/" + name + ".png")
    img = np.array(im)
    img = img.astype(np.float64) / 255
    m, n, t = img.shape
    img[42, 0 : n - 1] = 0
    X, B = slice_and_combine(img, m, n, t, 300, pictype)
    X = (X * 255).astype(np.uint8)
    B = (B * 255).astype(np.uint8)
    image1 = Image.fromarray(X)
    if image1.mode == "RGBA":
        image1 = image1.convert("RGB")
    image2 = Image.fromarray(B)
    if image2.mode == "RGBA":
        image2 = image2.convert("RGB")
    if pictype == "A":
        image1.save(name + "1A.jpg")
        image2.save(name + "2A.jpg")
    else:
        image1.save(name + "1B.jpg")
        image2.save(name + "2B.jpg")
    PSNR = cal_PSNR(img, X)
    return PSNR


# , "256_256_casino", "256_256_hand"
name_list = ["256_256_buildings", "256_256_casino", "256_256_hand"]
typeset = ["A", "B"]
for pictype in typeset:
    for name in name_list:
        start_time = time.time()
        PSNR = open_pic(name, pictype)
        end_time = time.time()
        execution_time = end_time - start_time
        print("use phase Ⅱ-" + pictype + ", running time is", execution_time)
        print(PSNR)
