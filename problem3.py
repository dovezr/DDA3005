#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import time
import math
import cv2 as cv


m_global, n_global = None, None  # store size of the frames


def get_video(location, interval):
    global m_global, n_global
    video_capture = cv.VideoCapture(location)
    num_of_frame = video_capture.get(cv.CAP_PROP_FRAME_COUNT)
    print('number of frames: ', num_of_frame)
    size = math.ceil(num_of_frame / interval)
    print('selected frames: ', size)
    count, num = 0, 0
    if video_capture.isOpened():
        open, frame = video_capture.read()
        m_global, n_global = frame.shape[:2]
    video_matrix = [np.zeros((size, m_global * n_global)), np.zeros((size, m_global * n_global)), np.zeros((size, m_global * n_global))]
    while open:
        ret, frame = video_capture.read()
        if frame is None:
            break
        if ret:
            if count % interval == 0:
                for i in range(3):
                    video_matrix[i][num] = frame[:, :, i].flatten('F')
                num += 1
            count += 1
    return video_matrix


def extract_by_scipy(type_of_v, type_of_iteration, matrix):
    u, s, v = svds(matrix, k=1)
    u_utilize = u[:, -1]
    s_utilize = s[-1]
    v_utilize = v[-1][0]
    background = s_utilize * v_utilize * u_utilize
    return background, 0


def background_extraction(type_of_v, type_of_iteration, matrix, threshold=1e-4):
    square = matrix.T @ matrix
    width = matrix.shape[1]
    if type_of_v == "full":
        basic = [0] * width
        v = np.full_like(basic, (1 / width) ** 0.5, dtype=np.double)
        if type_of_iteration == "normal":
            sigma_sq = v.T @ square @ v
            current_iteration = 0
            while (np.linalg.norm((square @ v - sigma_sq * v) / width)) > threshold:
                current_iteration += 1
                v_new = square @ v
                v = v_new / np.linalg.norm(v_new)
                sigma_sq = v.T @ v_new
            print('number of iterations:', current_iteration)
            background = v[0] * (matrix @ v)
            return background, current_iteration
        elif type_of_iteration == "inverse":
            sigma_square = v.T @ square @ v
            square_new = square - sigma_square * np.eye(width)
            square_new_inverse = np.linalg.inv(square_new)
            current_iteration = 0
            while (np.linalg.norm((square @ v - sigma_square * v) / width)) > threshold:
                current_iteration += 1
                v_new = square_new_inverse @ v
                v = v_new / np.linalg.norm(v_new)
                sigma_square_new, sigma_square = sigma_square, v.T @ square @ v
            print('number of iterations:', current_iteration)
            background = v[0] * (matrix @ v)
            return background, current_iteration
    elif type_of_v == "one":
        basic = [0] * width
        basic[0] = 1
        v = np.array(basic)
        if type_of_iteration == "normal":
            sigma_sq = v.T @ square @ v
            current_iteration = 0
            while (np.linalg.norm((square @ v - sigma_sq * v) / width)) > threshold:
                current_iteration += 1
                v_new = square @ v
                v = v_new / np.linalg.norm(v_new)
                sigma_sq = v.T @ v_new
            print('number of iterations:', current_iteration)
            background = v[0] * (matrix @ v)
            return background, current_iteration


# ----------This is a divider-----------------------------

def process_channel(type_of_v, type_of_iteration, channel, func, L):
    print(f'channel{channel}')
    start_time = time.time()
    m, i = func(type_of_v, type_of_iteration, L[channel].T.astype(float))
    return m, i, time.time() - start_time


def reshape_and_clip(m):
    m = np.reshape(m, (m_global, n_global), 'F')
    np.clip(m, 0, 255, out=m)
    return m


def test(path, interval):
    L = get_video(path, interval) 

    funcs = {
        'standard': extract_by_scipy,
        'extract': background_extraction,
    }
    M = [None, None, None, None]
    iteation_list = [None, None, None, None]
    time_list = [None, None, None, None]
    time_sum_list = [None, None, None, None]
    iteration_sum_list = [None, None, None, None]
    B = [None, None, None, None]

    j = 0
    for value in funcs.values():
        if value == extract_by_scipy:
            M[j] = [None, None, None]
            iteation_list[j] = [None, None, None]
            time_list[j] = [None, None, None]
            for i in range(3):
                M[j][i], iteation_list[j][i], time_list[j][i] = process_channel('scipy', 'scipy', i, value, L)
            time_sum_list[j] = time_list[j][0] + time_list[j][1] + time_list[j][2]
            M[j][0], M[j][1], M[j][2] = map(reshape_and_clip, [M[j][0], M[j][1], M[j][2]])
            B[j] = np.stack([M[j][2], M[j][1], M[j][0]], axis=-1).astype(np.uint8)
            j += 1
        else:
            check = [['full', 'normal'], ['one', 'normal'], ['full', 'inverse']]
            for i in range(3):
                current_v = check[i][0]
                current_iteration_type = check[i][1]
                M[j] = [None, None, None]
                iteation_list[j] = [None, None, None]
                time_list[j] = [None, None, None]
                for i in range(3):
                    M[j][i], iteation_list[j][i], time_list[j][i] = process_channel(current_v, current_iteration_type, i, value, L)
                time_sum_list[j] = time_list[j][0] + time_list[j][1] + time_list[j][2]
                iteration_sum_list[j] = iteation_list[j][0] + iteation_list[j][1] + iteation_list[j][2]
                M[j][0], M[j][1], M[j][2] = map(reshape_and_clip, [M[j][0], M[j][1], M[j][2]])
                B[j] = np.stack([M[j][2], M[j][1], M[j][0]], axis=-1).astype(np.uint8)
                j += 1

    return B, iteration_sum_list, time_sum_list



# In[57]:


paths = ['test_videos/640_360/sanfrancisco_01.mp4', 'test_videos/1280_720/road.mp4', 'test_videos/640_360/walking.mp4']
# we select the three path sanfrancisco_01.mp4, road.mp4, and walking.mp4
for path in paths:
    B, iteation_list, time_sum_list = test(path, 4)
    print(time_sum_list)
    plt.imshow(B[0])    # background by scipy package
    plt.show()
    plt.imshow(B[1])    # background by full-power algorithm
    plt.show()
    plt.imshow(B[2])    # background by full-inverse algorithm
    plt.show()
    plt.imshow(B[3])    # background by one-power algorithm
    plt.show()


# In[59]:


# We define a function to compare the influemce of the different 
# intervals and initial points on the performance of the algorithm

def compare_interval_and_initial(a, b, c):
    # a, b, c are start, end, and step of the interval
    B_compare = [None] * 4
    iteation_list_compare = [None] * 4
    time_sum_list_compare = [None] * 4
    for i in range(4):
        B_compare[i] = [None] * ((b - a - 1) // c + 1)
        iteation_list_compare[i] = [None] * ((b - a - 1) // c + 1)
        time_sum_list_compare[i] = [None] * ((b - a - 1) // c + 1)
    interval_list = list(range(a, b, c))
    for k in range(a, b, c):
        B, iteation_list, time_sum_list = test(path, k)
        print(time_sum_list)

        plt.imshow(B[0])
        plt.show()
        plt.imshow(B[1])
        plt.show()
        plt.imshow(B[2])
        plt.show()
        plt.imshow(B[3])
        plt.show()

        for i in range(4):
            B_compare[i][(k - a) // c] = B[i]
            iteation_list_compare[i][(k - a) // c] = iteation_list[i]
            time_sum_list_compare[i][(k - a) // c] = time_sum_list[i]
        
    plt.plot(interval_list, time_sum_list_compare[0], marker='o', markersize=4, color = 'blue', label="scipy")
    plt.plot(interval_list, time_sum_list_compare[1], marker='o', markersize=4, color = 'red', label="full-power")
    plt.plot(interval_list, time_sum_list_compare[2], marker='o', markersize=4, color = 'green', label="one-power")
    plt.plot(interval_list, time_sum_list_compare[3], marker='o', markersize=4, color = 'yellow', label="full-inverse")
    plt.legend()
    plt.title('relationship between inteval and runtime for different implementation')
    plt.xlabel('interval')
    plt.ylabel('runtime')
    plt.show()


    #plt.plot(interval_list, iteation_list_compare[0], marker='o', markersize=4, color = 'blue', label="scipy")
    plt.plot(interval_list, iteation_list_compare[1], marker='o', markersize=4, color = 'red', label="full-power")
    plt.plot(interval_list, iteation_list_compare[2], marker='o', markersize=4, color = 'green', label="one-power")
    plt.plot(interval_list, iteation_list_compare[3], marker='o', markersize=4, color = 'yellow', label="full-inverse")
    plt.legend()
    plt.title('relationship between inteval and iteration for different implementation')
    plt.xlabel('interval')
    plt.ylabel('iteration')
    plt.show()
compare_interval_and_initial(3, 16, 2)


# In[60]:


# check the path pigeons.mp4 with interval start=3, end=16, step=2
path = 'test_videos/1280_720/pigeons.mp4'
compare_interval_and_initial(3, 16, 2)


# In[61]:


# check the path pedestrians.mp4 with interval start=3, end=16, step=2
path = 'test_videos/1280_720/pedestrians.mp4'
compare_interval_and_initial(3, 16, 2)


# In[64]:


# check the path sunset.mp4 with interval start=3, end=16, step=2
path = 'test_videos/1920_1080/sunset.mp4'
compare_interval_and_initial(3, 16, 2)

