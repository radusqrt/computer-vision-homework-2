import numpy as np
import cv2 as cv
import sys
import math

from util import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.setrecursionlimit(100000)


def rgb_to_hsv_skin_filter(img):
    # Convert BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([20, 180, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_skin, upper_skin)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(img, img, mask=mask)
    return res


def remove_noisy_components(img):
    visited = {}
    height, width, _ = img.shape

    for i in range(height):
        for j in range(width):
            if visited.get((i, j), False) is False and not is_black_pixel(img[i, j]):
                pixels = fill([(i, j)], img, visited)
                if len(pixels) < 1000:
                    fill([(i, j)], img, visited, True)


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def find_pca(img):
    all_vectors = []
    visited = {}
    height, width, _ = img.shape

    for i in range(erosion.shape[0]):
        for j in range(erosion.shape[1]):
            if visited.get((i, j), False) is False and not is_black_pixel(img[i, j]):

                X = np.array(list(fill([(i, j)], img, visited)))
                pca = PCA(n_components=2)
                pca.fit(X)
                # print(pca.components_)
                # print(pca.explained_variance_)
                # print(pca.mean_)

                # plot data
                plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
                curr_vectors = []
                for length, vector in zip(pca.explained_variance_, pca.components_):
                    v = vector * 3 * np.sqrt(length)
                    draw_vector(pca.mean_, pca.mean_ - v)
                    print('draw', pca.mean_, pca.mean_ - v)
                    curr_vectors.append((pca.mean_, pca.mean_ - v))
                all_vectors.append(curr_vectors)
                plt.axis('equal')

    plt.show()
    return all_vectors


img = cv.imread('imgs/1.jpg')
skin_hsv_filtered = rgb_to_hsv_skin_filter(img)
remove_noisy_components(skin_hsv_filtered)
dilation = cv.dilate(skin_hsv_filtered, np.ones((5, 5)), iterations = 10)
erosion = cv.erode(dilation, np.ones((5, 5)), iterations = 10)
cv.imshow('erosion', erosion)
cv.waitKey(0)
pca_vectors = find_pca(erosion)

for [v0, v1] in pca_vectors:
    center = tuple([int(x) for x in v0[0]])
    center = (center[1], center[0])
    y_length = int(np.linalg.norm(v0[0] - v0[1]))
    x_length = int(np.linalg.norm(v1[0] - v1[1]))
    angle = - math.atan2(v0[0][1] - v0[1][1], v0[0][0] - v0[1][0]) * 180 / math.pi
    print(angle)
    cv.ellipse(img, center, (x_length, y_length), angle, 0, 360, 255, 5)

cv.imshow('final', img)
cv.waitKey(0)