import numpy
from PIL import Image, ImageOps
import cv2
import scipy.misc

def detect(img_array):
    """
    detect a number in side a black-white image
    :param img_array: a numpy array
    :return: detect_img_array
    """
    min_i, max_i, min_j, max_j = 1000, -1000, 1000, -1000
    pos, detect_img_array = [], []
    row, col = len(img_array), len(img_array[1])
    print row, col
    for i in range(row):
        for j in range(col):
            if sum(img_array[i][j]) < 255:
                pos.append((i,j))
    for i,j in pos:
        if i < min_i:
            min_i = i
        if i > max_i:
            max_i = i
        if j < min_j:
            min_j = j
        if j > max_j:
            max_j = j
    min_i -= 25
    max_i += 25
    min_j -= 25
    max_j += 25
    print min_i, max_i, min_j, max_j
    for i in range(min_i, max_i):
        detect_img_array.append(img_array[i])
    for i in range(len(detect_img_array)):
        detect_img_array[i] = detect_img_array[i][min_j:max_j]
    detect_img_array = numpy.array(detect_img_array) # convert to numpy array
    return detect_img_array
