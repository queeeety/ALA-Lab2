import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imread
import os
from sklearn.preprocessing import Binarizer
from PIL import Image


def checker(eigenvalues, eigenvectors, matrix):
    for i in range(len(eigenvalues)):
        if not np.allclose(np.dot(matrix, eigenvectors[:, i]), eigenvalues[i] * eigenvectors[:, i]):
            return False
    return True


def display_image(image_array):
    plt.imshow(image_array, cmap='gray')
    plt.show()


def main():
    matrix = np.array([[7, 2, 3], [2, 5, 1], [3, 1, 11]])

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if not checker(eigenvalues, eigenvectors, matrix):
        print("Something went wrong")
        return

    # read image
    image_raw = imread.imread("image.jpg")

    # print the size of the original image and the number of color channels
    print("Original image")
    print("Image size (height, width):", image_raw.shape[:2])
    print("Number of color channels:", image_raw.shape[2], "\n")

    # convert to grayscale
    image_sum = image_raw.sum(axis=2)
    image_bw = image_sum / image_sum.max()

    # better way to convert to grayscale
    image_gray = np.dot(image_raw[...,:3], [0.2989, 0.5870, 0.1140])
    image_bw2 = image_gray / image_gray.max()

    # print the size of the bw image and the number of color channels
    print("Grayscale image")
    print("Image size (height, width):", image_bw2.shape[:2])
    print("Number of color channels: 1 (cause it is black-shaded, lol)\n")

    # display image
    display_image(image_raw)
    display_image(image_bw)
    display_image(image_bw2)

    img_vector = image_raw.shape


if __name__ == "__main__":
    main()
