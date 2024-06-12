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

    image_raw = imread.imread("image.jpg")

    image_sum = image_raw.sum(axis=2)
    print(image_sum.shape)
    image_bw = image_sum / image_sum.max()
    print(image_bw.max())

    display_image(image_raw)
    display_image(image_bw)

    img_vector = image_raw.shape




if __name__ == "__main__":
    main()