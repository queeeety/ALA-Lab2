import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imread
from sklearn.decomposition import PCA, IncrementalPCA
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
    image_gray = np.dot(image_raw[...,:3], [0.2989, 0.5870, 0.1140])
    image_bw2 = image_gray / image_gray.max()

    # print the size of the bw image and the number of color channels
    print("Grayscale image")
    print("Image size (height, width):", image_bw2.shape[:2])
    print("Number of color channels: 1 (cause it is black-shaded, lol)\n")

    # display image
    display_image(image_raw)
    display_image(image_bw2)

    ### TASK 3 ###

    # Flatten the image_bw2 matrix into a 2D array
    image_bw_flattened = image_bw2.reshape(image_bw2.shape[0], -1)

    # Apply PCA to the flattened matrix
    pca = PCA()
    pca.fit(image_bw_flattened)

    # Calculate the cumulative variance explained by each principal component
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100

    # Знаходження кількості компонент для покриття 95% дисперсії
    num_components = np.argmax(cumulative_variance >= 95) + 1

    print("Cumulative variance:", len(cumulative_variance))
    print("Number of components needed to cover 95% of the variance:", num_components)

    # Побудова графіку
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, linewidth=2)
    plt.axvline(x=num_components, color='k', linestyle='--')
    plt.axhline(y=95, color='r', linestyle='-')
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Number of components explaining 95% variance: {num_components}')
    plt.show()

    # Reconstruct the image using the number of components calculated in the previous step
    ipca = IncrementalPCA(n_components=num_components)
    image_reconstructed = ipca.inverse_transform(ipca.fit_transform(image_bw_flattened))

    # Plotting the reconstructed image
    plt.figure(figsize=[12, 8])
    plt.imshow(image_reconstructed.reshape(image_bw2.shape), cmap='gray')
    plt.show()

    ### TASK 4 ###


if __name__ == "__main__":
    main()
