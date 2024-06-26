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
    return


### TASK 5 ###
def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector


def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    decrypted_vector = np.dot(np.linalg.inv(diagonalized_key_matrix), encrypted_vector)
    decrypted_message = "".join([chr(int(np.round(np.real(char)))) for char in decrypted_vector])

    return decrypted_message


def main():
    message = "Hello, world!"
    key_matrix = np.random.randint(0, 256, (len(message), len(message)))
    print("Original message:", message)
    encrypted_message = encrypt_message(message, key_matrix)
    print("Encrypted message:", encrypted_message)
    decrypted_message = decrypt_message(encrypted_message, key_matrix)
    print("Decrypted message:", decrypted_message)


    ### TASK 1 ###
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

    c = [5, 15, 30, 50, 100, 200]
    plt.figure(figsize=[20, 15])

    for i in range(6):
        # Apply IncrementalPCA with the current number of components
        ipca = IncrementalPCA(n_components=c[i])
        image_reconstructed = ipca.inverse_transform(ipca.fit_transform(image_bw_flattened))

        # Create a subplot for the current reconstructed image
        plt.subplot(2, 3, i + 1)
        plt.imshow(image_reconstructed.reshape(image_bw2.shape), cmap='gray')
        plt.title(f'Reconstructed image with {c[i]} components')

        # Display the figure with all subplots
    plt.tight_layout()
    plt.show()
    return



if __name__ == "__main__":
    main()
