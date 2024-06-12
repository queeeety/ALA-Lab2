import numpy as np


def checker(eigenvalues, eigenvectors, matrix):
    for i in range(len(eigenvalues)):
        if not np.allclose(np.dot(matrix, eigenvectors[:, i]), eigenvalues[i] * eigenvectors[:, i]):
            return False
    return True


def main():
    matrix = np.array([[7, 2, 3], [2, 5, 1], [3, 1, 11]])

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if not checker(eigenvalues, eigenvectors, matrix):
        print("Something went wrong")
        return

    print("Everything is correct")
    print("Eigenvalues: ", eigenvalues)
    print("Eigenvectors: ", eigenvectors)


if __name__ == "__main__":
    main()