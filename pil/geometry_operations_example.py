import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np


def plot_image(image_1, image_2, title_1="Original", title_2="New Image"):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap="gray")
    plt.title(title_2)
    plt.show()


image = Image.open("../images/lenna.png")
# plt.imshow(image)
# plt.show()

width, height = image.size
new_width = 2 * width
new_height = height
new_image = image.resize((new_width, new_height))
new_image = new_image.rotate(angle=45)
# plt.imshow(new_image)
# plt.show()

# array operations
image = np.array(image)
# new_image = image * 10  # adds 20 to every pixel in the array
Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8)
new_image = image + Noise
# plt.imshow(new_image)
# plt.show()

# matrix operation, gray scale images are matrices
im_gray = Image.open("../images/barbara.png")
im_gray = ImageOps.grayscale(im_gray)
im_gray = np.array(im_gray.copy())
# plt.imshow(im_gray, cmap='Accent')
# plt.show()

U, s, V = np.linalg.svd(im_gray, full_matrices=True)
S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
S[:image.shape[0], :image.shape[0]] = np.diag(s)
# plot_image(U, V, title_1="Matrix U", title_2="Matrix V")

# This code represents a loop that iteratively decreases the number of components used
# in a singular value decomposition (SVD) of a given matrix.In particular, the loop considers
# 5 different values for the number of components (n_component), starting from 1 and increasing by
# a factor of 10 until 500 is reached. For each value of n_component, a new truncated SVD is computed
# from the original SVD decomposition, using only the first n_component columns of the matrix S and
# the first n_component rows of the matrix V. The truncated SVD is then used to reconstruct the original
# matrix A, by computing the product of the matrix U, the truncated matrix S_new, and the truncated matrix
# V_new. This approximated matrix is displayed as a grayscale image using the `imshow` function of matplotlib library.
for n_component in [1, 10, 100, 200, 500]:
    S_new = S[:, :n_component]
    V_new = V[:n_component, :]
    A = U.dot(S_new.dot(V_new))
    plt.imshow(A, cmap='gray')
    plt.title("Number of Components:" + str(n_component))
    plt.show()
if __name__ == '__main__':
    print(new_image.shape)
