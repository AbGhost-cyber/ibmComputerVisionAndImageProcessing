import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np


def plot_image(image_1, image_2, title_1="Orignal", title_2="New Image"):
    plt.figure(figsize=(7, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)
    plt.title(title_2)
    plt.show()


# Loads the image from the specified file
image = Image.open("../images/lenna.png")
rows, cols = image.size
# Creates values using a normal distribution with a mean of 0 and standard deviation of 15,
# the values are converted to unit8 which means the values are between 0 and 255
noise = np.random.normal(0, 15, (rows, cols, 3)).astype(np.uint8)
# add the noise to the image
noisy_image = image + noise
noisy_image = Image.fromarray(noisy_image)
# plot_image(image, noisy_image, title_1="Original", title_2="Image Plus Noise")

# filter noise
# Smoothing filters average out the Pixels within a neighborhood,
# they are sometimes called low pass filters. For mean filtering,
# the kernel simply averages out the kernels in a neighborhood.

# Create a kernel which is a 5 by 5 array where each value is 1/36
kernel = np.ones((5, 5)) / 36
kernel_filter = ImageFilter.Kernel((5, 5), kernel.flatten())
# The function filter performs a convolution between the image and the kernel on each color channel independently.
image_filtered = noisy_image.filter(kernel_filter)
# Plots the Filtered and Image with Noise using the function defined at the top
# plot_image(image_filtered, noisy_image, title_1="Filtered image", title_2="Image Plus Noise")

# using gaussian blur
# accuracy <https://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf>
# To perform Gaussian Blur we use the filter function on an image using the predefined filter ImageFilter.GaussianBlur
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur(4))
# plot_image(image_filtered, noisy_image, title_1="Filtered image", title_2="Image Plus Noise")

# image sharpening
# common kernel for image sharpening, we can also sharpen using a predefined filter from PIL SHARPEN
kernel = np.array([[-1, -1, -1,
                    -1, 9, -1,
                    -1, -1, -1]])
kernel = ImageFilter.Kernel((3, 3), kernel.flatten())
# Applys the sharpening filter using kernel on the original image without noise
sharpened = image.filter(kernel)
# Plots the sharpened image and the original image without noise
# plot_image(sharpened, image, title_1="Sharpened image", title_2="Image")

# Edges
img_gray = Image.open('../images/cameraman.jpeg')
img_gray = img_gray.filter(ImageFilter.MedianFilter)
plt.figure(figsize=(7, 7))
# Renders the image
plt.imshow(img_gray, cmap="gray")
plt.show()
if __name__ == '__main__':
    print()
