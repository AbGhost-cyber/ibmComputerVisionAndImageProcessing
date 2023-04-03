import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image_1, image_2, title_1="Original", title_2="New Image", is_gray=True):
    cmap = 'gray' if is_gray else "viridis"
    plt.figure(figsize=(7, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap=cmap)
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap=cmap)
    plt.title(title_2)
    plt.show()


image = cv2.imread("../images/lenna.png")
noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
noisy_image = image + noise
# plot_image(image, noisy_image, title_1="Original", title_2="Image Plus Noise")

# filtering noise
# Create a kernel which is a 6 by 6 array where each value is 1/36
kernel = np.ones((6, 6)) / 36
# Filters the images using the kernel, filter2d performs 2d filtering convolution
image_filtered = cv2.filter2D(src=noisy_image, ddepth=-1, kernel=kernel)

# filter the image using gaussian blur
# The `sigmaX` and `sigmaY` parameters control the Gaussian kernel standard
# deviations in the horizontal and vertical directions, higher value wil make the image blurry
image_filtered = cv2.GaussianBlur(noisy_image, (5, 5), sigmaX=10, sigmaY=10)
# plot_image(image_filtered, noisy_image, title_1="Filtered image", title_2="Image Plus Noise")

# sharpening
# Common Kernel for image sharpening
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])
# Applys the sharpening filter using kernel on the original image without noise
sharpened = cv2.filter2D(image, -1, kernel)
# plot_image(sharpened, image, title_1="Sharpened image", title_2="Image")

# Edge detection
img_gray = cv2.imread('../images/coleen_gray.jpeg', cv2.IMREAD_GRAYSCALE)
# apply Gaussian blur to smooth the image edges and reduce image noise,
# sets the blur kernel size to 3X3 which means the filter will look at a 3X3 pixel window at a time
img_gray = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=0.1, sigmaY=0.1)
# 16-bit integer
ddepth = cv2.CV_16S
# Applies the filter on the image in the X direction, the dx param is set to 1 to indicate
# that we want to detect edges in the horizontal direction and dy 0 to indicate that
# we don't want to detect edges in the vertical direction
grad_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
# Applies the filter on the image in the Y direction
grad_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
# Converts the values back to a number between 0 and 255
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
# Adds the derivative in the X and Y direction
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
# plot and compare edge detected image with original
# plot_image(img_gray, grad, title_2="Edge Detected Image")

# median filtering
image = cv2.imread("../images/cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
filtered_image = cv2.medianBlur(image, ksize=7)
# plt.imshow(filtered_image,cmap="gray")
# plt.show()

# thresholding
ret, otsu = cv2.threshold(src=image, thresh=0, maxval=255, type=cv2.THRESH_OTSU + cv2.THRESH_BINARY)
plt.imshow(otsu, cmap="gray")
plt.show()
if __name__ == '__main__':
    print()
