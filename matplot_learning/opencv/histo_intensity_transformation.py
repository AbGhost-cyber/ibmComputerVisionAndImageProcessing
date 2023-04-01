import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image_1, image_2, title_1="Original", title_2="New Image"):
    plt.figure(figsize=(7, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap="gray")
    plt.title(title_2)
    plt.show()


def plot_hist(old_image, new_image, title_old="Original", title_new="New Image"):
    intensity_values = np.array([x for x in range(256)])
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image], [0], None, [256], [0, 256])[:, 0], width=5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image], [0], None, [256], [0, 256])[:, 0], width=5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()


toy_image = np.array([[0, 2, 2], [1, 1, 1], [1, 1, 2]], dtype=np.uint8)
neg_toy_image = -1 * toy_image + 255
# plt.figure(figsize=(7, 7))
# plt.subplot(1, 2, 1)
# plt.imshow(toy_image, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(neg_toy_image, cmap='gray')
# plt.show()

# plt.bar([x for x in range(6)], [1, 5, 2, 0, 0, 0])
# plt.show()

# GRAY SCALE HISTOGRAM
gold_hill = cv2.imread("../images/goldhill.bmp", cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([gold_hill], [0], None, [256], [0, 256])
intensity_values = np.array([x for x in range(hist.shape[0])])
# plt.bar(intensity_values, hist[:, 0], width=5)
# plt.title("Bar histogram")
# plt.show()

# a probability mass function (PMF) describes the probability distribution of a discrete random variable.
# It provides the probability of occurrence for each possible value of the random variable.
PMF = hist / (gold_hill.shape[0] * gold_hill.shape[1])

color = ('blue', 'green', 'red')

baboon = cv2.imread("../images/baboon.png")

# for i, col in enumerate(color):
#     histr = cv2.calcHist(images=[baboon], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
#     plt.plot(intensity_values, histr, color=col, label=col + ' channel')
#     plt.xlim([0, 256])
# plt.legend()
# plt.title("Histogram Channels")
# plt.show()

# reverse image intensity
image = cv2.imread("../images/mammogram.png", cv2.IMREAD_GRAYSCALE)
img_neg = - 1 * image + np.copy(image).shape[0]
cv2.rectangle(image, pt1=(160, 212), pt2=(250, 289), color=255, thickness=2)
# plt.figure(figsize=(6, 6))
# plt.imshow(image, cmap="gray")
# plt.show()

# brightness and contrast adjustment
alpha = 2  # simple contrast control
beta = 0  # simple brightness control
new_image = cv2.convertScaleAbs(gold_hill, alpha=alpha, beta=beta)
# plot_image(gold_hill, new_image, title_2="brightness control")
# plot_hist(gold_hill, new_image)

zelda = cv2.imread("../images/zelda.png", cv2.IMREAD_GRAYSCALE)
new_image = cv2.equalizeHist(zelda)
# plot_image(zelda, new_image, "Orignal", "Histogram Equalization")


# TODO how to flatten an array

def thresholding(input_img, threshold, max_value=255, min_value=0):
    N, M = input_img.shape
    image_out = np.zeros((N, M), dtype=np.uint8)

    for i in range(N):
        for j in range(M):
            if input_img[i, j] > threshold:
                image_out[i, j] = max_value
            else:
                image_out[i, j] = min_value

    return image_out


# threshold toy image
threshold = 1
max_value = 2
min_value = 0
thresholding_toy = thresholding(toy_image, threshold=threshold, max_value=max_value, min_value=min_value)
# plt.figure(figsize=(7, 7))
# plt.subplot(1, 2, 1)
# plt.imshow(toy_image, cmap="gray")
# plt.title("Original Image")
# plt.subplot(1, 2, 2)
# plt.imshow(thresholding_toy, cmap="gray")
# plt.title("Image After Thresholding")
# plt.show()

# thresholding camera man
threshold = 87
max_value = 255
min_value = 0
image = cv2.imread("../images/cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
new_image = thresholding(image, threshold=threshold, max_value=max_value, min_value=min_value)
plot_image(image, new_image, "Orignal", "Image After Thresholding")
if __name__ == '__main__':
    print()
    # print("toy image\n", toy_image)
    # print("image negatives\n", neg_toy_image)
