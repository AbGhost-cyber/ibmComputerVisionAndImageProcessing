import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
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


toy_image = np.zeros((6, 6))
toy_image[1:5, 1:5] = 255
toy_image[2:4, 2:4] = 0
# plt.imshow(toy_image, cmap='gray')
# plt.show()

# rescale
# fx: scale factor along the horizontal axis
# fy: scale factor along the vertical axis
# INTER_NEAREST uses the nearest pixel
new_toy = cv2.resize(toy_image, None, fx=2, fy=1, interpolation=cv2.INTER_NEAREST)
# plt.imshow(new_toy, cmap='gray')
# plt.show()

image = cv2.imread("../images/lenna.png")
new_image = cv2.resize(image, None, fx=0.5, fy=1, interpolation=cv2.INTER_NEAREST)
# new_image1 = cv2.resize(image, (1, 2), interpolation=cv2.INTER_NEAREST)  # silly method
# plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
# plt.show()

# translation: Translation is when you shift the location of the image.
# tx is the number of pixels you shift the location in the horizontal direction
# and ty is the number of pixels you shift in the vertical direction.
# You can create the transformation matrix  𝑀 to shift the image.
tx = 0
ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])
rows, cols, _ = image.shape
new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
crop_image = new_image[ty:, :, :]
# plt.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
# plt.show()

# rotation: We can rotate an image by angle θ which is achieved by the Rotation Matrix getRotationMatrix2D.
# center: Center of the rotation in the source image. We will only use the center of the image.
# angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation
# (the coordinate origin is assumed to be the top-left corner).
# scale: Isotropic scale factor, in this course the value will be one.
theta = 45.0
M = cv2.getRotationMatrix2D(center=(3, 3), angle=theta, scale=1)
new_toy_image = cv2.warpAffine(toy_image, M, (6, 6))
# plot_image(toy_image, new_toy_image, title_1="Original", title_2="rotated image")
M = cv2.getRotationMatrix2D(center=(cols // 2 - 1, rows // 2 - 1), angle=theta, scale=1)
new_image = cv2.warpAffine(image, M, (cols, rows))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
if __name__ == '__main__':
    print(cols // 2 - 1)
