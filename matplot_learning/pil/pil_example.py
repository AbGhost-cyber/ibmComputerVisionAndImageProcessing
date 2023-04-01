from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
import numpy as np


# import pandas as pd

def get_concat_h(img1, img2):
    dst = Image.new('RGB', (img1.width + img2.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    return dst


my_image = "matplot_learning/images/lenna.png"
cwd = os.getcwd()
image_path = os.path.join(cwd, my_image)

image = Image.open(my_image)
# plt.imshow(image)
# plt.show()
im = image.load()
x = 0
y = 1
image_gray = ImageOps.grayscale(image)
# image_gray.quantize(256 // 3)
# image_gray.show()

# quantization¬
# for n in range(3, 8):
#     plt.figure(figsize=(7, 7))
#     plt.imshow(get_concat_h(image_gray, image_gray.quantize(256 // 2 ** n)))
#     plt.title('256 Quantization Levels left vs {} Quantization Levels right'.format(256 // 2 ** n))
# plt.show()

# color channels
baboon = Image.open("../images/baboon.png")
baboon_array = np.array(baboon)
red, green, blue = baboon.split()
# plt.imshow(get_concat_h(baboon, blue))
# plt.show()

# with Numpy
array = np.array(image)
print(array.shape)  # returns rows, columns and shapes

# plot the array as an image
plt.figure(figsize=(7, 7))
plt.imshow(array[:, 0: 256, :])
plt.show()

baboon_red = baboon_array.copy()
baboon_red[:, :, 1] = 0
baboon_red[:, :, 2] = 0
plt.figure(figsize=(7, 7))
plt.imshow(baboon_red)
plt.show()

if __name__ == '__main__':
    print(array.max())
    print(array.min())
