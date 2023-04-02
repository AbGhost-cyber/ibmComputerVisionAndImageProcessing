import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

baboon = np.array(Image.open("../images/baboon.png"))


def show_img(src):
    plt.figure(figsize=(5, 5))
    plt.imshow(src)
    plt.show()


# Flipping images involves reordering the indices of the pixels such that it changes the orientation of the image
image = Image.open("../images/cat.png")
array = np.array(image)
width, height, color = array.shape

array_flip = np.zeros(array.shape, dtype=np.uint8)

# traditional flipping, we assign the first row of pixels of the original array to the new array's last row
# we repeat the process for every row, incrementing the row number from the original array and decreasing
# the new array's index to assign the pixel accordingly.
for i, row in enumerate(array):
    array_flip[width - 1 - i, :] = row

# using PIL
im_flip = ImageOps.flip(image)
im_flip = im_flip.transpose(1)
# show_img(im_flip)

im_mirror = ImageOps.mirror(image)
# show_img(im_mirror)

flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
        "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
        "ROTATE_90": Image.ROTATE_90,
        "ROTATE_180": Image.ROTATE_180,
        "ROTATE_270": Image.ROTATE_270,
        "TRANSPOSE": Image.TRANSPOSE,
        "TRANSVERSE": Image.TRANSVERSE}
# for key, values in flip.items():
#     plt.figure(figsize=(7, 7))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("original")
#     plt.subplot(1, 2, 2)
#     plt.imshow(image.transpose(values))
#     plt.title(key)
#     plt.show()

# cropping using traditional method and PIL functions
upper = 150
lower = 400
left = 150
right = 400
crop_top = array[upper: lower, :, :]
crop_horizontal = crop_top[:, 150:400, :]

image = Image.open("../images/cat.png")
crop_image = image.crop((left, upper, right, lower))

# changing specific image pixels
array_sq = np.copy(array)
array_sq[upper: lower, left:right, 1:2] = 0

# drawing over an image
image_draw = image.copy()
shape = [left, upper, right, lower]
image_fn = ImageDraw.Draw(im=image_draw)
image_fn.text(xy=(0, 0), text="box", fill=(0, 0, 0))
image_fn.rectangle(xy=shape, fill='red')
image_draw.paste(crop_image, box=(left, upper))
show_img(image_draw)
if __name__ == '__main__':
    print(array_sq.shape)
