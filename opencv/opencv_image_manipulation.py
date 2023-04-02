import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_img(src):
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    plt.show()


image = cv2.imread("../images/cat.png")

# flipping using flip method, flipcode = 0 flip in x-axis positive value,
# flipcode > 0 flip in y-axis positive value
# flipcode < 0 flip in x and y axis negative value
for flipcode in [0, 1, -1]:
    im_flip = cv2.flip(image, flipcode)
    # plt.imshow(cv2.cvtColor(im_flip, cv2.COLOR_BGR2RGB))
    # plt.title("flipcode: " + str(flipcode))
    # plt.show()

# using rotate
flip = {"ROTATE_90_CLOCKWISE": cv2.ROTATE_90_CLOCKWISE,
        "ROTATE_90_COUNTERCLOCKWISE": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "ROTATE_180": cv2.ROTATE_180}
# for key, value in flip.items():
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title("orignal")
#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(cv2.rotate(image, value), cv2.COLOR_BGR2RGB))
#     plt.title(key)
#     plt.show()









# cropping
upper = 150
lower = 400
left = 150
right = 400

crop_top = image[upper: lower, :, :]

# changing specific image pixels
array_sq = np.copy(image)
array_sq[upper:lower, left:right, 0:2] = 0

# plt.figure(figsize=(7, 7))
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("orignal")
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(array_sq, cv2.COLOR_BGR2RGB))
# plt.title("Altered Image")
# plt.show()

start_point, end_point = (left, upper), (right, lower)
image_draw = np.copy(image)
cv2.rectangle(image_draw, pt1=start_point, pt2=end_point, color=(0, 255, 0), thickness=3)
image_draw = cv2.putText(img=image, text='Stuff', org=(30, 512),
                         color=(255, 255, 255), fontFace=4, fontScale=5,
                         thickness=2)
show_img(image_draw)

if __name__ == '__main__':
    print()
