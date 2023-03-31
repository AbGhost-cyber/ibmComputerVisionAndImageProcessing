import cv2
import matplotlib.pyplot as plt

my_image = "images/lenna.png"

# the image won't look normal cause the order of RGB channels are different in OpenCV
image = cv2.imread(my_image, cv2.IMREAD_GRAYSCALE)
# we then convert from BGR to RGB
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(7, 7))
# plt.imshow(new_image, cmap='gray')
# plt.show()

# with color channels
baboon = cv2.imread("images/baboon.png")
# plt.figure(figsize=(7, 7))
# plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
# plt.show()

blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]
im_bgr = cv2.vconcat([blue, green, red])
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title('RGB Image')
plt.subplot(122)
plt.imshow(im_bgr, cmap='gray')
plt.title("Different color channels blue(top), green (middle), red(bottom)")
plt.show()
if __name__ == '__main__':
    print()
