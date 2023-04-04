import cv2
import numpy as np
import pandas as pd
import time
import random
import json
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

root_folder = "./cats_dogs_images/"
with open(root_folder + "_annotations.json") as path:
    annotations = json.load(path)


# function to extract images from our json
def get_image_paths():
    file_paths = []
    for file_name in annotations['annotations'].keys():
        if file_name.endswith(".jpg"):
            file_path = root_folder + file_name
            file_paths.append(file_path)
    return file_paths


image_paths = get_image_paths()
train_images = []
train_labels = []
class_object = annotations['labels']

for (i, image_path) in enumerate(image_paths):
    # read image and convert to grayscale
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    # label image using annotations
    short_img_path = image_path[len(root_folder):]
    img_label = annotations['annotations'][short_img_path][0]['label']
    label = class_object.index(img_label)
    tmp_label = img_label
    # resize image
    image = cv2.resize(image, (32, 32))
    # flatten the image
    pixels = image.flatten()
    # Append flattened image to
    train_images.append(pixels)
    train_labels.append(label)
    # print('Loaded...', '\U0001F483', 'Image', str(i + 1), 'is a', tmp_label)

# we need to convert the images to float32 because openCv only identifies such array types for the training samples
# and array of shape (label size, 1) for the training labels
train_images = np.array(train_images).astype('float32')
train_labels = np.array(train_labels).astype(int)
train_labels = train_labels.reshape((train_labels.size, 1))

# split data into training anf test set
test_size = 0.2
train_samples, test_samples, train_labels, test_labels = train_test_split(train_images, train_labels,
                                                                          test_size=test_size, random_state=0)

# we will use cv2 KNN

start_datetime = datetime.now()

knn = cv2.ml.KNearest_create()
knn.train(train_samples, cv2.ml.ROW_SAMPLE, train_labels)

# get different values of K
# We will try multiple values of k to find the optimal value for the dataset we have.
# k refers to the number of nearest neighbours to include in the majority of the voting process.
k_values = [1, 2, 3, 4, 5]
k_result = []
for k in k_values:
    ret, result, neighbors, dist = knn.findNearest(test_samples, k=k)
    k_result.append(result)
flattened = []

for res in k_result:
    flat_result = [item for sublist in res for item in sublist]
    flattened.append(flat_result)
end_datetime = datetime.now()
# print('Training Duration: ' + str(end_datetime - start_datetime))

# evaluation
accuracy_res = []
con_matrix = []
for k_res in k_result:
    label_names = [0, 1]
    cmx = confusion_matrix(test_labels, k_res, labels=label_names)
    con_matrix.append(cmx)
    # get values for when we predict accurately
    matches = k_res == test_labels
    correct = np.count_nonzero(matches)
    # calculate accuracy
    accuracy = correct * 100.0 / result.size
    accuracy_res.append(accuracy)
    accuracy_res.append(accuracy)
# stor accuracy for later when we create the graph
res_accuracy = {k_values[i]: accuracy_res[i] for i in range(len(k_values))}
list_res = sorted(res_accuracy.items())

# plot for confusion_matrix
t = 0
# for each value of k we will create a confusion matrix
# for array in con_matrix:
#     df_cm = pd.DataFrame(array)
#     sns.set(font_scale=1.4)  # for label size
#     sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".0f")  # font size
#     t += 1
#     title = "Confusion Matrix for k equals " + str(t)
#     plt.title(title)
#     plt.show()
x, y = zip(*list_res)
# plt.plot(x, y)
# plt.show()

# We will get the best value of k to train the model to test the model on our image:
k_best = max(list_res,key=lambda item:item[1])[0]
print(k_best)
if __name__ == '__main__':
    print()
