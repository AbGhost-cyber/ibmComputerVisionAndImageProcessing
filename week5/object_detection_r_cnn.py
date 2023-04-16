import torch
from torchvision import transforms, models
from torch import no_grad

# libraries for image processing
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# This function will assign a string name to a predicted class and eliminate
# predictions whose likelihood is under a threshold
# pred: a list where each element contains a tuple that corresponds to information about the different objects;
# Each element includes a tuple with the class yhat, probability of belonging to that class and the coordinates
# of the bounding box corresponding to the object
# image: frozen surface
# predicted_classes: a list where each element contains a tuple that corresponds to information about
# the different objects; Each element includes a tuple with the class name, probability of belonging to
# that class and the coordinates of the bounding box corresponding to the object
#
def get_predictions(pred, threshold=0.8, objects=None):
    predicted_classes = [(COCO_INSTANCE_CATEGORY_NAMES[i], p, [(box[0], box[1]), (box[2], box[3])]) for i, p, box in
                         zip(list(pred[0]['labels'].numpy()), pred[0]['scores'].detach().numpy(),
                             list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes = [stuff for stuff in predicted_classes if stuff[1] > threshold]

    if objects and predicted_classes:
        predicted_classes = [(name, p, box) for name, p, box in predicted_classes if name in objects]
    return predicted_classes


# draw box around each object
# predicted_classes: a list where each element contains a tuple that corresponds to information about
# the different objects; Each element includes a tuple with the class name, probability of belonging to that
# class and the coordinates of the bounding box corresponding to the object
def draw_box(predicted_classes, image, rect_th=10, text_size=3, text_th=3):
    img = (np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)), 0, 1), cv2.COLOR_RGB2BGR), 0,
                   1) * 255).astype(np.uint8).copy()
    for predicted_class in predicted_classes:
        label = predicted_class[0]
        probability = predicted_class[1]
        box = predicted_class[2]
        box[0] = [round(x) for x in box[0]]
        box[1] = [round(x) for x in box[1]]

        cv2.rectangle(img, box[0], box[1], (0, 255, 0), rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img, label, box[0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
        cv2.putText(img, label + ": " + str(round(probability, 2)), box[0], cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0, 255, 0), thickness=text_th)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    del img
    del image


# this function will free up some memory:
def save_RAM(image_=False):
    global image, img, pred
    torch.cuda.empty_cache()
    del img
    del pred
    if image_:
        image.close()
        del image


model_ = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()

# Transfer learning involves using a pre-trained model to solve a related task to the original training task.
# this is why we need to freeze the model's parameters
for name, param in model_.named_parameters():
    # This freezes the parameters so that they are not updated during training.
    param.requires_grad = False
print("done")


def model(x):
    with no_grad():
        yhat = model_(x)
    return yhat


image = Image.open("images/DLguys.jpeg")
# images/jeff_hinton.png
half = 0.5
height, width = image.size
# image.resize([int(half * s) for s in image.size])
image.resize((int(height * half), int(width * half)))

# converts the image to a PyTorch Tensor of shape `(C, H, W)
transform = transforms.Compose([transforms.ToTensor()])
img = transform(image)

# # make a prediction
# pred = model([img])
#
# # likelihood or prob of each class
# mscore = pred[0]['scores'][0]
#
# # the class number corresponds to the index of the list with the corresponding category name
# index = pred[0]['labels'][0].item()
# mlabel = COCO_INSTANCE_CATEGORY_NAMES[index]
#
# # we can have the coordinates of the bounding box
# bounding_box = pred[0]['boxes'][0].tolist()
# # round the components of the bounding box
# top, left, bottom, right = [round(x) for x in bounding_box]

# convert tensor into open cv array anf plot image
# clipped_img = np.clip(
#     cv2.cvtColor(np.clip(img.numpy().transpose((1, 2, 0)), 0, 1), cv2.COLOR_RGB2BGR), 0, 1)
# img_plot = (clipped_img * 255).astype(np.uint8)
# cv2.rectangle(img_plot, (top, left), (bottom, right), (0, 255, 0), 10)  # draw rect with coordinates
# plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
# plt.show()
# del img_plot, top, left, right, bottom  # save memory

# pred_class = get_predictions(pred, objects="person")
# draw_box(pred_class, img, rect_th=1, text_size=0.5, text_th=1)
# del pred_class
if __name__ == '__main__':
    print(img.shape)
