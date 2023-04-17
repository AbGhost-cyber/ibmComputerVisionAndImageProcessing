# Libraries for Data Processing and Visualization
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import copy
from matplotlib.pyplot import imshow
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from ipywidgets import IntProgress
import time
from sklearn.model_selection import train_test_split
import os

# Deep Learning Libraries
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn as nn

torch.manual_seed(0)


def plot_stuff(COST, ACC):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color=color)
    ax1.set_xlabel('Iteration', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()


def imshow_(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.permute(1, 2, 0).numpy()
    print(inp.shape)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


def result(model, x, y):
    # x,y=sample
    z = model(x.unsqueeze_(0))
    _, yhat = torch.max(z.data, 1)

    if yhat.item() != y:
        text = "predicted: {} actual: {}".format(str(yhat.item()), y)
        print(text)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the device type is", device)

train_images = []
train_labels = []

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def label_images(is_not_stop=True):
    label = "not_stop" if is_not_stop else "stop"
    path = "/Users/mac/Downloads/not_stop/" if is_not_stop else "/Users/mac/Downloads/stop/"
    img_array = os.listdir(path)
    for img in img_array:
        if img.__contains__(".DS_Store"):
            continue
        image = Image.open(path + img).convert('RGB')
        # preprocess
        composed = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(degrees=5),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)])
        train_images.append(composed(image))
        train_labels.append(label)


label_images(is_not_stop=False)
label_images()

# i = 0
# for x, y in zip(train_images, train_labels):
#     imshow_(x, "y=: " + y)
#     i += 1
#     if i == 3:
#         break

# set training
percentage_train = 0.9
# convert to numeric and then to tensor
train_labels = LabelEncoder().fit_transform(train_labels)
train_labels = torch.tensor(train_labels)
dataset_list = list(zip(train_images, train_labels))
train_list, test_list = train_test_split(dataset_list, train_size=percentage_train, random_state=1)

# setting hyperparameters
n_epochs = 10
batch_size = 32
lr = 0.000001
momentum = 0.9  # Momentum is a term used in the gradient descent algorithm to improve training results
# for every epoch, using a learning rate scheduler changes the range of the
# learning rate from a maximum or minimum value
lr_scheduler = True
base_lr = 0.001
max_lr = 0.01

# load the pre-trained model resnet 18
model = models.resnet18(pretrained=True)
# train only the output layer
for param in model.parameters():
    param.requires_grad = False

n_classes = len(np.unique(train_labels, return_counts=True)[0])

# Replace the output layer model.fc of the neural network with a nn.Linear object, to classify
# n_classes different classes. For the parameters in_features remember the last hidden layer has 512 neurons.
model.fc = nn.Linear(512, n_classes)
model.to(device)  # set device type
# Cross-entropy loss, or log loss, measures the performance of a classification model combines LogSoftmax
# in one object class. It is useful when training a classification problem with C classes.
criterion = nn.CrossEntropyLoss()

dataset = TensorDataset(*map(torch.stack, zip(*train_list)))
val_set = TensorDataset(*map(torch.stack, zip(*train_list)))
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1)

# will update the weights of the model for us
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# https://arxiv.org/pdf/1506.01186.pdf Cyclical learning rates
if lr_scheduler:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5,
                                                  mode="triangular2")


def train_model(print_=True):
    loss_list = []
    accuracy_list = []
    correct = 0
    # global:val_set
    n_test = len(val_set)
    accuracy_best = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Loop through epochs
    # Loop through the data in loader
    print("The first epoch should take several minutes")
    for epoch in tqdm(range(n_epochs)):

        loss_sublist = []
        # Loop through the data in loader

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.train()

            z = model(x)
            loss = criterion(z, y)
            loss_sublist.append(loss.data.item())
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
        print("epoch {} done".format(epoch))

        scheduler.step()
        loss_list.append(np.mean(loss_sublist))
        correct = 0

        for x_test, y_test in validation_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / n_test
        accuracy_list.append(accuracy)
        if accuracy > accuracy_best:
            accuracy_best = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        if print_:
            print('learning rate', optimizer.param_groups[0]['lr'])
            print("The validaion  Cost for each epoch " + str(epoch + 1) + ": " + str(np.mean(loss_sublist)))
            print("The validation accuracy for epoch " + str(epoch + 1) + ": " + str(accuracy))
    model.load_state_dict(best_model_wts)
    return accuracy_list, loss_list, model


start_datetime = datetime.now()
start_time = time.time()

accuracy_list, loss_list, model = train_model()

end_datetime = datetime.now()
current_time = time.time()
elapsed_time = current_time - start_time
print("elapsed time", elapsed_time)
plot_stuff(loss_list, accuracy_list)
# Save the model to model.pt
torch.save(model.state_dict(), 'model.pt')
# predict
image = Image.open('/Users/mac/Downloads/stop_sign_1.jpeg')
transform = composed = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
x = transform(image)
z = model.forward(x.unsqueeze_(0))
_, yhat = torch.max(z.data, 1)
prediction = "Stop"
if yhat == 1:
    prediction = "Not Stop"

imshow_(transform(image), "stop_sign_1.jpeg" + ": Prediction = " + prediction)

if __name__ == '__main__':
    print()
