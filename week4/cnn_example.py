# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to transform data
import torchvision.transforms as transforms
# Allows us to download the dataset
import torchvision.datasets as dsets
# Used to graph data and loss curves
import matplotlib.pylab as plt
# Allows us to use arrays to manipulate and store data
import numpy as np

IMAGE_SIZE = 16


# Define the function for plotting out the kernel parameters of each channels
def plot_channels(W):
    n_out = W.shape[0]
    n_in = W.shape[1]
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(n_out, n_in)
    fig.subplots_adjust(hspace=0.1)
    out_index = 0
    in_index = 0

    # plot outputs as rows, inputs as columns
    for ax in axes.flat:
        if in_index > n_in - 1:
            out_index = out_index + 1
            in_index = 0
        ax.imshow(W[out_index, in_index, :, :], vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        in_index = in_index + 1

    plt.show()


# Define the function for plotting out the kernel parameters of each channel with Multiple outputs
def plot_parameters(W, number_rows=1, name="", i=0):
    W = W.data[:, i, :, :]
    n_filters = W.shape[0]
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(number_rows, n_filters // number_rows)
    fig.subplots_adjust(hspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Set the label for the sub-plot.
            ax.set_xlabel("kernel:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(W[i, :], vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle(name, fontsize=10)
    plt.show()


# Define the function for plotting the activations of the convolution layers
def plot_activations(A, number_rows=1, name="", i=0):
    A = A[0, :, :, :].detach().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()
    fig, axes = plt.subplots(number_rows, n_activations // number_rows)
    fig.subplots_adjust(hspace=0.9)

    for i, ax in enumerate(axes.flat):
        if i < n_activations:
            # Set the label for the sub-plot.
            ax.set_xlabel("activation:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


# function for plotting out data samples as images.
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))
    plt.show()


# First the image is resized then converted to a tensor
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

# load the training dataset by setting the parameter train to True
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)

validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)


# show_data(train_dataset[3])

class CNN(nn.Module):

    # number of output channels for first and second layers, out_1, out_2
    # channel width = no of feature maps in a convolutional layer where each map represents a feature
    def __init__(self, out_1=16, out_2=32):
        super(CNN, self).__init__()
        #  we start with 1 channel because we have a single black and white image
        # Channel Width after this layer is 16
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Channel Width after this layer is 8
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        # Channel Width after this layer is 4
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # In total, we have out_2 (32) channels which are each 4 * 4 in
        # size based on the width calculation above. Channels are squares.

        # The output is a value for each class
        self.fc1 = nn.Linear(out_2 * 4 * 4, 10)

    # prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu_(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu_(x)
        x = self.maxpool2(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

        # Outputs result of each stage of the CNN, relu, and pooling layers

    def activations(self, x):
        # Outputs activation this is not necessary
        z1 = self.cnn1(x)
        a1 = torch.relu(z1)
        out = self.maxpool1(a1)

        z2 = self.cnn2(out)
        a2 = torch.relu(z2)
        out1 = self.maxpool2(a2)
        out = out.view(out.size(0), -1)
        return z1, a1, z2, a2, out1, out


# Create the model object using CNN class, 16 output channel for the first layer, 32 for the second

model = CNN(out_1=16, out_2=32)
plot_parameters(model.state_dict()['cnn1.weight'], number_rows=4, name="1st layer kernels before training ")
plot_parameters(model.state_dict()['cnn2.weight'], number_rows=4, name='2nd layer kernels before training')

if __name__ == '__main__':
    print()
