import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
# used to help create the dataset and perform the mini-batch
from torch.utils.data import Dataset, DataLoader
# pyTorch Neural Network
import torch.nn as nn
import tensorflow as tf


class plot_error_surfaces(object):

    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples=30, go=True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                yhat = 1 / (1 + np.exp(-1 * (w2 * self.x + b2)))
                Z[count1, count2] = -1 * np.mean(
                    self.y * np.log(yhat + 1e-16) + (1 - self.y) * np.log(1 - yhat + 1e-16))
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize=(7.5, 5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1, cmap='viridis',
                                                   edgecolor='none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()

    # Setter
    def set_para_loss(self, model, loss):
        self.n = self.n + 1
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())
        self.LOSS.append(loss)

    # Plot diagram
    def final_plot(self):
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c='r', marker='x', s=200, alpha=1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()

    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x[self.y == 0], self.y[self.y == 0], 'ro', label="training points")
        plt.plot(self.x[self.y == 1], self.y[self.y == 1] - 1, 'o', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label="estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-0.1, 2))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.show()
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.title('Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')


# Plot the diagram

def PlotStuff(X, Y, model, epoch, leg=True):
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    if leg == True:
        plt.legend()
    else:
        pass


# Setting the seed will allow us to control randomness and give us reproducibility
torch.manual_seed(0)


# Create the custom Data class which inherits Dataset
class Data(Dataset):

    # Constructor
    def __init__(self):
        # Create X values from -1 to 1 with step .1
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1)
        # Create Y values all set to 0
        self.y = torch.zeros(self.x.shape[0], 1)
        # Set the X values above 0.2 to 1
        self.y[self.x[:, 0] > 0.2] = 1
        # Set the .len attribute because we need to override the __len__ method
        self.len = self.x.shape[0]

    # Getter that returns the data at the given index
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get length of the dataset
    def __len__(self):
        return self.len


# Create logistic_regression class that inherits nn.Module which is the base class for all neural networks
class logistic_regression(nn.Module):

    # Constructor
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        # Single layer of Logistic Regression with number of inputs being n_inputs and there being 1 output
        # This linear layer computes the weighted sum of the input features, which is then passed
        # into the sigmoid activation function in the forward method.
        self.linear = nn.Linear(in_features=n_inputs, out_features=1)

    # Prediction
    def forward(self, x):
        # first computes the weighted sum of feature x by passing x through the linear layer,
        # then it applies the sigmoid activation function to produce the output probability
        yhat = torch.sigmoid(self.linear(x))
        return yhat


data_set = Data()
model = logistic_regression(n_inputs=1)
criterion = nn.BCELoss()
x, y = data_set[0]

# Create the plot_error_surfaces object

# 15 is the range of w
# 13 is the range of b
# data_set[:][0] are all the X values
# data_set[:][1] are all the Y values

# get_surface = plot_error_surfaces(15, 13, data_set[:][0], data_set[:][1])

# x, y = data_set[0]
# print("x = {},  y = {}".format(x, y))
# plt.plot(data_set.x[data_set.y == 0], data_set.y[data_set.y == 0], 'ro', label="y=0")
# plt.plot(data_set.x[data_set.y == 1], data_set.y[data_set.y == 1]-1, 'o', label="y=1")
# plt.xlabel('x')
# plt.legend()
# plt.show()
if __name__ == '__main__':
    print()
