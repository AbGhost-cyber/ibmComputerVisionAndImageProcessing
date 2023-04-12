# Used to graph data and loss curves
import matplotlib.pyplot as plt
# Allows us to use arrays to manipulate and store data
import numpy as np
# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to use activation functions
import torch.nn.functional as F
# Used to graph data and loss curves
from matplotlib.colors import ListedColormap
# Used to help create the dataset and perform mini-batch
from torch.utils.data import Dataset, DataLoader

# ensures that the same set of random numbers will be generated every time the code is run
torch.manual_seed(1)
np.random.seed(1)


# Define a function to plot the decision region

def plot_decision_regions_3class(model, data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _, yhat = torch.max(model(XX), 1)
    yhat = yhat.numpy().reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light, shading='auto')
    plt.plot(X[y[:] == 0, 0], X[y[:] == 0, 1], 'ro', label='y=0')
    plt.plot(X[y[:] == 1, 0], X[y[:] == 1, 1], 'go', label='y=1')
    plt.plot(X[y[:] == 2, 0], X[y[:] == 2, 1], 'o', label='y=2')
    plt.title("decision region")
    plt.legend()
    plt.show()


# Create the dataset class

class Data(Dataset):

    # K = number of classes
    # N = number of data points per class
    def __init__(self, K=3, N=500):
        # two dimensional data points
        D = 2
        X = np.zeros((N * K, D))
        y = np.zeros(N * K, dtype='uint8')  # class labels
        for j in range(K):
            # range of indices that correspond to the subset of `X` and `y` for the current class K.
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.rand(N) * 0.2  # theta
            # np.c_ function is used for concatenating the `sin` and `cos` values along
            # the second axis to create a 2D array of shape `(N, 2)` where the first column contains
            # the values of x and second column contains y values
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.len = y.shape[0]

        # Getter

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def plot_data(self):
        plt.plot(self.x[self.y[:] == 0, 0].numpy(), self.x[self.y[:] == 0, 1].numpy(), 'o', label="y=0")
        plt.plot(self.x[self.y[:] == 1, 0].numpy(), self.x[self.y[:] == 1, 1].numpy(), 'ro', label="y=1")
        plt.plot(self.x[self.y[:] == 2, 0].numpy(), self.x[self.y[:] == 2, 1].numpy(), 'go', label="y=2")
        plt.legend()
        plt.show()


# create nn module using module list
class Net(nn.Module):

    # constructor
    # given a list of integers, representing the size of each layer
    # in the network, store the linear layers in the network
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        # don't apply relu activation in output layer
        for layer in self.hidden[:-1]:
            x = torch.relu_(layer(x))
        x = self.hidden[-1](x)
        return x


def train(data_set, model, criterion, train_loader, optimizer, epochs=100):
    # Lists to keep track of loss and accuracy
    LOSS = []
    ACC = []

    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
        LOSS.append(loss.item())
        ACC.append(accuracy(model, data_set))
        # Prints the Loss and Accuracy vs Epoch graph
    results = {"Loss": LOSS, "Accuracy": ACC}
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(LOSS, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()
    return results


def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()


# Create the dataset and plot it
data_set = Data()
# data_set.plot_data()
data_set.y = data_set.y.view(-1)

# Initialize a dictionary to contain the cost and accuracy
Results = {"momentum 0": {"Loss": 0, "Accuracy:": 0}, "momentum 0.1": {"Loss": 0, "Accuracy:": 0}}

# Train a model with 1 hidden layer and 50 neurons
# Size of input layer is 2, hidden layer is 50, and output layer is 3
# Our X values are x and y coordinates and this problem has 3 classes
Layers = [2, 50, 3]
# create model
model = Net(Layers)
learning_rate = 0.10
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.2)
train_loader = DataLoader(dataset=data_set, batch_size=20)
criterion = nn.CrossEntropyLoss()
Results['momentum 0'] = train(data_set, model, criterion, train_loader, optimizer, epochs=100)
plot_decision_regions_3class(model, data_set)

# Plot the Loss result for each term
# for key, value in Results.items():
#     plt.plot(value['Loss'], label=key)
#     plt.legend()
#     plt.xlabel('epoch')
#     plt.ylabel('Total Loss or Cost')
#     plt.show()

# Plot the Accuracy result for each term
# for key, value in Results.items():
#     plt.plot(value['Accuracy'],label=key)
#     plt.legend()
#     plt.xlabel('epoch')
#     plt.ylabel('Accuracy')
#     plt.show()
if __name__ == '__main__':
    print()
