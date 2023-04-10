# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to transform tensors
import torchvision.transforms as transforms
# Allows us to download datasets
import torchvision.datasets as dsets
# Allows us to use activation functions
import torch.nn.functional as F
# Used to graph data and loss curves
import matplotlib.pylab as plt
# Allows us to use arrays to manipulate and store data
import numpy as np
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

# Setting the seed will allow us to control randomness and give us reproducibility
torch.manual_seed(2)


# Create the model class using Sigmoid as the activation function
class Net(nn.Module):

    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        # D_in is the input size of the first layer (size of input layer)
        # H1 is the output size of the first layer and input size of the second layer (size of first hidden layer)
        # H2 is the outpiut size of the second layer and the input size of the third layer (size of second hidden layer)
        # D_out is the output size of the third layer (size of output layer)
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    def forward(self, x):
        # Puts x through the first layers then the sigmoid function
        x = torch.sigmoid(self.linear1(x))
        # Puts results of previous line through second layer then sigmoid function
        x = torch.sigmoid(self.linear2(x))
        # Puts result of previous line through third layer
        x = self.linear3(x)
        return x


# Create the model class using Relu as the activation function

class NetRelu(nn.Module):

    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    def forward(self, x):
        x = torch.relu_(self.linear1(x))
        x = torch.relu_(self.linear2(x))
        x = self.linear3(x)
        return x


# Model training function

# `model` (the neural network model to be trained),
# `criterion` (the loss function to be used),
# `train_loader` (the data loader for the training dataset),
# `validation_loader` (the data loader for the validation dataset),
# `optimizer` (the optimization algorithm to be used, such as stochastic gradient descent)
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}

    # numbers of times we train on the entire training dataset
    for epoch in range(epochs):
        # sets up an inner loop that will iterate over the training data,
        # where `i` is the index of the current batch and `(x, y)` are the inputs and labels for the current batch.
        for i, (x, y) in enumerate(train_loader):
            # sets the gradients of all model parameters to zero,
            # which is necessary before computing gradients for a new batch
            optimizer.zero_grad()
            # calculates prediction for the current batch x(reshaped to a 2d tensor with shape (batch_size, 28*28)
            z = model(x.view(-1, 28 * 28))
            # calculates the loss
            loss = criterion(z, y)
            # computes the gradients of the model parameters with respect to the loss
            loss.backward()
            # updates the model parameters using the gradients computed by `loss.backward()
            optimizer.step()
            # appends the current batch's training loss
            useful_stuff['training_loss'].append(loss.data.item())

        correct = 0
        for x, y in validation_loader:
            # Make a prediction
            z = model(x.view(-1, 28 * 28))
            # Get the class that has the maximum value
            _, label = torch.max(z, 1)
            # Check if our prediction matches the actual class
            correct += (label == y).sum().item()

            # Saves the percent accuracy
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)

    return useful_stuff


train_dataset = dsets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
# Create the validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
# Create the criterion function
criterion = nn.CrossEntropyLoss()
# Create the training data loader and validation data loader object
train_loader = DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Set the parameters to create the model
input_dimensions = 28 * 28
hidden_dimension1 = 50
hidden_dimension2 = 50
output_dimension = 10  # Number of classes
# Set the number of iterations
cust_epochs = 10

# Train the model with sigmoid function
learning_rate = 0.01
model = Net(input_dimensions, hidden_dimension1, hidden_dimension2, output_dimension)
optimizer = SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)

# Train the model with relu function
learning_rate = 0.01
# Create an instance of the NetRelu model
modelRelu = NetRelu(input_dimensions, hidden_dimension1, hidden_dimension2, output_dimension)
# Create an optimizer that updates model parameters using the learning rate and gradient
optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
# Train the model
training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)

# Compare the training loss
plt.plot(training_results['training_loss'], label='sigmoid')
plt.plot(training_results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()
plt.show()

if __name__ == '__main__':
    print(training_results_relu)
    print(training_results)