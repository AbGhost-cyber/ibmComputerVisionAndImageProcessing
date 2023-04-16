import numpy as np
# train_labels = [1, 2, 3, 4, 5, 6, 7]
# train_img = np.array(["m.jpg", "you.jpg"])
# n_classes = len(np.unique(train_labels, return_counts=True)[0])
#
# for i, (x, y) in enumerate(zip(train_img, train_labels)):
#     print(i)
#     print(x)
#     print(y)
import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision.transforms import transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

x = []
img = Image.open("/Users/mac/Downloads/stop/24.jpg").convert('RGB')
composed = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
x.append(composed(img))
y = torch.randn(10, 1)
data = list(zip(x, y))

# Convert the tuple to a TensorDataset
dataset = TensorDataset(*map(torch.stack, zip(*data)))

# Print the first element of the dataset
print(dataset[0])
if __name__ == '__main__':
    print()
