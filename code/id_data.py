# Standard imports.
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import FakeData, CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt

# Function to display images
def show_images(dataloader, classes, num_images=4):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Denormalize the images (if they were previously normalized)
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    images = images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)

    # Convert images to numpy arrays for plotting
    np_images = images.numpy()
    np_images = np.transpose(np_images, (0, 2, 3, 1))

    # Plot the images
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    for i in range(num_images):
        axes[i].imshow(np_images[i])
        axes[i].set_title(f'Label: {classes[labels[i]]}')
        axes[i].axis('off')
    plt.show()


# We will use CIFAR-10 as our in-distribution dataset.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load train set and test set
batch_size = 32
ds_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Train-Val split
val_size = int(0.1 * len(ds_train))
I = np.random.permutation(len(ds_train))
ds_train = Subset(ds_train, I[val_size:])
ds_val = Subset(ds_train, I[:val_size])

# Setup DataLoaders
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

# In case we want to pretty-print classifications.
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

