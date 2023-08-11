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
    
    
    # OOD dataset
cifar100_ds = CIFAR100(root='./data', train=True, download=True, transform=transform)

# Take a subset of CIFAR-100 which cateogories belong to 'house objects'
ood_classes = ['bottle', 'clock', 'plate', 'telephone']
ood_classes_idx = {c:cifar100_ds.class_to_idx[c] for c in ood_classes}
ood_idx_classes = {v:k for k, v in ood_classes_idx.items()}
ood_indices = [i for i in range(len(cifar100_ds)) if cifar100_ds.targets[i] in ood_classes_idx.values()]

# Compute the subset
ds_fake = Subset(cifar100_ds, ood_indices)
dl_fake = torch.utils.data.DataLoader(ds_fake, batch_size = batch_size, shuffle = False, num_workers = 2)