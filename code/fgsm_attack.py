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

# Select best device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Implement the FGSM attack
def fgsm_attack(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    perturbed_data = data + epsilon * data.grad.data.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Clip perturbed data to [0, 1]
    return perturbed_data

def targeted_fgsm_attack(model, data, target_class, epsilon):
    model.eval()
    data.requires_grad = True

    output = model(data)
    target = torch.full((data.size(0),), target_class, dtype=torch.long, device=data.device)
    loss = F.cross_entropy(output, target)  # Loss for the target class
    model.zero_grad()
    loss.backward()

    # Compute the perturbation and add it to the data
    perturbation = epsilon * data.grad.data.sign()
    perturbed_data = data + perturbation
    perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Clip perturbed data to [0, 1]

    return perturbed_data