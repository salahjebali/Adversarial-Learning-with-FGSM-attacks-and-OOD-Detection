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
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, PrecisionRecallDisplay, precision_recall_curve,  roc_curve, auc, precision_recall_curve, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay

from id_data.py import *
from model.py import *
from ood_data.py import *
from fgsm_attack.py import *
from ood_detection.py import *

# Function to calculate accuracy on a given dataloader
def get_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Calculate accuracy on the test set before the attack
accuracy_before_attack = get_accuracy(model, dl_test)
print(f"Accuracy on the test set before the attack: {accuracy_before_attack:.4f}")

# Function to evaluate accuracy on adversarial examples for different epsilon values
def evaluate_adversarial_accuracy(model, dataloader, epsilon_values):
    accuracies = []
    for epsilon in epsilon_values:
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            perturbed_inputs = fgsm_attack(model, inputs, labels, epsilon)
            outputs = model(perturbed_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        accuracies.append(accuracy)
    return accuracies

# Choose a range of epsilon values to evaluate
epsilon_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]

# Evaluate accuracy on adversarial examples
adversarial_accuracies = evaluate_adversarial_accuracy(model, dl_test, epsilon_values)

# Visualize the results
plt.plot(epsilon_values, adversarial_accuracies, marker='o')
plt.xlabel('Epsilon (ε)')
plt.ylabel('Accuracy on Adversarial Examples')
plt.title('Accuracy vs. Epsilon for FGSM Attack')
plt.show()

def train(model, dataloader, optimizer, criterion, epsilon):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out gradients
        optimizer.zero_grad()

        # Clean data training
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Adversarial training
        perturbed_inputs = fgsm_attack(model, inputs, labels, epsilon)
        perturbed_outputs = model(perturbed_inputs)
        loss_adv = criterion(perturbed_outputs, labels)
        loss_adv.backward()

        # Update parameters
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item() + loss_adv.item()

    accuracy_train = 100 * correct / total
    loss_train = running_loss / len(dataloader)

    return accuracy_train, loss_train


# Initialize model, optimizer, and criterion
robust_model = CNN()
robust_model = robust_model.to(device)
optimizer = optim.Adam(robust_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 50
epsilon = 0.1  # Set the desired epsilon value for FGSM attack

# Training loop
for epoch in range(epochs):
    accuracy_train, loss_train = train(robust_model, dl_train, optimizer, criterion, epsilon)

    # Calculate test accuracy and loss
    accuracy_test = evaluate_adversarial_accuracy(robust_model, dl_test, epsilon)

    # Print the results for each epoch
    print(f"Epoch: {epoch + 1}, Epsilon: {epsilon:.2f}, Train Accuracy: {accuracy_train:.2f}%, Train Loss: {loss_train:.4f}, Test Accuracy: {accuracy_test:.2f}%")
    
robust_model = robust_model.to(device)

# Function to evaluate accuracy on adversarial examples for different epsilon values
def evaluate_adversarial_accuracy(model, dataloader, epsilon_values):
    accuracies = []
    for epsilon in epsilon_values:
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            perturbed_inputs = fgsm_attack(model, inputs, labels, epsilon)
            outputs = model(perturbed_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        accuracies.append(accuracy)
    return accuracies

# Choose a range of epsilon values to evaluate
epsilon_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]

# Evaluate accuracy on adversarial examples
standard_model_accuracies = evaluate_adversarial_accuracy(model, dl_test, epsilon_values)
robust_model_accuracies = evaluate_adversarial_accuracy(robust_model, dl_test, epsilon_values)

# Plot the accuracies vs. epsilon values
plt.plot(epsilon_values, standard_model_accuracies, label='Standard Model', marker='o')
plt.plot(epsilon_values, robust_model_accuracies, label='Robust Model', marker='o')

# Add labels, title, and legend
plt.xlabel('Epsilon (ε)')
plt.ylabel('Accuracy on Adversarial Examples')
plt.title('Accuracy vs. Epsilon for FGSM Attack')
plt.legend()

# Show the plot
plt.show()
