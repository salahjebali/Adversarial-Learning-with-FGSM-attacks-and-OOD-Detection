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

# USE THIS CELL TO TRAIN MODEL FROM SCRATCH.
model = CNN().to(device)

# Train for only 50 epochs.
epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Main training loop.
for epoch in range(epochs):
    running_loss = 0.0
    # Iterate over all batches.
    for (i, (Xs, ys)) in enumerate(dl_train, 0):
        Xs = Xs.to(device)
        ys = ys.to(device)

        # Make a gradient step.
        optimizer.zero_grad()
        outputs = model(Xs)
        loss = criterion(outputs, ys)
        loss.backward()
        optimizer.step()

        # Track epoch loss.
        running_loss += loss.item()

    # Print average epoch loss.
    print(f'{epoch + 1} loss: {running_loss / len(dl_train):.3f}')

print('Finished Training')
torch.save(model.state_dict(), './cifar10_CNN.pth')


# Function to collect all logits from the model on entire dataset.
def collect_logits(model, dl):
    logits = []
    with torch.no_grad():
        for (Xs, _) in dl:
            logits.append(model(Xs.to(device)).cpu().numpy())
    return np.vstack(logits)

# Collect logits on CIFAR-10 test set (ID) and noise (very OOD).
logits_ID = collect_logits(model, dl_test)
logits_OOD = collect_logits(model, dl_fake)

# Plot the *distribution* of max logit outputs.
_ = plt.hist(logits_ID.max(1), 50, density=True, alpha=0.5, label='ID')
_ = plt.hist(logits_OOD.max(1), 50, density=True, alpha=0.5, label='OOD')
plt.legend()

# Take the logits from the take section and use softmax function to transform them in probabilities
temp = 0.5
id_probs = F.softmax(torch.tensor(logits_ID/temp), dim=1).numpy()
ood_probs = F.softmax(torch.tensor(logits_OOD/temp), dim=1).numpy()

# Since scikit-learn functions do not support multi class classification, we use ID pred as pos and OOD pred as neg
all_labels = np.hstack([np.ones_like(id_probs.max(1)), np.zeros_like(ood_probs.max(1))])
all_preds = np.hstack([id_probs.max(1), ood_probs.max(1)])

fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc_id = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_id)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for ID CIFAR-10')
plt.legend(loc="lower right")
plt.show()

precision, recall, _ = precision_recall_curve(all_labels, all_preds)

# Calculate Precision-Recall curve and PR-AUC score for both ID and OOD scoring
#precision_id, recall_id, _ = precision_recall_curve(all_labels == 1, all_predictions)
pr_auc_id = average_precision_score(all_labels, all_preds)

# Plot Precision-Recall curve
pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc_id, estimator_name='ID CIFAR-10')
pr_display.plot()
plt.title('Precision-Recall Curve for ID CIFAR-10')
plt.show()



