import torch
from typing import List

def cross_entropy_loss_embedded(list_pred:List, list_y:List) -> float:
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0

    for y_pred, y_true in zip(list_pred, list_y):
        target = y_true.argmax(dim=1)  # Targets are indices for each item in the batch
        loss = loss_fn(y_pred, target)
        total_loss += loss

    return total_loss / len(list_pred)  # Average over the number of attributes

def accuracy_embedded(list_pred:List, list_y:List) -> float:
    total_correct = 0
    total_samples = 0

    for y_pred, y_true in zip(list_pred, list_y):
        preds = y_pred.argmax(dim=1)
        targets = y_true.argmax(dim=1)
        total_samples += preds.size(0)  # Increment sample count based on batch size
        total_correct += (preds == targets).sum().item()

    return total_correct / total_samples