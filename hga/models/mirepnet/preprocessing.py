"""Official MIRepNet EA and inverse-distance channel interpolation."""
import random
import numpy as np
import torch
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import cdist
from .channel_list import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_missing_channels_diff(x, target_channels, actual_channels):
    B, C, T = x.shape
    num_target = len(target_channels)

    existing_pos = np.array([channel_positions[ch] for ch in actual_channels])

    target_pos = np.array([channel_positions[ch] for ch in target_channels])

    W = np.zeros((num_target, C))
    for i, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
        if target_ch in actual_channels:
            src_idx = actual_channels.index(target_ch)
            W[i, src_idx] = 1.0
        else:
            dist = cdist([pos], existing_pos)[0]
            weights = 1 / (dist + 1e-6)
            weights /= weights.sum()
            W[i] = weights

    padded = np.zeros((B, num_target, T))
    for b in range(B):
        padded[b] = W @ x[b]

    return padded


def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def train(model, train_loader, criterion, optimizer, device, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        _, outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100

    return epoch_loss, accuracy, current_lr
