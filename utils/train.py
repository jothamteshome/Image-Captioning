import torch

from torch import device, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.models import ImageCaptioningNetwork


def trainEpoch(model: ImageCaptioningNetwork, optimizer: AdamW, train_loader: DataLoader, device: device, pad_idx: int) -> float:
    """
    
    Handles training a model for a single epoch

    Parameters:
        model (ImageCaptioningNetwork):     Model to train on dataset for one epoch
        optimizer (AdamW):                  Optimizer to adjust parameters during training
        train_loader (DataLoader):          DataLoader contining training data to train model on
        device (device):                    Device to copy data to while training
        pad_idx (int):                      ID representing padding token

    Returns:
        float:      A float value representing the evaluation loss at the current training step

    """
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Set model to training mode and initialize loss
    model.train()
    running_loss = 0


    # Train on all batches of training dataloader
    for images, captions in tqdm(train_loader, unit='batch', desc='Training'):
        optimizer.zero_grad()

        # Move data to device
        images, captions = images.to(device), captions.to(device)
        inputs, targets = captions[:, :-1], captions[:, 1:]

        # Compute model_predictions
        predictions, _ = model(images, inputs)

        # Create mask of all tokens that are not "<pad>" tokens
        mask = targets != pad_idx

        # Reshape inputs for computing loss
        predictions = predictions.view(predictions.size(0) * predictions.size(1), -1)
        targets = targets.contiguous().flatten()

        # Compute loss and add to running total
        loss = criterion(predictions, targets).view(mask.shape) * mask
        loss = torch.sum(loss) / torch.sum(mask)
        running_loss += loss.item()

        # Handle backpropagation
        loss.backward()
        optimizer.step()

    return running_loss / len(train_loader)