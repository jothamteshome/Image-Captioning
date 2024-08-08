import torch

from torch import device, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import getGPT2Tokenizer
from utils.evaluate import evaluateEpoch
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
    criterion = nn.CrossEntropyLoss(reduction='none')

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


def trainModel(model: ImageCaptioningNetwork, train_loader: DataLoader, val_loader: DataLoader, device: device, num_epochs:int=5) -> None:
    """
    
    Runs full training loop of model for given amount of epochs


    Parameters:
        model (ImageCaptioningNetwork):     Model to train on dataset for one epoch
        train_loader (DataLoader):          DataLoader contining training data to train model on
        val_loader (DataLoader):            DataLoader contining validation data to evaluate model on
        device (device):                    Device to copy data to while training
    
    """

    # Initialize optimizer for updating model parameters
    optimizer = AdamW(model.parameters(), lr=1e-4)

    pad_idx = getGPT2Tokenizer().pad_token_id

    # Run for set number of epochs
    for epoch in range(num_epochs):

        # Train model for current epoch and get training loss
        train_loss = trainEpoch(model, optimizer, train_loader, device, pad_idx)

        # Save checkpoint after epoch model
        torch.save(model.state_dict(), f"google_model_checkpoints/checkpoint_epoch_{epoch+1}.pt")

        # Evaluate model for current epoch and get validation loss
        val_loss = evaluateEpoch(model, val_loader, device, pad_idx)

        # Print current epoch and loss metrics for training and testing
        print(f"Epoch {epoch+1} | train_loss: {train_loss} | val_loss: {val_loss}")