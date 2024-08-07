import torch

from torch import device, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.models import ImageCaptioningNetwork

def evaluateEpoch(model: ImageCaptioningNetwork, dataloader: DataLoader, device: device, pad_idx: int) -> float:
    """
    
    Function to run evaluation of model after training for a single epoch

    Parameters:
        model (ImageCaptioningNetwork):     Trained model to evaluate
        dataloader (DataLoader):            DataLoader containing evaluation data to compute loss on
        device (device):                    Device to move data to while evaluating
        pad_idx (int):                      ID representing padding token to generate mask

    
    Returns:
        float:      A float value representing the evaluation loss at the current time step

    """
    # Initialize loss function
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Set model to evaluation mode and initialize loss
    model.eval()
    running_loss = 0


    with torch.no_grad():
        # Train on all batches of dataloader
        for images, captions in tqdm(dataloader, unit='batch', desc='Evaluating'):

            # Move data to device
            images, captions = images.to(device), captions.to(device)
            inputs, targets = captions[:, :-1], captions[:, 1:]

            # Create mask of all tokens that are not "<pad>" tokens
            mask = targets != pad_idx

            # Compute model_predictions
            predictions, _ = model(images, inputs)

            # Reshape inputs for computing loss
            predictions = predictions.view(predictions.size(0) * predictions.size(1), -1)
            targets = targets.contiguous().flatten()

            # Compute loss and add to running total
            loss = criterion(predictions, targets).view(mask.shape) * mask
            loss = torch.sum(loss) / torch.sum(mask)
            running_loss += loss.item()

    return running_loss / len(dataloader)