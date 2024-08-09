import os
import json
import torch

from torch import device, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import getGPT2Tokenizer
from utils.evaluate import evaluateEpoch
from utils.models import ImageCaptioningNetwork


class TrainingArgs:
    """
    
    Class for defining training arguments to train model with

    Attributes:
        checkpoint_dir (str):   Name of the directory to store model checkpoints
        device (device):        The device to train the model on
        learning_rate (float):  Value to use as the learning rate for model training
        model_title (str):      Brief name for the model
        num_epochs (int):       Number of epochs to train the model for
        weight_decay (int):     Value to use as the weight decay for model training
        save_dir (str):         Name of the directory to save the model in
    
    """
    def __init__(self, override:bool=False) -> None:
        self.checkpoint_dir = "checkpoints"
        self.device = device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = 1e-3
        self.model_title = "image_captioning_network"
        self.num_epochs = 5
        self.weight_decay = 0
        self.save_dir = f"{self.model_title}__lr={self.learning_rate:.2E},epochs={self.num_epochs},decay={self.weight_decay}"
        self.metrics_json = f"saved_models/{self.save_dir}/eval_metrics.json"

        # Raise error if model being trained already exists and override is set to False
        try:
            os.makedirs(f"saved_models/{self.save_dir}/{self.checkpoint_dir}")
        except OSError:
            if not override:
                raise OSError("This model already exists. Call with `override=True` set to override existing results")

        with open(self.metrics_json, 'w') as f:
            json.dump({'checkpoints': {}, 'final_model': {}}, f)


    def saveMetrics(self, metrics: dict, epoch:int|None=None) -> None:
        """
        
        Function to save evaluation metrics at current time


        Parameters:
            metrics (dict):     Dictionary containing evaluation metrics
            epoch (int|None):   Integer representing current epoch for checkpoint if not None

        """

        # Read metrics json file
        with open(self.metrics_json, 'r') as f:
            current_metrics = json.load(f)

        # If saving a checkpoint
        if epoch is not None:
            # Initialize dictionary for checkpoint
            current_metrics['checkpoints'][f'epoch-{epoch}'] = {}

            # Create metric if it doesn't exist or update existing value
            for metric in metrics:
                current_metrics['checkpoints'][f'epoch-{epoch}'][metric] = metrics[metric]
        else:
            # Create metric if it doesn't exist or update existing value
            for metric in metrics:
                current_metrics['final_model'][metric] = metrics[metric]


        # Save json back to file
        with open(self.metrics_json, 'w') as f:
            json.dump(current_metrics, f)



def trainEpoch(model: ImageCaptioningNetwork, optimizer: AdamW, train_loader: DataLoader, device: device, pad_idx: int) -> float:
    """
    
    Handles training a model for a single epoch

    
    Parameters:
        model (ImageCaptioningNetwork):     Model to train on dataset for one epoch
        optimizer (AdamW):                  Optimizer to adjust parameters during training
        train_loader (DataLoader):          DataLoader contining training data to train model on
        device (device):                    Device to move data to while training
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


def trainModel(model: ImageCaptioningNetwork, train_loader: DataLoader, val_loader: DataLoader, train_args: TrainingArgs) -> None:
    """
    
    Runs full training loop of model for given amount of epochs


    Parameters:
        model (ImageCaptioningNetwork):     Model to train on dataset for one epoch
        train_loader (DataLoader):          DataLoader contining training data to train model on
        val_loader (DataLoader):            DataLoader contining validation data to evaluate model on
        train_args (TrainingArgs):          Arguments to set during the training of the model
    
    """

    # Initialize optimizer for updating model parameters
    optimizer = AdamW(model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    # Get pad token id from tokenizer
    pad_idx = getGPT2Tokenizer().pad_token_id

    # Run for set number of epochs
    for epoch in range(train_args.num_epochs):

        # Train model for current epoch and get training loss
        train_loss = trainEpoch(model, optimizer, train_loader, train_args.device, pad_idx)

        # Save checkpoint after epoch
        torch.save(model.state_dict(), f"saved_models/{train_args.save_dir}/{train_args.checkpoint_dir}/epoch-{epoch+1}.pt")

        # Evaluate model for current epoch and get validation loss
        val_loss = evaluateEpoch(model, val_loader, train_args.device, pad_idx)

        # Print current epoch and loss metrics for training and testing
        print(f"Epoch {epoch+1} | train_loss: {train_loss} | val_loss: {val_loss}")
        train_args.saveMetrics({'train_loss': train_loss, 'val_loss': val_loss}, epoch=epoch+1)

    
    # Save fully trained model
    torch.save(model.state_dict(), f"saved_models/{train_args.save_dir}/fully_trained_model.pt")
    train_args.saveMetrics({'train_loss': train_loss, 'val_loss': val_loss})

    # Delete checkpoint equivalent to final model
    os.remove(f"saved_models/{train_args.save_dir}/{train_args.checkpoint_dir}/epoch-{train_args.num_epochs}.pt")