import torch

from torch import nn, tensor
from torch.utils.data import DataLoader
from torcheval.metrics import BLEUScore
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from utils.data import getGPT2Tokenizer
from utils.models import ImageCaptioningNetwork

def evaluateEpoch(model: ImageCaptioningNetwork, dataloader: DataLoader, device: torch.device, pad_idx: int) -> float:
    """
    
    Function to run evaluation of model after training for a single epoch

    Parameters:
        model (ImageCaptioningNetwork):     Trained model to evaluate
        dataloader (DataLoader):            DataLoader containing evaluation data to compute loss on
        device (torch.device):              Device to move data to while evaluating
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



def generateCaption(model: ImageCaptioningNetwork, image: tensor, tokenizer: GPT2TokenizerFast, device: torch.device) -> str:
    """
    
    Function to generate a caption for a single image

    
    Parameters:
        model (ImageCaptioningNetwork):     Model used to generate caption from image
        image (tensor):                     Image to generate caption for
        tokenizer (GPT2TokenizerFast):      Tokenizer used to tokenize and decode generated caption
        device (torch.device):              Device to generate captions on


    Returns:
        str:    Generated caption in string format

    """
    # Get BOS and EOS token id from tokenizer
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id

    # Initialize generated caption and initial hidden state of GRU layer
    generated_caption = [bos_id]
    curr_state = (None, None)

    # Loop until 50 tokens are generated
    for _ in range(50):

        # Convert token to tensor
        curr_token = torch.tensor([generated_caption[-1]]).unsqueeze(0).to(device)

        # Pass image, token, and previous hidden state to model
        with torch.no_grad():
            decoder_output, curr_state = model(image, curr_token, curr_state)

        # Determine next token from output
        next_token = torch.argmax(torch.nn.functional.softmax(decoder_output, dim=-1), dim=-1).item()

        # Append generated token to caption list
        generated_caption.append(next_token)

        # Break if EOS token found
        if next_token == eos_id:
            break

    # Remove BOS token from begining of sequence
    generated_caption = generated_caption[1:]

    # Remove EOS token from end of sequence
    if generated_caption[-1] == eos_id:
        generated_caption = generated_caption[:-1]


    return tokenizer.decode(generated_caption)



def BLEUEval(model: ImageCaptioningNetwork, dataloader: DataLoader, device: torch.device) -> float:
    """
    
    Calculate the BLEU score for the generated captions

    
    Parameters:
        model (ImageCaptioningNetwork):     Model to generate captions with
        dataloader (DataLoader):            DataLoader containing images and raw captions to compare generated captions to
        device (torch.device):              Device to load the data onto for caption generation

        
    Returns:
        float:  Float representing the BLEU score for the model

    """

    # Set model to evaluation mode
    model.eval()

    # Initialize tokenizer
    tokenizer = getGPT2Tokenizer()

    # Initialize BLEU score metric
    bleu_score = BLEUScore(n_gram=2)

    with torch.no_grad():
        # Evaluate all batches of dataloader
        for images, reference_captions in tqdm(dataloader, unit='batch', desc='Computing BLEU score'):
            # Move data to device
            images = images.to(device)

            # Generate caption
            generated_caption = generateCaption(model, images, tokenizer, device)

            # Update current BLEU score
            bleu_score.update([generated_caption], reference_captions)

    # Compute BLEU score
    score = bleu_score.compute().item()

    return score