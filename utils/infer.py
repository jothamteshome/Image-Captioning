import torch

from PIL import Image
from torch import tensor
from torchvision import transforms

from utils.data import getGPT2Tokenizer
from utils.evaluate import generateCaption
from utils.models import loadModel

def processImage(image_path: str) -> tensor:
    """
    
    Handles processing of an image to work for the ImageEncoder model


    Parameters:
        image_path (str):   Path to the image being loaded


    Returns:
        tensor:     Tensor containing the processed image
    
    """

    # Transformations required by ResNet50 model
    resnetTransform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    
    # Open image
    image = Image.open(image_path)
    
    # Process image using transform and unsqueeze for batch dimension
    processed_image = resnetTransform(image).unsqueeze(0)

    return processed_image


def captionImage(img_path: str, trained_model_path: str) -> str:
    """
    
    Captions an image given an filepath to an image and a trained model


    Parameters:
        img_path (str):             Path to an image to generate caption for
        trained_model_path (str):   Path to trained model used to generate caption


    Returns:
        str:    Caption for the image that was passed in
    
    """
    
    # Initialize the device, tokenizer, and model to use for captioning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = getGPT2Tokenizer()
    model = loadModel(trained_model_path).to(device)

    # Process the image and load to device memory
    img = processImage(img_path).to(device)

    # Generate and print a caption
    caption = generateCaption(model, img, tokenizer, device)
    print(caption)

    return caption