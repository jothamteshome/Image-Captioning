import torch

from PIL import Image
from torch import tensor
from torchvision import transforms

from utils.data import getWord2VecEmbeddings
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


def generateCaption(img_path: str) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word2vec, max_length = getWord2VecEmbeddings()

    start_idx, pad_idx, end_idx = word2vec.wv.key_to_index.get('<start>'), \
        word2vec.wv.key_to_index.get('<pad>'), \
        word2vec.wv.key_to_index.get('<end>')

    model = loadModel(len(word2vec.wv), pad_idx).to(device)

    generated_caption = [start_idx]
    hidden_state = None

    img = processImage(img_path).to(device)

    for _ in range(max_length):
        curr_token = torch.tensor([generated_caption[-1]]).unsqueeze(0).to(device)


        with torch.no_grad():
            decoder_output, hidden_state = model(img, curr_token, hidden_state)
        
        next_token = torch.argmax(torch.nn.functional.softmax(decoder_output, dim=-1), dim=-1).item()

        generated_caption.append(next_token)

        if next_token == end_idx:
            break
    
    # Decode generated tokens and join them into a single string element
    decoded_caption = [word2vec.wv.index_to_key[token] for token in generated_caption]
    decoded_caption = " ".join(decoded_caption)

    print(decoded_caption)
    
    return decoded_caption