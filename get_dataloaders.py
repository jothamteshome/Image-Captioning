import nltk
import numpy as np

from gensim.models import Word2Vec
from PIL import Image
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor

nltk.download('punkt')


class CocoCaptionDataset(Dataset):
    """
    
    Custom CocoCaptionDataset that tokenizes each caption and creates
    image-caption pairs for each caption in the dataset


    Attributes:
        coco (CocoCaptions):                        CocoCaptions dataset from torchvision
        word2vec (Word2Vec):                        The trained Word2Vec model
        caption_length (int):                       Length of the largest caption in dataset
        transform (Compose):                        List of transforms to apply to an image
        root (str):                                 Path to root directory images are stored in
        examples: list[tuple[int, tensor]]:         List containing image-caption pairs

    """
    def __init__(self, root: str, annFile: str, transform: Compose, 
                 word2vec: Word2Vec, caption_length: int) -> None:
        """
        
        Constructor for CocoCaptionDataset

        
        Parameters:
            root (str):                     Path to root directory images are stored in
            annFile (str):                  Path to annotation file describing images
            transform (Compose):            List of transforms to apply to an image
            word2vec (Word2Vec):            The trained Word2Vec model
            caption_length (int):           Length of the largest caption in dataset
        
        """
        super(CocoCaptionDataset, self).__init__()

        # Load CocoCaptions Dataset from torchvision
        self.coco = CocoCaptions(root=root, annFile=annFile, transform=transform)

        self.word2vec = word2vec
        self.caption_length = caption_length
        self.transform = transform
        self.root = root
        self.examples = []

        # Loop through ids in dataset
        for id in range(len(self.coco)):
            img_id = self.coco.ids[id]
            annotations = self.coco.coco.imgToAnns[img_id]

            for annotation in annotations:
                self.examples.append((img_id, self.processAnnotation(annotation)))
        
        

    def __len__(self) -> int:
        """
        
        Returns the length of the dataset

        
        Returns:
            int: Length of dataset
        
        """
        return len(self.examples)
    

    def __getitem__(self, id: int) -> tuple[tensor, tensor]:
        """
        
        Gets the item located at the current id

        
        Parameters:
            id (int):   Index of an element in the dataset


        Returns:
            tensor: Tensor containing processed image
            tensor: Tensor containing ids of tokens in caption
        
        """
        img_id, caption = self.examples[id]

        # Load the image at the current id
        path = self.coco.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(f"{self.root}\\{path}").convert('RGB')


        # Apply ResNet50 transform to image
        image = self.transform(image)

        return image, caption
    

    def processAnnotation(self, annotation: dict) -> tensor:
        """
        
        Processes a single annotation by tokenizing the caption and
        replacing each token with its id representation from trained Word2Vec model


        Parameters:
            annotation (_Annotation): An annotation object containing information about image


        Returns:
            tensor: Tensor containing embedded sentence
        
        """
        # Convert caption to lower case then tokenize
        tokenized_caption = nltk.word_tokenize(annotation['caption'].lower())

        # Word2Vec key to index dictionary
        key2idx = self.word2vec.wv.key_to_index

        # IDs of special tokens
        start_idx, end_idx = key2idx.get("<start>"), key2idx.get("<end>")
        pad_idx, unk_idx = key2idx.get("<pad>"), key2idx.get("<unk>")

        # Begin each caption with start_idx
        embedded_sentence = [start_idx]

        # Loop through tokens in caption and assign their index
        for token in tokenized_caption:
            # Add index of word if it exists, otherwise add "<UNK>" value
            embedded_sentence.append(key2idx.get(token, unk_idx))

        # Add end_idx to mark end of caption
        embedded_sentence.append(end_idx)

        # Pad captions shorter than longest caption
        if len(embedded_sentence) < self.caption_length:
            embedded_sentence.extend([pad_idx] * (self.caption_length - len(embedded_sentence)))

        return tensor(embedded_sentence)


def getTrainLoaders(word2vec: Word2Vec, caption_length: int) -> tuple[DataLoader, DataLoader]:
    """
    
    Loads the training and validation datasets

    
    Parameters:
        word2vec (Word2Vec):    The trained Word2Vec model
        caption_length (int):   Length of the longest caption in the dataset


    Returns:
        DataLoader: DataLoader for training dataset
        DataLoader: DataLoader for validation dataset
    
    """

    # Transformation that ResNet50 model expects
    resnet_transform = Compose([
                        Resize(256),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    
    # Training CocoCaptions dataset
    train_dataset = CocoCaptionDataset(root="coco/images",
                          annFile="coco/annotations/captions_train2014.json",
                          transform=resnet_transform,
                          word2vec=word2vec,
                          caption_length=caption_length)
    
    # Valdiation CocoCaptions dataset
    val_dataset = CocoCaptionDataset(root="coco/images",
                          annFile="coco/annotations/captions_val2014.json",
                          transform=resnet_transform,
                          word2vec=word2vec,
                          caption_length=caption_length)
    

    # DataLoaders for both training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    return train_loader, val_loader


def _getTokenizedCaptions(dataset: CocoCaptions) -> tuple[list[str], int]:
    """
    
    Loops through dataset and returns list of tokenized captions

    
    Parameters:
        dataset (CocoCaptions): CocoCaptions dataset to extract captions from

    
    Returns:
        list[str]:  List of tokenized captions from dataset
        int:        Length of the longest caption in the dataset

    """

    tokenized_captions = []
    longest_caption_length = 0

    # Loop through dataset
    for id in range(len(dataset)):
        img_id = dataset.ids[id]
        annotations = dataset.coco.imgToAnns[img_id]

        # Loop though all annotations associated with current image id
        for annotation in annotations:
            # Convert caption to lower case then tokenize
            tokenized_caption = nltk.word_tokenize(annotation['caption'].lower())

            # Determine longest caption length available in dataset
            if len(tokenized_caption) > longest_caption_length:
                longest_caption_length = len(tokenized_caption)

            # Append tokenized caption
            tokenized_captions.append(tokenized_caption)

    return tokenized_captions, longest_caption_length + 2


def _trainWord2Vec(tokenized_captions: list[str]) -> tuple[Word2Vec, np.ndarray]:
    """

    Trains the Word2Vec model using tokenized captions
    

    Parameters:
        tokenized_captions (list[str]): A list of tokenized captions

    
    Returns:
        Word2Vec:   The trained Word2Vec model
        ndarray:    A matrix containing all Word2Vec embedding layers

    """

    # Train Word2Vec model using tokenized captions 
    word2vec = Word2Vec(tokenized_captions, min_count=2, seed=47)

    # Initialize an embedding matrix
    embedding_matrix = np.zeros((len(word2vec.wv), word2vec.vector_size))

    # Add word embedding vectors to embedding matrix
    for word, idx in word2vec.wv.key_to_index.items():
        embedding_vector = word2vec.wv[word]
        embedding_matrix[idx] = embedding_vector

    return word2vec, embedding_matrix


def getWord2VecEmbeddings() -> tuple[Word2Vec, np.ndarray, int]:
    """

    Trains a Word2Vec model for word embeddings using the captions
    found in the training/validation data

    
    Returns:
        Word2Vec:   The trained Word2Vec model
        ndarray:    A matrix containing all Word2Vec embedding layers
        int:        The length of the longest caption found in the dataset 

    """

    # Load training dataset from torchvision
    train_data = CocoCaptions(root="coco/images", annFile="coco/annotations/captions_train2014.json")
    
    # Load validation dataset from torchvision
    val_data = CocoCaptions(root="coco/images", annFile="coco/annotations/captions_val2014.json")
    
    # Get tokenized_captions from train_dataset and val_dataset
    train_tokens, train_longest_caption  = _getTokenizedCaptions(train_data)
    val_tokens, val_longest_caption = _getTokenizedCaptions(val_data)

    # Add tokenized captions and special tokens
    tokenized_captions = train_tokens + val_tokens + \
        [["<pad>"] * 5] + [["<unk>"] * 5] + [["<start>"] * 5] + [["<end>"] * 5]
    
    # Determine max caption length in dataset
    caption_length = max(train_longest_caption, val_longest_caption)

    # Train Word2Vec model using tokenized captions 
    word2vec, embedding_matrix = _trainWord2Vec(tokenized_captions)
    
    # Add 2 to caption_length to account for start and end tokens
    return word2vec, embedding_matrix, caption_length + 2
