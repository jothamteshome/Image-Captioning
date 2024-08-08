import nltk

from gensim.models import Word2Vec
from PIL import Image
from torch import tensor, stack
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
        max_length (int):                           Length of the largest caption in dataset
        transform (Compose):                        List of transforms to apply to an image
        root (str):                                 Path to root directory images are stored in
        examples (list[tuple[int, tensor]]):        List containing image-caption pairs

    """
    def __init__(self, root: str, annFile: str, transform: Compose, 
                 word2vec: Word2Vec, max_length: int) -> None:
        """
        
        Constructor for CocoCaptionDataset

        
        Parameters:
            root (str):                     Path to root directory images are stored in
            annFile (str):                  Path to annotation file describing images
            transform (Compose):            List of transforms to apply to an image
            word2vec (Word2Vec):            The trained Word2Vec model
            max_length (int):               Length of the largest caption in dataset
        
        """
        super(CocoCaptionDataset, self).__init__()

        # Load CocoCaptions Dataset from torchvision
        self.coco = CocoCaptions(root=root, annFile=annFile, transform=transform)

        self.word2vec = word2vec
        self.max_length = max_length
        self.transform = transform
        self.root = root
        self.examples = []

        # Loop through ids in dataset
        for id in range(len(self.coco)):
            img_id = self.coco.ids[id]
            annotations = self.coco.coco.imgToAnns[img_id]

            for annotation in annotations:
                self.examples.append((img_id, self.encodeAnnotation(annotation)))
        
        

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
        img_id, encoded_caption = self.examples[id]

        # Load the image at the current id
        path = self.coco.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(f"{self.root}\\{path}").convert('RGB')


        # Apply ResNet50 transform to image
        image = self.transform(image)

        return image, encoded_caption
    

    def encodeAnnotation(self, annotation: dict) -> tensor:
        """
        
        Processes a single annotation by tokenizing the caption and
        replacing each token with its id representation from trained Word2Vec model


        Parameters:
            annotation (dict): An annotation object containing information about image


        Returns:
            tensor: Tensor containing embedded sentence
        
        """
        # Convert caption to lower case then tokenize
        tokenized_caption = ["<start>"] + nltk.word_tokenize(annotation['caption'].lower()) + ["<end>"]


        # Pad captions shorter than longest caption
        if len(tokenized_caption) < self.max_length:
            tokenized_caption.extend(["<pad>"] * (self.max_length - len(tokenized_caption)))


        # Word2Vec key to index dictionary
        key2idx = self.word2vec.wv.key_to_index

        # Id of special "<unk>" token for out-of-vocabulary tokens
        unk_idx = key2idx.get("<unk>")

        # Begin each caption with start_idx
        encoded_caption = []

        # Loop through tokens in caption and assign their index
        for token in tokenized_caption:
            # Add index of word if it exists, otherwise add "<unk>" value
            encoded_caption.append(key2idx.get(token, unk_idx))

        return tensor(encoded_caption)
    


class FinalEvalCocoCaptionDataset(CocoCaptionDataset):
    """
    
    Custom FinalEvalCocoCaptionDataset that tokenizes each caption and creates
    image-caption_dict pairs for each image in the dataset


    Attributes:
        coco (CocoCaptions):                        CocoCaptions dataset from torchvision
        word2vec (Word2Vec):                        The trained Word2Vec model
        max_length (int):                           Length of the largest caption in dataset
        transform (Compose):                        List of transforms to apply to an image
        root (str):                                 Path to root directory images are stored in
        examples (list[dict[str, int|list]]):       List containing image-caption pairs

    """

    def __init__(self, root: str, annFile: str, transform: Compose, 
                 word2vec: Word2Vec, max_length: int) -> None:
        """
        
        Constructor for FinalEvalCocoCaptionDataset

        
        Parameters:
            root (str):                     Path to root directory images are stored in
            annFile (str):                  Path to annotation file describing images
            transform (Compose):            List of transforms to apply to an image
            word2vec (Word2Vec):            The trained Word2Vec model
            max_length (int):               Length of the largest caption in dataset
        
        """
        super(CocoCaptionDataset, self).__init__()

        # Load CocoCaptions Dataset from torchvision
        self.coco = CocoCaptions(root=root, annFile=annFile, transform=transform)

        self.word2vec = word2vec
        self.max_length = max_length
        self.transform = transform
        self.root = root
        self.examples = []

        # Loop through ids in dataset
        for id in range(len(self.coco)):
            img_id = self.coco.ids[id]
            annotations = self.coco.coco.imgToAnns[img_id]

            raw_captions = []

            # Limit to a max of 5 annotations
            for annotation in annotations[:5]:
                raw_captions.append(annotation['caption'].lower())

            self.examples.append({'img_id': img_id, 'raw_captions': tuple(raw_captions)})


    def __getitem__(self, id: int) -> tuple[tensor, tuple[str]]:
        """
        
        Gets the item located at the current id

        
        Parameters:
            id (int):   Index of an element in the dataset


        Returns:
            tensor:     Tensor containing processed image
            tuple[str]: Tuple containing raw caption strings
        
        """
        img_id, raw_captions = self.examples[id].values()

        # Load the image at the current id
        path = self.coco.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(f"{self.root}\\{path}").convert('RGB')


        # Apply ResNet50 transform to image
        image = self.transform(image)

        return image, raw_captions


def getTrainLoaders(word2vec: Word2Vec, max_length: int) -> tuple[DataLoader, DataLoader]:
    """
    
    Loads the training and validation datasets

    
    Parameters:
        word2vec (Word2Vec):    The trained Word2Vec model
        max_length (int):       Length of the longest caption in the dataset


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
                          max_length=max_length)
    
    # Valdiation CocoCaptions dataset
    val_dataset = CocoCaptionDataset(root="coco/images",
                                        annFile="coco/annotations/captions_val2014.json",
                                        transform=resnet_transform,
                                        word2vec=word2vec,
                                        max_length=max_length)
    

    # DataLoaders for both training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    return train_loader, val_loader


def getFinalEvalLoader(word2vec: Word2Vec, max_length: int) -> DataLoader:
    """
    
    Loads the final evaluation dataset

    
    Parameters:
        word2vec (Word2Vec):    The trained Word2Vec model
        max_length (int):       Length of the longest caption in the dataset


    Returns:
        DataLoader: DataLoader for final evaluation dataset
    
    """
    def final_eval_collate_fn(batch):
        """
        
        Handles the collation for a batch of data in the validation DataLoader

        Parameters:
            batch: A batch of data to be collated
        
            
        Returns:
            tensor:             A tensor containing a batch of images
            tuple[tuple[str]]:  A tuple of tuples of strings containing raw captions

        """
        images, raw_captions = zip(*batch)

        images = stack(images)

        return images, raw_captions
    
    # Transformation that ResNet50 model expects
    resnet_transform = Compose([
                        Resize(256),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    

    # FinalEvalCocoCaptions dataset
    dataset = CocoCaptionDataset(root="coco/images",
                                        annFile="coco/annotations/captions_val2014.json",
                                        transform=resnet_transform,
                                        word2vec=word2vec,
                                        max_length=max_length)
    
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=final_eval_collate_fn, shuffle=True)

    return dataloader


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
    max_length = 0

    # Loop through dataset
    for id in range(len(dataset)):
        img_id = dataset.ids[id]
        annotations = dataset.coco.imgToAnns[img_id]

        # Loop though all annotations associated with current image id
        for annotation in annotations:
            # Convert caption to lower case then tokenize
            tokenized_caption = nltk.word_tokenize(annotation['caption'].lower())

            # Determine longest caption length available in dataset
            if len(tokenized_caption) > max_length:
                max_length = len(tokenized_caption)

            # Append tokenized caption
            tokenized_captions.append(tokenized_caption)

    return tokenized_captions, max_length


def getWord2VecEmbeddings() -> tuple[Word2Vec, int]:
    """

    Trains a Word2Vec model for word embeddings using the captions
    found in the training/validation data

    
    Returns:
        Word2Vec:   The trained Word2Vec model
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
    max_length = max(train_longest_caption, val_longest_caption)

    # Train Word2Vec model using tokenized captions 
    word2vec = Word2Vec(tokenized_captions, min_count=2, seed=47)
    
    # Add 2 to max_length to account for start and end tokens
    return word2vec, max_length + 2
