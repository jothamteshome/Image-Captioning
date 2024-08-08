import nltk

from gensim.models import Word2Vec
from PIL import Image
from torch import tensor, stack
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor
from transformers import GPT2TokenizerFast

nltk.download('punkt')


class CocoCaptionDataset(Dataset):
    """
    
    Custom CocoCaptionDataset that tokenizes each caption and creates
    image-caption pairs for each caption in the dataset


    Attributes:
        coco (CocoCaptions):                        CocoCaptions dataset from torchvision
        examples (list[tuple[int, tensor]]):        List containing image-caption pairs
        root (str):                                 Path to root directory images are stored in
        tokenizer (GPT2TokenizerFast):              Pretrained GPT2 tokenize with added special tokens
        transform (Compose):                        List of transforms to apply to an image
        
    """
    def __init__(self, root: str, annFile: str, transform: Compose) -> None:
        """
        
        Constructor for CocoCaptionDataset

        
        Parameters:
            root (str):                     Path to root directory images are stored in
            annFile (str):                  Path to annotation file describing images
            transform (Compose):            List of transforms to apply to an image
        
        """
        super(CocoCaptionDataset, self).__init__()

        # Load CocoCaptions Dataset from torchvision
        self.coco = CocoCaptions(root=root, annFile=annFile, transform=transform)
        self.examples = []
        self.root = root
        self.tokenizer = getGPT2Tokenizer()
        self.transform = transform


        # Loop through dataset to get image id and caption lists
        image_ids, captions = self.getImageAndCaptionLists()

        # Tokenize the data using GPT2TokenizerFast
        tokenized_data = self.tokenizer(captions, add_special_tokens=False, padding=False, return_length=True, return_attention_mask=False)

        # Add special tokens and padding to tokenized caption data
        self.formatExamples(image_ids, tokenized_data)
        

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

        return image, encoded_caption
    

    def formatExamples(self, image_ids: list, tokenized_data: dict) -> None:
        """
        
        Takes input ids and tokenized data, adds special tokens and padding to
        tokenized data, and adds image-caption pairs to self.examples

        Parameters:
            image_ids (list):       List of image ids
            tokenized_data (dict):  Dictionary containing information about tokenized captions
        
        """
        # Get tokenized captions and caption lenghths
        input_ids = tokenized_data['input_ids']
        lengths = tokenized_data['length']
        max_length = max(lengths)


        # Loop through each id and add bos token, eos token and padding token
        for i in range(len(image_ids)):
            # Format each caption with special tokens
            formatted_text = [self.tokenizer.bos_token_id] + input_ids[i] + \
                  [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id] * (max_length - lengths[i])
            
            # Add image-caption pair to examples list
            self.examples.append((image_ids[i], tensor(formatted_text)))
    

    def getImageAndCaptionLists(self) -> tuple[list, list]:
        """
        
        Loops through the dataset and returns a list of image ids and
        a list of an equal amount of captions

        
        Returns:
            list:   List of image ids
            list:   List of captions
        
        """

        # Initialize lists to store images and captions
        image_ids, captions = [], []

        # Loop through ids in dataset
        for id in range(len(self.coco)):
            img_id = self.coco.ids[id]
            annotations = self.coco.coco.imgToAnns[img_id]

            for annotation in annotations:
                image_ids.append(img_id)
                captions.append(annotation['caption'])

        return image_ids, captions
        


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


def getTrainLoaders() -> tuple[DataLoader, DataLoader]:
    """
    
    Loads the training and validation datasets

    
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
                          transform=resnet_transform)
    
    # Valdiation CocoCaptions dataset
    val_dataset = CocoCaptionDataset(root="coco/images",
                                        annFile="coco/annotations/captions_val2014.json",
                                        transform=resnet_transform)
    

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



def getGPT2Tokenizer():
    """
    
    Initializes GPT2 tokenizer and adds special tokens before returning

    Returns:
        GPT2TokenizerFast:      The pretrained tokenizer with added special tokens
    
    """
    # Initialize base GPT2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', clean_up_tokenization_spaces=True, padding_size='left')


    # Special tokens to add to tokenizer
    special_tokens_dict = {
        'eos_token': '<|eos|>',
        'pad_token': '<|pad|>',
        'bos_token': '<|bos|>',
        'unk_token': '<|unk|>'
    }

    # Add special tokens
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer