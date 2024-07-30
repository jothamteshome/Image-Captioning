import nltk
import torch


from gensim.models import Word2Vec
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms

nltk.download('punkt')


class CocoCaptionDataset(Dataset):
    def __init__(self, root, annFile, transform, word2vec, caption_length):
        super(CocoCaptionDataset, self).__init__()
        self.coco = datasets.CocoCaptions(root=root, annFile=annFile, transform=transform)

        
        self.examples = []

        for id in range(len(self.coco)):
            img_id = self.coco.ids[id]
            annotations = self.coco.coco.imgToAnns[img_id]

            for annotation in annotations:
                tokenized_caption = nltk.word_tokenize(annotation['caption'])


                # IDs of special tokens
                start_idx, end_idx = word2vec.wv.key_to_index.get("<START>"), word2vec.wv.key_to_index.get("<END>")
                pad_idx, unk_idx = word2vec.wv.key_to_index.get("<PAD>"), word2vec.wv.key_to_index.get("<UNK>")

                # Begin each caption with start_idx
                embedded_sentence = [start_idx]

                for token in tokenized_caption:
                    # Add index of word if it exists, otherwise add "<UNK>" value
                    embedded_sentence.append(word2vec.wv.key_to_index.get(token, unk_idx))

                # Add end_idx to mark end of caption
                embedded_sentence.append(end_idx)

                # Pad captions shorter than longest caption
                if len(embedded_sentence) < caption_length:
                    embedded_sentence.extend([pad_idx] * (caption_length - len(embedded_sentence)))

                self.examples.append((img_id, torch.tensor(embedded_sentence)))
        
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.examples)
    

    def __getitem__(self, id):
        img_id, caption = self.examples[id]

        path = self.coco.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(f"{self.root}\\{path}").convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption


def getTrainLoaders(word2vec, caption_length):
    resnetTransform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    
    train_dataset = CocoCaptionDataset(root="coco/images",
                          annFile="coco/annotations/captions_train2014.json",
                          transform=resnetTransform,
                          word2vec=word2vec,
                          caption_length=caption_length)
    

    val_dataset = CocoCaptionDataset(root="coco/images",
                          annFile="coco/annotations/captions_val2014.json",
                          transform=resnetTransform,
                          word2vec=word2vec,
                          caption_length=caption_length)
    

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    return train_loader, val_loader


# Loops through dataset and returns list of captions
def _getCaptions(dataset):
    captions = []

    for id in range(len(dataset)):
        img_id = dataset.ids[id]
        annotations = dataset.coco.imgToAnns[img_id]

        for annotation in annotations:
            captions.append(annotation['caption'])

    return captions


# Builds the vocabulary in vectorizer using the captions in the training data
def getWord2VecEmbeddings():
    train_data = datasets.CocoCaptions(root="coco/images",
                          annFile="coco/annotations/captions_train2014.json")
    
    val_data = datasets.CocoCaptions(root="coco/images",
                          annFile="coco/annotations/captions_val2014.json")
    
    captions = _getCaptions(train_data) + _getCaptions(val_data)

    tokenized_captions = [["<PAD>"] * 5] + [["<UNK>"] * 5] + [["<START>"] * 5] + [["<END>"] * 5]
    caption_length = 0

    for caption in captions:
        tokenized_caption = nltk.word_tokenize(caption)
        
        if len(tokenized_caption) > caption_length:
            caption_length = len(tokenized_caption)

        tokenized_captions.append(tokenized_caption)

    word2vec = Word2Vec(tokenized_captions, min_count=5, seed=47)
    
    # Add 2 to caption_length to account for start and end tokens
    return word2vec, caption_length + 2
