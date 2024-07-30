import torch

from torch import nn
from torchvision import models

class ImageFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageFeatureExtractor, self).__init__()

        resnet = models.resnet50(weights='DEFAULT')
        resnet.fc = nn.Linear(resnet.fc.in_features, embedding_dim)

        self.resnet = resnet


    def forward(self, images):
        return self.resnet(images)
    

class LSTMNetwork(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, hidden_size):
        super(LSTMNetwork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_embeddings)


    def forward(self, features, captions):
        embeddings = self.embedding(captions)

        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), 1)

        output, _ = self.lstm(inputs)
        output = self.linear(output)

        return output
    

class ImageCaptioningNetwork(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, hidden_size):
        super(ImageCaptioningNetwork, self).__init__()

        self.feature_extractor = ImageFeatureExtractor(embedding_dim)
        self.lstm_network = LSTMNetwork(embedding_dim, num_embeddings, hidden_size)

    
    def forward(self, images, captions):
        features = self.feature_extractor(images)
        output = self.lstm_network(features, captions)

        return output
