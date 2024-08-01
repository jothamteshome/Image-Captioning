import torch

from torch import nn
from torchvision import models

# Pretrained ResNet50 model used to extract features from images
class ImageFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageFeatureExtractor, self).__init__()

        # Initialize ResNet50 model with default weights
        self.resnet = models.resnet50(weights='DEFAULT')

        # Replace fully connected layer to reshape features to embedding dimension
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)

    def forward(self, images):
        return self.resnet(images)
    

# Custom LSTM network to decode image features into tokens
class LSTMNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_size, embedding_matrix):
        super(LSTMNetwork, self).__init__()

        # Initialize embedding layer from pretrained word2vec model
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(embedding_matrix))


    def forward(self, features, captions):
        # Create embedding tensor from caption tokens
        embeddings = self.embedding(captions)

        # Reshape image features and concatenate image features with embeddings
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), 1)

        # Pass embeddings and features into lstm layer then linear layer for token scores
        output, _ = self.lstm(inputs)
        output = self.linear(output)

        return output
    

# Full network to extract image features then generate captions
class ImageCaptioningNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_size, embedding_matrix):
        super(ImageCaptioningNetwork, self).__init__()

        # Feature extraction network
        self.feature_extractor = ImageFeatureExtractor(embedding_dim)

        # Caption generation network
        self.lstm_network = LSTMNetwork(embedding_dim, hidden_size, embedding_matrix)

    
    def forward(self, images, captions):

        # Generate features from pretrained network
        with torch.no_grad():
            features = self.feature_extractor(images)

        # Generate caption scores from images and captions
        output = self.lstm_network(features, captions)

        return output
