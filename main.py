import torch

from torch import nn, optim
from tqdm import tqdm


from get_dataloaders import getTrainLoaders, getWord2VecEmbeddings
from get_models import ImageCaptioningNetwork

# Handles training step for a single epoch
def train_epoch(model, optimizer, train_loader, device):
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Set model to training mode and initialize loss
    model.train()
    running_loss = 0


    # Train on all batches of training dataloader
    for images, captions in tqdm(train_loader, unit='batch', desc='Training'):
        optimizer.zero_grad()

        # Move data to device
        images = images.to(device)
        captions = captions.to(device)

        # Compute predictions and loss
        outputs = model(images, captions)

        # First position comes from concatenating features with embeddings
        loss = criterion(outputs[:, 1:], captions)
        running_loss += loss.item()

        # Handle backpropagation
        loss.backward()
        optimizer.step()

    return running_loss / len(train_loader)


# Function that runs the evaluation step of the model
def evaluate_model(model, dataloader, device):
    # Set model to evaluation mode and initialize loss
    model.eval()
    running_loss = 0

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        # Evaluate all batches of dataloader
        for images, captions in dataloader:
            # Move data to device
            images = images.to(device)
            captions = captions.to(device)

            # Compute predictions and loss
            outputs = model(images)
            loss = criterion(outputs, captions)
            running_loss += loss.item()


    return running_loss / len(dataloader)


def train_model(model, train_loader, val_loader, device, num_epochs=5):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, train_loader, device)
        val_loss = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1} | train_loss: {train_loss} | val_loss: {val_loss}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    word2vec, caption_length = getWord2VecEmbeddings()
    train_loader, val_loader = getTrainLoaders(word2vec, caption_length)

    model = ImageCaptioningNetwork(embedding_dim=256,
                                   num_embeddings=len(word2vec.wv.key_to_index),
                                   hidden_size=256).to(device)


    train_model(model, train_loader, val_loader, device)




if __name__ == "__main__":
    main()