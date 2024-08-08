from utils.data import getTrainLoaders
from utils.models import loadModel
from utils.train import trainModel, TrainingArgs


def main():
    train_args = TrainingArgs()

    # Load dataset
    train_loader, val_loader = getTrainLoaders()

    # Load pretrained model if path is passed in, otherwise load new model
    model = loadModel().to(train_args.device)

    # Run training loop for model
    trainModel(model, train_loader, val_loader, train_args)

if __name__ == "__main__":
    main()