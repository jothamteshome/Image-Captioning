from utils.data import getTrainLoaders, getFinalEvalLoader
from utils.evaluate import BLEUEval
from utils.models import loadModel
from utils.train import trainModel, TrainingArgs

def train():
    # Initialize training arguments
    train_args = TrainingArgs(num_epochs=7, learning_rate=5e-4)

    # Load training and validation data
    train_loader, val_loader = getTrainLoaders()

    # Load pretrained model if path is passed in, otherwise load new model
    model = loadModel().to(train_args.device)

    # Run training loop for model
    trainModel(model, train_loader, val_loader, train_args)

    # Compute BLEU score on trained model
    bleu_loader = getFinalEvalLoader()
    bleu_score = BLEUEval(model, bleu_loader, train_args.device)
    train_args.saveMetrics({'bleu_score': bleu_score})


if __name__ == "__main__":
    train()