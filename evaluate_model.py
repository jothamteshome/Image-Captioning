import torch

from torch import nn
from torcheval.metrics import BLEUScore
from tqdm import tqdm

torch.ops.load_library("build//lib.win-amd64-cpython-311//decode_captions.cp311-win_amd64.pyd")
        

def generateCaptions(model, images, word2vec, max_tokens, device):
    key2idx = word2vec.wv.key_to_index

    encoded_captions = torch.full((images.shape[0], 1), key2idx.get("<start>")).to(device)

    for _ in range(max_tokens):
        with torch.no_grad():
            output = model(images, encoded_captions)
            probabilities = nn.functional.softmax(output[:, -1], dim=1)

            generated_token_ids = torch.topk(probabilities, k=5, dim=1)[1][:, 4].view(-1, 1)
            encoded_captions = torch.cat([encoded_captions, generated_token_ids], dim=1)

            if (generated_token_ids == key2idx.get('<end>')).any():
                break

    
    captions = torch.ops.decode_captions.decodeBatchedCaptions(encoded_captions.cpu(), word2vec.wv.index_to_key)

    return captions


# Function that runs the evaluation step of the model
def evaluate_model(model, dataloader, device, word2vec, max_tokens):
    # Set model to evaluation mode and initialize loss
    model.eval()

    bleu_score = BLEUScore(n_gram=2)

    with torch.no_grad():
        # Evaluate all batches of dataloader
        for images, reference_captions in tqdm(dataloader, unit='batch', desc='Evaluating'):
            # Move data to device
            images = images.to(device)

            generated_captions = generateCaptions(model, images, word2vec, max_tokens, device)

            bleu_score.update(generated_captions, reference_captions)

    score = bleu_score.compute()
    print(score)

    return score