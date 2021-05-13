import torch
import torch.nn as nn
from load_data import load_data


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    batch_size = 50

    train_data, test_data, vocab = load_data(batch_size)
    model = TransformerModel().to(device)
