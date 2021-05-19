import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from load_word_data import *
import torch
import math
import time


class TransformerModel(nn.Module):
    """
    Sequence to sequence Transformer model.
    Encoder part consists of two TransformerEncoderLayers.
    Decoder part only consists of a linear transformation.

    Inspired by the tutorial at https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    TODO: Should the model have TransformerDecoderLayers as well?
    """

    def __init__(self, vocab_size, embedding_size, encoder_nn_dim, num_encoder_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, num_heads, encoder_nn_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size
        self.decoder = nn.Linear(embedding_size, vocab_size)
        self.init_weights()

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_encoding = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(0), :]
        return self.dropout(x)


def train_one_epoch(epoch, criterion, optimizer, scheduler):
    """
    Perform a forward and backward pass for each batch of data.
    :param epoch: Epoch number.
    :param criterion: Cross-entropy loss function.
    :param optimizer: SGD optimizer.
    :param scheduler: Decays the learning rate every step_size epochs.
    """
    model.train()  # Activate training mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    log_interval = 50

    for batch_num, i in enumerate(range(0, data.size(0) - 1, bptt)):
        batch, targets = get_batch(data, i, bptt)
        optimizer.zero_grad()
        if batch.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(batch.size(0)).to(device)

        output = model(batch, src_mask)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch_num % log_interval == 0 and batch_num > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch_num, len(data) // bptt, scheduler.get_last_lr()[0],
                                                      elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def train():
    """
    Train the model for num_epochs epochs with cross-entropy loss and SGD.
    An example of generated text is printed before and after each epoch.
    """
    criterion = nn.CrossEntropyLoss()
    learning_rate = 5.0
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    best_val_loss = float('inf')
    num_epochs = 100
    best_model = None
    print(f'Example of generated text (before training): \n{generate_text()}')
    print('Starting training...')

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        train_one_epoch(epoch, criterion, optimizer, scheduler)
        val_loss = evaluate(criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print(f'| Example of generated text:\n| {generate_text()}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    print('=' * 89)
    print('| End of training | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
        best_val_loss, math.exp(best_val_loss)))
    print('=' * 89)

    torch.save(best_model.state_dict(), 'word_transformer_model.pth')
    print('Saved PyTorch Model State to word_transformer_model.pth')

    print(f'Example of generated text:\n| {generate_text(max_num_tokens=1000)}')


def evaluate(criterion):
    """
    Evaluate the model predictions with cross-entropy loss.
    """
    model.eval()  # Activate evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)

    with torch.no_grad():
        for i in range(0, data.size(0) - 1, bptt):
            batch, targets = get_batch(data, i, bptt)
            if batch.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(batch.size(0)).to(device)
            output = model(batch, src_mask)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(batch) * criterion(output_flat, targets).item()

    return total_loss / (len(data) - 1)


def generate_text(start_token='.', max_num_tokens=25):
    """
    Generate a sequence of text.
    Feed the model with a first token and let it predict the most likely next token,
    use the predicted token as the input for the next prediction.
    Repeat max_num_tokens times or until a EOS token is returned.
    """
    # TODO: Handle EOS token.

    model.eval()  # Activate evaluation mode
    previous_tokens = torch.tensor([vocab[start_token]], dtype=torch.long).to(device)
    src_mask = model.generate_square_subsequent_mask(previous_tokens.size(0)).to(device)
    predicted_tokens = []

    with torch.no_grad():
        for i in range(max_num_tokens):
            output = model(previous_tokens, src_mask)
            output_flat = output.view(-1, vocab_size)
            current_predicted_tokens = get_most_probable_tokens(output_flat, vocab)
            next_token = current_predicted_tokens[-1]
            predicted_tokens.append(next_token)
            previous_tokens = torch.cat([
                previous_tokens,
                torch.tensor([vocab[next_token]], dtype=torch.long).to(device)
            ])
            src_mask = model.generate_square_subsequent_mask(previous_tokens.size(0)).to(device)

    model.train()  # Activate training mode
    return ' '.join(predicted_tokens)


def load_model_and_generate_text():
    model.load_state_dict(torch.load("word_transformer_model.pth", map_location=device))
    print(f'Model loaded from file.\n'
          f'Example of generated text:\n'
          f'{generate_text(max_num_tokens=100)}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    data, vocab = load_data()
    vocab_size = len(vocab.stoi)
    bptt = 35

    model = TransformerModel(
        vocab_size=vocab_size,
        embedding_size=200,
        encoder_nn_dim=200,
        num_encoder_layers=2,
        num_heads=2,
        dropout=0.2
    ).to(device)

    # Choose to train or load model from file
    # by commenting out one of the lines below

    train()
    # load_model_and_generate_text()
