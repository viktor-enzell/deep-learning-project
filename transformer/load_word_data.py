from collections import Counter
from torchtext.vocab import Vocab
import torch
from torchtext.data.utils import get_tokenizer


def load_data():
    """
    Read all tokens/words in book and create a vocabulary.
    Crate tensor representations of the tokens using vocabulary indexes of tokens.
    Batch the data.

    Data is handled similarly to https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    print('Loading data...')

    book = open('../goblet_book.txt', 'r')
    counter = Counter()
    tokenizer = get_tokenizer('basic_english')

    for line in book.readlines():
        counter.update(tokenizer(line))
    book.close()

    vocab = Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    data = data_process(tokenizer, vocab)

    num_batches = 20
    data = batch_data(data, num_batches)

    return data, vocab


def data_process(tokenizer, vocab):
    """
    Read all data in book and store token indexes in tensor.
    """
    book = open('../goblet_book.txt', 'r')

    data = [torch.tensor([vocab[token] for token in tokenizer(line)],
                         dtype=torch.long) for line in book.readlines()]
    book.close()

    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batch_data(data, num_batches):
    """
    Batching data in same way as:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Divide the dataset into num_batches parts.
    batch_size = data.size(0) // num_batches

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, batch_size * num_batches)

    # Evenly divide the data across the batch_size batches.
    data = data.view(num_batches, -1).t().contiguous()
    return data.to(device)


def get_batch(data, i, batch_seq_len):
    """
    Get a batch of data and corresponding targets.
    """
    seq_len = min(batch_seq_len, len(data) - 1 - i)
    batch = data[i:i + seq_len]
    target = data[i + 1:i + 1 + seq_len].reshape(-1)
    return batch, target
