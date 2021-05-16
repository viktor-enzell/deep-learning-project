from collections import Counter
from torchtext.vocab import Vocab
import torch
import numpy as np


def load_data():
    """
    One-hot encodes all characters in book.
    """
    book = open('goblet_book.txt', 'r')
    counter = Counter()
    book_chars = []

    for line in book.readlines():
        counter.update(list(line))
        book_chars += list(line)

    vocab = Vocab(counter)
    data = chars_to_one_hot(book_chars, vocab)

    return data, vocab


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def batch_data(data, batch_size):
    return torch.split(data, batch_size)


def chars_to_one_hot(chars, vocab):
    indexes = np.array([vocab[char] for char in chars])

    one_hot = np.zeros((indexes.size, indexes.max() + 1))
    one_hot[np.arange(indexes.size), indexes] = 1
    one_hot = torch.as_tensor(one_hot, dtype=torch.long)

    return one_hot


def one_hot_to_chars(one_hots, vocab):
    indexes = torch.argmax(one_hots, dim=1)
    chars = [vocab.itos[index] for index in indexes]

    return chars
