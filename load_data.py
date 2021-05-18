from collections import Counter
from torchtext.vocab import Vocab
import torch
import numpy as np


def load_data(batch_size):
    """
    One-hot encodes all characters in book and creates batches
    for training data and test data.
    train_data: one-hot encoded characters
                from book_chars[0] to book_chars[book_length - 1].
                divided into batches of size batch_size.
    test_data: one-hot encoded characters
                from book_chars[1] to book_chars[book_length].
                divided into batches of size batch_size.
    vocab: vocabulary used in book.
    """
    book = open('goblet_book.txt', 'r')
    counter = Counter()
    book_chars = []

    for line in book.readlines():
        counter.update(list(line))
        book_chars += list(line)

    book_length = len(book_chars)
    vocab = Vocab(counter)

    data = chars_to_one_hot(book_chars, vocab)

    train_data = batch_data(data[0:book_length - 1], batch_size)
    test_data = batch_data(data[1:book_length], batch_size)

    return train_data, test_data, vocab


def batch_data(data, batch_size):
    return torch.split(data, batch_size)


def chars_to_one_hot(chars, vocab):
    indexes = np.array([vocab[char] for char in chars])

    one_hot = np.zeros((indexes.size, indexes.max() + 1))
    one_hot[np.arange(indexes.size), indexes] = 1
    one_hot = torch.as_tensor(one_hot)

    return one_hot


def one_hot_to_chars(one_hots, vocab):
    indexes = torch.argmax(one_hots, dim=1)
    chars = [vocab.itos[index] for index in indexes]

    return chars
