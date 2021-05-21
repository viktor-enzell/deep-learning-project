from collections import Counter
import numpy as np
import os
from rnn.RNN import RNN
from tools import ComputePerplexity
'''
Test the RNN
'''
def create_vocab(C):
    char_to_int = {}
    int_to_char = {}
    d = len(C)

    for i in range(d):
        char_to_int[C[i]] = i
        int_to_char[i] = C[i]
    
    return char_to_int, int_to_char

def chars_to_one_hot(chars, vocab):
    indexes = np.array([vocab[char] for char in chars])

    one_hot = np.zeros((indexes.size, indexes.max() + 1))
    one_hot[np.arange(indexes.size), indexes] = 1

    return one_hot

book = open('goblet_book.txt', 'r')
counter = Counter()
book_chars = []

for line in book.readlines():
    counter.update(list(line))
    book_chars += list(line)

book_length = len(book_chars)
C = list(set(book_chars))
char_to_int, int_to_char = create_vocab(C)

data = chars_to_one_hot(book_chars, char_to_int).T

m = 100
K = data.shape[0]
sigma = 1/m
myRNN = RNN(m, K, sigma)

n_update = 100000
myRNN.MiniBatchGradient(data, n_update, int_to_char)

h0 = np.zeros(m)
e = 0
seq_length = 25
X = data[:, e:e+seq_length]
Y = data[:, e+1:e+seq_length+1]
P, _, _ = myRNN.Forward(X, h0)

perplexity = ComputePerplexity(P, Y)
print(perplexity)