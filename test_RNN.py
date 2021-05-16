from collections import Counter
import numpy as np
from RNN import RNN
from tools import ComputePerplexity
'''
Test the RNN
'''
def create_vocab(C):
    char_to_int = {}
    d = len(C)

    for i in range(d):
        char_to_int[C[i]] = i
    
    return char_to_int

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
vocab = create_vocab(C)

data = chars_to_one_hot(book_chars, vocab).T

m = 100
K = data.shape[0]
sigma = 0.01
myRNN = RNN(m, K, sigma)

n_update = 100000
myRNN.MiniBatchGradient(data, n_update)

h0 = np.zeros(m)
e = 0
seq_length = 25
X = data[:, e:e+seq_length]
Y = data[:, e+1:e+seq_length+1]
P, _, _ = myRNN.Forward(X, h0)

perplexity = ComputePerplexity(P, Y)
print(perplexity)