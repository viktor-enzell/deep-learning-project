import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
class LSTM(nn.Module):
    def __init__(self,input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = 0.05)
        self.connected = nn.Linear(hidden_size, output_size)

    def forward(self, sequence, state):
        embedding = self.embedding(sequence)
        output, state = self.lstm(embedding, state)
        output = self.connected(output)
        return output, (state[0].detach(), state[1].detach())

def train():
    return

def prepare_data(data_path):
    data = open(data_path,"r").read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    chars_to_idx = {ch:i for i, ch in enumerate(chars)}
    idx_to_char = {i:ch for i, ch in enumerate(chars)}

    data = list(data)
    for i, ch in enumerate(data):
        data[i] = chars_to_idx[ch]
    data = torch.tensor(data).to(dev)
    data = torch.unsqueeze(data, dim=1)

    return data, data_size, vocab_size, chars_to_idx, idx_to_char

if __name__ == "__main__":
    stopsigns = [".", "!", "?"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    book_path = "../goblet_book.txt"
    seq_len = 25
    op_seq_len = 200
    data = open(book_path, "r").read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print("----------------------------------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")

    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]
    
    data = torch.tensor(data).to(dev)
    data = torch.unsqueeze(data,dim=1)
    
    model = LSTM(vocab_size, vocab_size, 512, 3).to(dev)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    for i_epoch in range(1, 101):
        data_ptr = np.random.randint(100)
        n = 0
        running_loss = 0
        smooth_loss = 0
        hidden_state = None

        while True:
            input_seq = data[data_ptr:data_ptr + seq_len]
            target_seq = data[data_ptr + 1: data_ptr + seq_len + 1]

            output, hidden_state = model.forward(input_seq, hidden_state)

            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))

            running_loss += loss.item()
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_ptr += seq_len
            n += 1
            
            if data_ptr + seq_len + 1 > data_size:
                break
        scheduler.step()
        print("\nEpoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))
        print("\nSmooth Loss: {0:.8f}".format(smooth_loss))
        data_ptr = 0
        hidden_state = None

        input_seq = torch.tensor([[char_to_ix["."]]]).to(dev)
        while True:
            output, hidden_state = model(input_seq, hidden_state)

            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample()

            print(ix_to_char[index.item()], end="")

            input_seq[0][0] = index.item()
            data_ptr += 1

            if data_ptr > op_seq_len and ix_to_char[index.item()] in stopsigns:
                break
    torch.save(model.state_dict(), "../test_lstm.pth")