import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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

def train(model, seq_len, lr, n_epoch, data, char_to_idx, idx_to_char, gen_seq_len, save_model = False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    f = open(fname, "a")
    for i_epoch in range(1, n_epoch+1):
        data_ptr = 0
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
        f.write("{0}\t{1}\n".format(i_epoch,running_loss/n))
        scheduler.step()
        print("\nEpoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))
        print("\nSmooth Loss: {0:.8f}".format(smooth_loss))
        synthesize_text(model, gen_seq_len, ".", char_to_idx, idx_to_char)
    f.close()
    if save_model:
        torch.save(model.state_dict(), "lstm-hn-{0}-lr-{1}-data-{2}.pth".format(n_hidden, lr,datapercent))

def synthesize_text(model, gen_seq_len, start_char, char_to_idx, idx_to_char):
    data_ptr = 0
    hidden_state = None

    input_seq = torch.tensor([[char_to_idx[start_char]]]).to(dev)
    while True:
        output, hidden_state = model(input_seq, hidden_state)

        output = F.softmax(torch.squeeze(output), dim=0)
        dist = Categorical(output)
        index = dist.sample()

        print(idx_to_char[index.item()], end="")

        input_seq[0][0] = index.item()
        data_ptr += 1

        if data_ptr > gen_seq_len:# and ix_to_char[index.item()] in stopsigns:
            break


def prepare_data(data_path, dev, datapercent = 100):
    data = open(data_path,"r").read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    data_size = int(data_size *datapercent/100)
    char_to_idx = {ch:i for i, ch in enumerate(chars)}
    idx_to_char = {i:ch for i, ch in enumerate(chars)}

    data = list(data)[:data_size]
    for i, ch in enumerate(data):
        data[i] = char_to_idx[ch]
    data = torch.tensor(data).to(dev)
    data = torch.unsqueeze(data, dim=1)

    return data, data_size, vocab_size, char_to_idx, idx_to_char

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 25
    gen_seq_len = 200
    n_hidden = 512
    lr=0.02
    datapercent = 100
    n_epoch = 25

    stopsigns = [".", "!", "?"]
    book_path = "../goblet_book.txt"
    fname = "lstm-hn-{0}-lr-{1}-data-{2}.txt".format(n_hidden, lr,datapercent)

    data, data_size, vocab_size, char_to_idx, idx_to_char = prepare_data(book_path, dev, datapercent=datapercent)

    print("Dataset loaded")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    
    model = LSTM(vocab_size, vocab_size, n_hidden, 3).to(dev)
    train(model, seq_len, lr, n_epoch, data, char_to_idx, idx_to_char, gen_seq_len)
    synthesize_text(model, 1000, ".", char_to_idx, idx_to_char)