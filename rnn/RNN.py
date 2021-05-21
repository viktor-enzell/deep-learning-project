import numpy as np
import matplotlib.pyplot as plt

class RNN():

    def __init__(self, m, K, sigma):
        # The size of the network
        self.m = m
        self.K = K
        self.sigma = sigma

        # The network
        self.RNN = {
            'b' : np.zeros((m, 1)),
            'c' : np.zeros((K, 1)),
            'U' : np.random.rand(m, K)*sigma,
            'W' : np.random.rand(m, m)*sigma,
            'V' : np.random.rand(K, m)*sigma
        }
        # Best network
        self.RNN_star = self.RNN.copy()
        # The current gradients
        self.grad = {
            'b' : np.zeros((m, 1)),
            'c' : np.zeros((K, 1)),
            'U' : np.zeros((m, K)),
            'W' : np.zeros((m, m)),
            'V' : np.zeros((K, m))
        }
        # The AdaGrad values
        self.m_theta = {
            'b' : np.zeros((m, 1)),
            'c' : np.zeros((K, 1)),
            'U' : np.zeros((m, K)),
            'W' : np.zeros((m, m)),
            'V' : np.zeros((K, m))
        }
    
    def SynthSeq(self, h0, x0, n):
        Y = np.zeros((self.K, n))
        h_t = h0
        x_t = x0

        for i in range(n):
            # Forward pass
            a_t = self.RNN['W'] @ h_t + self.RNN['U'] @ x_t + self.RNN['b'][:, 0]
            h_t = np.tanh(a_t)
            o_t = self.RNN['V'] @ h_t + self.RNN['c'][:, 0]
            p_t = Softmax(o_t)

            # Sample
            R = np.random.choice(self.K, size=1, p=p_t)
            Y[R, i] = 1
            x_t = Y[:, i]
        return Y

    def Forward(self, X, h0):
        n = X.shape[1]
        P = np.zeros((self.K, n))
        h = np.zeros((self.m, n+1))
        a = np.zeros((self.m, n))
        h[:, 0] = h0
        for i in range(n):
            # Forward pass
            a[:, i] = self.RNN['W'] @ h[:, i] + self.RNN['U'] @ X[:, i] + self.RNN['b'][:, 0]
            h[:, i+1] = np.tanh(a[:, i])
            o_t = self.RNN['V'] @ h[:, i+1] + self.RNN['c'][:, 0]
            P[:, i] = Softmax(o_t)
        
        return P, h, a

    def ComputeLoss(self, X, Y, h0):
        # Forward pass
        P, _, _ = self.Forward(X, h0)

        # Compute the loss
        loss = np.sum(-np.log(np.sum(Y * P, axis=0)))
        return loss
    
    def ComputeGradients(self, X, Y, P, h, a):
        n = X.shape[1]
        # Firstly compute g, do
        g = -(Y - P)
        # Then compute the gradient for c
        self.grad['c'] = np.sum(g, axis=1)
        # Compute the gradient for V
        self.grad['V'] = g @ h[:, 1:n+1].T
        # Then dh and da
        da = np.zeros((self.m, n))
        dh = self.RNN['V'].T @ g
        da[:, -1] = np.diag(1 - np.tanh(a[:, -1])**2) @ dh[:, -1]
        for t in range(n-2, 0, -1):
            dh[:, t] = dh[:, t] + self.RNN['W'].T @ da[:, t+1]
            da[:, t] = np.diag(1 - np.tanh(a[:, t])**2) @ dh[:, t]
        # Compute the gradient for b
        self.grad['b'] = np.sum(da, axis=1)
        # Compute the gradient for W
        self.grad['W'] = da @ h[:, :n].T
        # Compute the gradient for U
        self.grad['U'] = da @ X.T
        # Avoid the exploding gradient
        for param in self.grad:
            self.grad[param] = np.maximum(np.minimum(self.grad[param], 5), -5)

        return

    def MiniBatchGradient(self, book_data, n_update, int_to_char, seq_length=25, eta=0.1):
        # Parameter initialization
        e = 0
        eps = 1e-10
        period_disp = 1000
        period_synth = 5000
        synth_length = 200
        n_plot = int((n_update-1) / period_disp) + 1
        smooth_loss_tab = np.zeros(n_plot)
        hprev = np.zeros(self.m)
        epoch_counter = 1
        smooth_loss_part = 0.999
        loss_part = 1 - smooth_loss_part
        len_data = book_data.shape[1]
        
        for i in range(n_update):
            # Initialize the training batch data
            X = book_data[:, e:e+seq_length]
            Y = book_data[:, e+1:e+seq_length+1]

            if i % period_disp == 0:
                # Compute the loss
                print("Epoch: ", epoch_counter, ", Update: ", i)
                cur_plot = int(i / period_disp)
                if cur_plot == 0:
                    loss = self.ComputeLoss(X, Y, hprev)
                    max_smooth_loss = loss
                    min_smooth_loss = loss
                    smooth_loss = loss
                smooth_loss_tab[cur_plot] = smooth_loss
                print("Smooth loss: ", smooth_loss_tab[cur_plot])
            
            if i % period_synth == 0:
                # Synthesize test
                print("Update: ", i)
                Y_synth = self.SynthSeq(hprev, X[:, 0], synth_length)
                seq_synth = OnehotToSeq(Y_synth, int_to_char)
                print("Sequence synthesized: ")
                print(seq_synth)
              
            # Forward pass
            P, h, a = self.Forward(X, hprev)
            
            # Backward pass
            self.ComputeGradients(X, Y, P, h, a)
            
            # AdaGrad update of the RNN
            for param in self.grad:
                self.m_theta[param] = self.m_theta[param] + self.grad[param]**2
                self.RNN[param] = self.RNN[param] - eta*self.grad[param] / np.sqrt(self.m_theta[param] + eps)
            
            # Update RNN_star if it has a better loss
            loss = self.ComputeLoss(X, Y, hprev)
            smooth_loss = smooth_loss_part*smooth_loss + loss_part*loss
            if smooth_loss < min_smooth_loss:
                self.RNN_star = self.RNN.copy()
                min_smooth_loss = smooth_loss
            
            # Update e and hprev
            hprev = h[:, -1]
            e = e + seq_length
            if e > len_data-seq_length-1:
                print("One epoch done!!")
                epoch_counter += 1
                e = 0
                hprev = np.zeros(self.m)
            
        # Plot the final evolation of the training
        x = np.linspace(0, n_update, n_plot)
        plt.plot(x, smooth_loss_tab)
        plt.xlabel("Update step")
        plt.ylabel("Smooth loss")
        plt.title("Smooth loss during training")
        print("Best loss: ", min_smooth_loss)
        plt.show()
            
        return

def Softmax(o):
	P = np.exp(o) / np.sum(np.exp(o), axis=0)
	return P

def OnehotToSeq(onehot, int_to_char):
    n = onehot.shape[1]
    seq = ""

    for i in range(n):
        idx = np.where(onehot[:, i] == 1)[0][0]
        seq += int_to_char[idx]

    return seq
    