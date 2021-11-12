import torch
from torch import nn
import numpy as np


class GRUCellV2(nn.Module):
    """
    GRU cell implementation
    """
    def __init__(self, input_size, hidden_size, activation=torch.tanh):
        """
        Initializes a GRU cell

        :param      input_size:      The size of the input layer
        :type       input_size:      int
        :param      hidden_size:     The size of the hidden layer
        :type       hidden_size:     int
        :param      activation:      The activation function for a new gate
        :type       activation:      callable
        """
        super(GRUCellV2, self).__init__()
        self.activation = activation

        # initialize weights by sampling from a uniform distribution between -K and K
        K = 1 / np.sqrt(hidden_size)
        # weights
        self.w_ih = nn.Parameter(torch.rand(3 * hidden_size, input_size) * 2 * K - K)
        self.w_hh = nn.Parameter(torch.rand(3 * hidden_size, hidden_size) * 2 * K - K)
        self.b_ih = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)
        self.b_hh = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)


    def forward(self, x, h):
        """
        Performs a forward pass through a GRU cell.
        Returns the current hidden state h_t for every datapoint in batch.
        
        :param      x:    an element x_t in a sequence
        :type       x:    torch.Tensor
        :param      h:    previous hidden state h_{t-1}
        :type       h:    torch.Tensor
        """

        stacked_i = torch.mm(x, self.w_ih.T) + self.b_ih
        stacked_h = torch.mm(h, self.w_hh.T) + self.b_hh

        ri, zi, ni = torch.chunk(stacked_i, 3, 1)
        rh, zh, nh = torch.chunk(stacked_h, 3, 1)

        rt = torch.sigmoid(ri + rh)
        zt = torch.sigmoid(zi + zh)
        nt = torch.tanh(ni + rt * nh)

        return (1 - zt) * nt + zt * h


class GRU2(nn.Module):
    """
    GRU network implementation
    """
    def __init__(self, input_size, hidden_size, bias=True, activation=torch.tanh, bidirectional=False):
        super(GRU2, self).__init__()
        self.bidirectional = bidirectional
        self.fw = GRUCellV2(input_size, hidden_size, activation=activation) # forward cell
        if self.bidirectional:
            self.bw = GRUCellV2(input_size, hidden_size, activation=activation) # backward cell
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """
        Performs a forward pass through the whole GRU network, consisting of a number of GRU cells.
        Takes as input a 3D tensor `x` of dimensionality (B, T, D),
        where B is the batch size;
              T is the sequence length (if sequences have different lengths, they should be padded before being inputted to forward)
              D is the dimensionality of each element in the sequence, e.g. word vector dimensionality

        The method returns a 3-tuple of (outputs, h_fw, h_bw), if self.bidirectional is True,
                           a 2-tuple of (outputs, h_fw), otherwise
        `outputs` is a tensor containing the output features h_t for each t in each sequence (the same as in PyTorch native GRU class);
                  NOTE: if bidirectional is True, then it should contain a concatenation of hidden states of forward and backward cells for each sequence element.
        `h_fw` is the last hidden state of the forward cell for each sequence, i.e. when t = length of the sequence;
        `h_bw` is the last hidden state of the backward cell for each sequence, i.e. when t = 0 (because the backward cell processes a sequence backwards)
        
        :param      x:    a batch of sequences of dimensionality (B, T, D)
        :type       x:    torch.Tensor
        """
        w_hy = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        b_hy = nn.Parameter(torch.rand(self.hidden_size))
        h = torch.randn(x.shape[0], self.hidden_size)
        outputs = []

        # Forward pass
        for i in range(x.shape[1]):
            h = self.fw.forward(x[:, i, :], h)
            output = torch.sigmoid(torch.mm(h, w_hy.T) + b_hy)
            outputs.append(output)

        h_bw = h.clone()
        # Backward pass
        if self.bidirectional:
            for i in range(x.shape[1] - 1, -1, -1):
                h_bw = self.bw.forward(x[:, i, :], h_bw)
                output = torch.sigmoid(torch.mm(h, w_hy.T) + b_hy)
                outputs.append(output)
            return torch.stack(outputs, dim=1), h, h_bw
        else:
            return torch.stack(outputs, dim=1), h


def is_identical(a, b):
    #print("Diff: ", np.abs(a - b))
    return "Yes" if np.all(np.abs(a - b) < 1e-6) else "No"


if __name__ == '__main__':
    seq_length = 35
    torch.manual_seed(100304343)
    x = torch.randn(5, seq_length, 10)
    gru = nn.GRU(10, 20, bidirectional=False, batch_first=True)
    outputs, h = gru(x)
    #print("output 1: ", outputs.shape)

    torch.manual_seed(100304343)
    x = torch.randn(5, seq_length, 10)
    gru2 = GRU2(10, 20, bidirectional=False)
    outputs, h_fw = gru2(x)
    #print('outputs 2: ', outputs.shape)

    print("Checking the unidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))

    torch.manual_seed(100304343)
    x = torch.randn(5, seq_length, 10)
    gru = GRU2(10, 20, bidirectional=True)
    outputs, h_fw, h_bw = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, seq_length, 10)
    gru2 = nn.GRU(10, 20, bidirectional=True, batch_first=True)
    outputs, h = gru2(x)

    print("Checking the bidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))
    print("Same hidden states of the backward cell?\t{}".format(
        is_identical(h[1].detach().numpy(), h_bw.detach().numpy())
    ))