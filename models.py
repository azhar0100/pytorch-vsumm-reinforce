import torch
import torch.nn as nn
from torch.nn import functional as F
from vasnet_model import SelfAttention

class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        # assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        # elif cell == 'attention':
        	# self.rnn = SelfAttention(input_size=in_dim,output_size=)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p

