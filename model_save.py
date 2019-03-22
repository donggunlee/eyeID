import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

class StackedBRNN(nn.Module):
    def __init__(self, input_size, rnn_hidden, rnn_layers,
                 fc_hidden, fc_layers, rnn_type=nn.LSTM,
                 dropout=0, nClass=57): # fc_hidden = 128
        super(StackedBRNN, self).__init__()

        self.dropout = dropout
        self.rnn_layers = rnn_layers
        self.rnns = nn.ModuleList()

        for i in range(rnn_layers):
            input_size = input_size if i == 0 else 2 * rnn_hidden
            self.rnns.append(rnn_type(input_size, rnn_hidden,
                                      num_layers=1,
                                      bidirectional=True))
        fc = []
        for i in range(fc_layers-1):
            input_size = rnn_hidden*rnn_layers*2 + 110 if i==0 else fc_hidden
            # input_size = rnn_hidden*rnn_layers*2 if i==0 else fc_hidden

            fc.append(nn.Linear(input_size, fc_hidden))
            fc.append(nn.ReLU(inplace=True))

        fc.append(nn.Linear(fc_hidden, nClass))
        # fc.append(nn.Softmax(dim=1)) # Crossentropy loss include logsoftmax

        self.fc = nn.Sequential(*fc)

    def forward(self, x, trial_vec):
        # Transpose batch and sequence dims # (NxTx2) --> (TxNx2)
        x = x.transpose(0, 1)

        # Encode all layers
        # pdb.set_trace()
        outputs = [x]
        for i in range(self.rnn_layers):
            rnn_input = outputs[-1]
            # Apply dropout to hidden input
            if self.dropout > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        output = torch.cat(outputs[1:], 2) # TxNxH

        # Average across time
        # output = torch.mean(output, 0) # TxNxH --> NxH

        # Select the last hidden value
        output_rnn = output[-1, :, :]    # TxNxH --> NxH

        # Normalize the output of RNN so that the sum is 1
        # output = output / sum(sum(output))

        # Concat with trial info
        output = torch.cat((output_rnn, trial_vec), 1)

        # MLP
        output = self.fc(output)

        return output_rnn, output