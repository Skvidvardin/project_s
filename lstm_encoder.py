import torch
from torch import nn

class LSTMEncoder(nn.Module):
    """
    This module encodes a sequence into a single vector using an LSTM.
    """

    def __init__(self, in_features, hidden_size: int = 200,
                 batch_first: bool = True,
                 bidirectional: bool = True):

        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=batch_first,
                            bidirectional=bidirectional)

    def forward(self, x, mask, lengths):

        packed_sequence = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        outputs, (hx, cx) = self.lstm(packed_sequence)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if self.lstm.bidirectional:
            final = torch.cat([hx[-2], hx[-1]], dim=-1)
        else:
            final = hx[-1]

        return outputs, final
