import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, layer_num=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, inp_seq, inp_seq_len):
        inp_seq = self.embedding(inp_seq)

        pass




