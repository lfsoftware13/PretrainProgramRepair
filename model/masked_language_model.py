import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer.transformer import Encoder, Decoder



class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pad_idx, n_head=2, n_layers=1, dropout=0.2, max_length=500):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_head = n_head
        self.dropout = dropout

        self.encoder = Encoder(vocab_size, hidden_size, n_layers, n_head, d_k=hidden_size, d_v=hidden_size,
                               d_model=self.hidden_size, d_inner=self.hidden_size, pad_idx=pad_idx, dropout=0.1,
                               n_position=max_length)

        self.decoder = Decoder(vocab_size, hidden_size, n_layers, n_head, d_k=hidden_size, d_v=hidden_size,
                               d_model=self.hidden_size, d_inner=self.hidden_size, pad_idx=pad_idx,
                               n_position=max_length, dropout=0.1)

    def forward(self, inp_seq, inp_seq_len, target_seq, target_len, inp_mask=None, target_mask=None):
        encoder_output = self.encoder(inp_seq)
        dec_output = self.decoder(target_seq, target_mask, encoder_output, inp_mask)
        pass




