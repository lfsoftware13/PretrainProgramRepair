import torch
import torch.nn as nn
import torch.nn.functional as F

from common.problem_util import to_cuda
from common.util import PaddedList
from model.transformer.transformer import Encoder, Decoder, Transformer


class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pad_idx, n_head=2, n_layers=1, dropout=0.2, max_length=500):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_head = n_head
        self.dropout = dropout

        self.d_k_v_q = int(hidden_size/n_head)

        self.transformer = Transformer(n_src_vocab=vocab_size, n_trg_vocab=vocab_size, src_pad_idx=pad_idx, trg_pad_idx=pad_idx,
            d_word_vec=hidden_size, d_model=hidden_size, d_inner=hidden_size,
            n_layers=n_layers, n_head=n_head, d_k=self.d_k_v_q, d_v=self.d_k_v_q, dropout=dropout, n_position=max_length,
            trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=True)

    def forward(self, inp_seq, inp_seq_len, target_seq, target_len, inp_mask=None, target_mask=None):
        o = self.transformer(inp_seq, target_seq)
        return o


def create_loss_fn(ignore_id):
    cross_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)
    def loss_fn(input_seq, target_seq):
        loss = cross_loss(input_seq, target_seq)
        return loss
    return loss_fn


def create_parse_input_batch_data_fn(ignore_id):
    def parse_input_tensor(batch_data):
        input_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['input_seq'], fill_value=ignore_id)))
        return [input_seq]
    return parse_input_tensor


def create_parse_target_batch_data_fn(ignore_id):
    def parse_target_tensor(batch_data):
        target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['target_seq'], fill_value=ignore_id)))
        return [target_seq]
    return parse_target_tensor


