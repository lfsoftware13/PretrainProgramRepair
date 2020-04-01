from thop import profile
from thop import clever_format
import torch
import torch.nn as nn

from seq2seq.models import EncoderRNN


def ffn_params():
    vocab_size = 5000
    hidden_size = 512
    model = nn.Sequential(nn.Embedding(vocab_size, hidden_size),
                   nn.Linear(hidden_size, hidden_size),
                   nn.ReLU(),
                   nn.Linear(hidden_size, vocab_size))
    inp = torch.ones([1, 200]).long()
    macs, params = profile(model, inputs=[inp])
    macs, params = clever_format([macs, params], "%.3f")
    print('计算量：{}， 参数量：{}'.format(macs, params))



def transformer_params():
    from model.transformer.transformer import Transformer
    model = Transformer(n_src_vocab=5000, n_trg_vocab=5000, src_pad_idx=0, trg_pad_idx=0, d_word_vec=512, d_model=512,
                        d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
                        trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True, bidirectional_decoder=False)
    inp = torch.ones([1, 200]).long()
    macs, params = profile(model, inputs=[inp, inp])
    macs, params = clever_format([macs, params], "%.3f")
    print('计算量：{}， 参数量：{}'.format(macs, params))


class simple_gru(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.output = nn.Linear(2*hidden_size, vocab_size)
    def forward(self, inp):
        inp = self.embed(inp)
        inp, _ = self.rnn(inp)
        inp = self.output(inp)
        return inp


def rnn_params():
    vocab_size = 5000
    max_len = 200
    hidden_size = 512

    model = simple_gru(vocab_size, hidden_size, num_layers=3)
    # model = EncoderRNN(vocab_size, max_len, input_size=hidden_size, hidden_size=hidden_size,
    #              n_layers=3, bidirectional=True, rnn_cell='gru', variable_lengths=False)
    inp = torch.ones([1, max_len]).long()
    macs, params = profile(model, inputs=[inp])
    macs, params = clever_format([macs, params], "%.3f")
    print('计算量：{}， 参数量：{}'.format(macs, params))


# def seq2seq_params():
#     from seq2seq.models import Seq2seq
#     vocab_size = 5000
#     max_len = 200
#     hidden_size = 512
#     encoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=3, batch_first=True,
#                       bidirectional=True)
#     decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional=True)
#     model = nn.Sequential(nn.Embedding(vocab_size, hidden_size),
#                           Seq2seq(encoder, decoder))
#     # model = Seq2seq(encoder, decoder)
#     inp = torch.ones([1, max_len]).long()
#     macs, params = profile(model, inputs=[inp])
#     macs, params = clever_format([macs, params], "%.3f")
#     print('计算量：{}， 参数量：{}'.format(macs, params))




if __name__ == '__main__':
    rnn_params()
