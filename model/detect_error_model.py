import torch
import torch.nn as nn

from model.graph_encoder_model import GraphEncoder


class ErrorDetectorModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query',
                 p2_type='static', p2_step_length=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.graph_encoder = GraphEncoder(hidden_size=hidden_size,
                 graph_embedding=graph_embedding,
                 graph_parameter=graph_parameter,
                 pointer_type=pointer_type,
                 embedding=self.embedding, embedding_size=hidden_size,
                 p2_type=p2_type,
                 p2_step_length=p2_step_length,
                 do_embedding=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, adjacent_matrix, inp_seq, inp_seq_len):
        _, _, encoder_logit = self.graph_encoder(adjacent_matrix, inp_seq, inp_seq_len)
        output_logit = self.output(encoder_logit).squeeze(-1)
        return output_logit


def create_loss_fn(ignore_id):
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def loss_fn(output_logit, target_replaced):
        mask = torch.ne(output_logit, ignore_id)
        replaced_loss = bce_loss(output_logit, target_replaced) * mask.float()
        replaced_loss = torch.sum(replaced_loss) / torch.sum(mask)
        return replaced_loss
    return loss_fn


if __name__ == '__main__':
    graph_parameter = {"rnn_parameter": {'vocab_size': 1000,
                                                   'max_len': 500, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': True, },
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 }
    m = ErrorDetectorModel(400, 1000, 'mixed', graph_parameter, pointer_type='query', p2_type='step', p2_step_length=2)
