import more_itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz.sandbox import unzip

from c_parser.ast_parser import parse_ast_code_graph
from model.detect_error_model import ErrorDetectorModel
from model.masked_language_model import MaskedLanguageModel


def set_model_requires_grad(m, requires_grad):
    for p in m.parameters():
        p.requires_grad = requires_grad
    if requires_grad:
        m.train()
    else:
        m.eval()


class PretrainMaskedCodeModel(nn.Module):
    def __init__(self, vocabulary, mask_language_model_param, detect_token_model_param, train_type, ignore_id, pad_id):
        '''
        :param vocabulary:
        :param mask_language_model_param:
        :param detect_token_model_param:
        :param train_type: 'only_gene', 'only_disc', 'both', 'none'
        '''
        super().__init__()
        self.vocabulary = vocabulary
        self.generator = MaskedLanguageModel(**mask_language_model_param)
        self.discriminator = ErrorDetectorModel(**detect_token_model_param)
        self.train_type = train_type
        self.ignore_id = ignore_id
        self.pad_id = pad_id

        self.change_model_train_type(self.train_type)

    def change_model_train_type(self, train_type):
        if self.train_type == train_type:
            return
        self.train_type = train_type

        if self.train_type == 'only_gene':
            set_model_requires_grad(self.generator, True)
            set_model_requires_grad(self.discriminator, False)
        elif self.train_type == 'only_disc':
            set_model_requires_grad(self.generator, False)
            set_model_requires_grad(self.discriminator, True)
        elif self.train_type == 'both':
            set_model_requires_grad(self.generator, True)
            set_model_requires_grad(self.discriminator, True)
        elif self.train_type == 'none':
            set_model_requires_grad(self.generator, False)
            set_model_requires_grad(self.discriminator, False)

    def forward(self, inp_seq, inp_seq_len, inp_mask, target_seq, target_len, target_mask, masked_positions):
        masked_output_logit = self.generator(inp_seq, inp_seq_len, inp_mask, target_seq, target_len, target_mask, masked_positions)

        masked_output_ids = self.extract_result_from_logit(masked_output_logit)
        output_tokens_seq = self.create_gene_output_code_seq(inp_seq, target_seq, masked_output_ids)

        disc_inputs = self.prepare_discriminator_input(output_tokens_seq, target_seq)
        disc_targets = self.prepare_discriminator_target(output_tokens_seq, target_seq)

        disc_outputs_logits = self.discriminator(*disc_inputs)
        return masked_output_logit, disc_outputs_logits, disc_inputs, disc_targets

    def prepare_discriminator_input(self, output_tokens_seq, target_seq):
        output_length_list = torch.sum(torch.ne(target_seq, self.ignore_id), dim=-1).tolist()
        output_seq_list = self.transform_input_tensor_to_list(output_tokens_seq, output_length_list)

        output_names_list = [[self.vocabulary.id_to_word(i) for i in one] for one in output_seq_list]

        res = [generate_one_graph_input(one, self.vocabulary) for one in output_names_list]
        disc_input_names, disc_input_length, disc_adj = list(zip(*res))

        disc_input_seq = [[self.vocabulary.word_to_id(i) for i in one] for one in disc_input_names]

        disc_inputs = parse_graph_input_from_mask_lm_output(disc_input_seq, disc_input_length, disc_adj)

        return disc_inputs

    def prepare_discriminator_target(self, output_tokens_seq, target_seq):
        gene_target_seq = parse_graph_output_from_mask_lm_output(output_tokens_seq, target_seq, ignore_id=self.ignore_id)
        return [gene_target_seq]

    def extract_result_from_logit(self, seq_logit):
        output_seq = torch.squeeze(torch.topk(F.softmax(seq_logit, dim=-1), dim=-1, k=1)[1], dim=-1)
        return output_seq

    def create_gene_output_code_seq(self, input_seq, target_seq, gene_output_seq):
        masked_positions = torch.ne(target_seq, self.ignore_id)
        output_tokens_seq = torch.where(masked_positions, gene_output_seq, input_seq)
        return output_tokens_seq

    def transform_input_tensor_to_list(self, input_seq, input_length):
        input_seq_list = input_seq.tolist()
        input_seq_list = [inp[:l] for inp, l in zip(input_seq_list, input_length)]
        return input_seq_list


def generate_one_graph_input(input_token_names, vocabulary):
    code_graph = parse_ast_code_graph(input_token_names)
    input_length = code_graph.graph_length + 2
    in_seq, graph = code_graph.graph
    begin_token = vocabulary.begin_tokens[0]
    end_token = vocabulary.end_tokens[0]
    input_seq = [begin_token] + in_seq + [end_token]
    adj = [[a + 1, b + 1] for a, b, _ in graph] + [[b + 1, a + 1] for a, b, _ in graph]
    return input_seq, input_length, adj


def parse_graph_input_from_mask_lm_output(input_seq, input_length, adj, use_ast=True):
    from common.problem_util import to_cuda
    from common.util import PaddedList

    def to_long(x):
        return to_cuda(torch.LongTensor(x))

    if not use_ast:
        adjacent_matrix = to_long(adj)
    else:
        adjacent_tuple = [[[i] + tt for tt in t] for i, t in enumerate(adj)]
        adjacent_tuple = [list(t) for t in unzip(more_itertools.flatten(adjacent_tuple))]
        size = max(input_length)
        # print("max length in this batch:{}".format(size))
        adjacent_tuple = torch.LongTensor(adjacent_tuple)
        adjacent_values = torch.ones(adjacent_tuple.shape[1]).long()
        adjacent_size = torch.Size([len(input_length), size, size])
        # info('batch_data input_length: ' + str(batch_data['input_length']))
        # info('size: ' + str(size))
        # info('adjacent_tuple: ' + str(adjacent_tuple.shape))
        # info('adjacent_size: ' + str(adjacent_size))
        adjacent_matrix = to_cuda(
            torch.sparse.LongTensor(
                adjacent_tuple,
                adjacent_values,
                adjacent_size,
            ).float().to_dense()
        )
    input_seq = to_long(PaddedList(input_seq))
    input_length = to_long(input_length)
    return adjacent_matrix, input_seq, input_length


def parse_graph_output_from_mask_lm_output(input_seq, target_seq, ignore_id=-1, check_error=True):
    mask = torch.ne(target_seq, ignore_id)
    target = torch.eq(input_seq, target_seq).int()
    target = torch.where(mask, target, -1)
    return target


