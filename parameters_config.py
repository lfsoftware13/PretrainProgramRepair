from common.evaluate_util import MaskedLanguageModelTokenAccuracy
from common.opt import OpenAIAdam
from common.pycparser_util import tokenize_by_clex_fn
from config import DATA_RECORDS_DEEPFIX_PRETRAIN_GEN_DBPATH
from experiment.load_data_vocabulary import load_deepfix_common_error_vocabulary
import pandas as pd


def test_masked_generator_model1(is_debug):
    vocabulary = load_deepfix_common_error_vocabulary()
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    fill_value = 0
    use_ast = False
    if use_ast:
        from experiment.load_data_vocabulary import load_graph_vocabulary
        vocabulary = load_graph_vocabulary(vocabulary)
    tokenize_fn = tokenize_by_clex_fn()

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    epoch_ratio = 1.0
    sample_count = 1008

    from experiment.load_dataset import load_deepfix_masked_dataset
    datasets = load_deepfix_masked_dataset(is_debug=is_debug, vocabulary=vocabulary)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.masked_language_model import MaskedLanguageModel
    from model.masked_language_model import create_parse_input_batch_data_fn
    from model.masked_language_model import create_parse_target_batch_data_fn
    from model.masked_language_model import create_loss_fn
    from common.evaluate_util import TokenAccuracy
    from model.masked_language_model import create_output_ids_fn
    return {
        'name': 'test_masked_generator_model1',
        'save_name': 'test_masked_generator_model1.pkl',
        'load_model_name': 'test_masked_generator_model1.pkl',
        # 'logger_file_path': 'graph_encoder_sample_config2.log',

        'do_save_records_to_database': False,
        'db_path': DATA_RECORDS_DEEPFIX_PRETRAIN_GEN_DBPATH,
        'table_basename': 'test_masked_generator_model1',
        'change_output_records_to_batch_fn': None,
        'create_save_database_records_fn': None,

        'model_fn': MaskedLanguageModel,
        'model_dict':
            {"vocab_size": vocabulary.vocabulary_size,
             "hidden_size": 400,
             "pad_idx": fill_value,
             "n_head": 8,
             "n_layers": 3,
             "max_length": max_length,
             'dropout': 0.2,
             'bidirectional_decoder': True,
             # 'model_type': 'only_encoder',
             'model_type': 'seq2seq',
             },

        'use_ast': use_ast,

        'do_sample_evaluate': False,

        'log_file_path': '/dev/shm/main.log',
        'extract_includes_fn': lambda x: x['includes'],
        'print_output': False,
        'print_output_fn': None,

        'max_save_distance': 15,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(),
        'parse_target_batch_data_fn': create_parse_target_batch_data_fn(ignore_id),
        'expand_output_and_target_fn': None,
        'create_output_ids_fn': create_output_ids_fn(),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [MaskedLanguageModelTokenAccuracy(ignore_token=ignore_id)],

        'ac_copy_train': False,
        'ac_copy_radio': 0.2,

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets
    }