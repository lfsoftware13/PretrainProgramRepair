from common.evaluate_util import MaskedLanguageModelTokenAccuracy, GraphPositionPredictAccuracy
from common.opt import OpenAIAdam
from common.pycparser_util import tokenize_by_clex_fn
from config import DATA_RECORDS_DEEPFIX_PRETRAIN_GEN_DBPATH
from experiment.load_data_vocabulary import load_deepfix_common_error_vocabulary
import pandas as pd

from experiment.load_dataset import load_graph_vocabulary
from model.generate_detect_model import PretrainMaskedCodeModel


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


def pretrain_masked_model1(is_debug):
    from c_parser.ast_parser import set_ast_config_attribute
    set_ast_config_attribute("add_sequence_link", False)

    vocabulary = load_deepfix_common_error_vocabulary()
    use_ast = True
    if use_ast:
        vocabulary = load_graph_vocabulary(vocabulary)

    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    fill_value = 0
    tokenize_fn = tokenize_by_clex_fn()

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    epoch_ratio = 1.0
    sample_count = 1008
    check_error_task = True
    random_mask_position = True
    train_type = 'both'

    from experiment.load_dataset import load_deepfix_masked_dataset
    datasets = load_deepfix_masked_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                           random_mask_position=random_mask_position, train_type=train_type)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.generate_detect_model import create_parse_input_batch_data_fn
    from model.generate_detect_model import create_parse_target_batch_data_fn
    from model.generate_detect_model import create_loss_fn
    from model.generate_detect_model import create_output_ids_fn
    return {
        'name': 'pretrain_masked_model1',
        'save_name': 'pretrain_masked_model1.pkl',
        'load_model_name': 'pretrain_masked_model1.pkl',
        # 'logger_file_path': 'pretrain_masked_model1.log',

        'do_save_records_to_database': False,
        'db_path': DATA_RECORDS_DEEPFIX_PRETRAIN_GEN_DBPATH,
        'table_basename': 'test_masked_generator_model1',
        'change_output_records_to_batch_fn': None,
        'create_save_database_records_fn': None,
# [vocabulary, mask_language_model_param, detect_token_model_param, train_type, ignore_id, pad_id, check_error_task]
        'model_fn': PretrainMaskedCodeModel,
        'model_dict':{
            "vocabulary": vocabulary,
            "mask_language_model_param": {
                "vocab_size": vocabulary.vocabulary_size,
                "hidden_size": 400,
                "pad_idx": fill_value,
                "n_head": 8,
                "n_layers": 3,
                "max_length": max_length,
                'dropout': 0.2,
                'bidirectional_decoder': True,
                # 'model_type': 'only_encoder',
                # 'model_type': 'seq2seq',
            },
            # hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query', p2_type='static', p2_step_length=0, check_error_task=True
            "detect_token_model_param": {
                "hidden_size": 400,
                "vocab_size": vocabulary.vocabulary_size,
                "graph_embedding": "mixed",
                "graph_parameter": {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                      'max_len': max_length, 'input_size': 400,
                                                      'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                      'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                      'variable_lengths': False, 'embedding': None,
                                                      'update_embedding': True, },
                                    "graph_type": "ggnn",
                                    "graph_itr": 3,
                                    "dropout_p": 0.2,
                                    "mask_ast_node_in_rnn": False
                                    },
                "pointer_type": 'query',
                "p2_type": "step",
                "p2_step_length": 2,
                "check_error_task": check_error_task,
            },
            "train_type": train_type,
            "ignore_id": ignore_id,
            "pad_id": pad_id,
            "check_error_task": check_error_task,
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
        'create_output_ids_fn': create_output_ids_fn(train_type=train_type),
        'train_loss': create_loss_fn(ignore_id, check_error_task=check_error_task, train_type=train_type),
        'evaluate_object_list': [MaskedLanguageModelTokenAccuracy(ignore_token=ignore_id, train_type=train_type),
                                 GraphPositionPredictAccuracy(ignore_token=ignore_id)],

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


def pretrain_masked_model2(is_debug):
    from c_parser.ast_parser import set_ast_config_attribute
    set_ast_config_attribute("add_sequence_link", False)

    vocabulary = load_deepfix_common_error_vocabulary()
    use_ast = True
    if use_ast:
        vocabulary = load_graph_vocabulary(vocabulary)

    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    inner_begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[1])
    inner_end_id = vocabulary.word_to_id(vocabulary.end_tokens[1])
    pad_id = vocabulary.word_to_id(vocabulary.addition_tokens[0])
    fill_value = 0
    tokenize_fn = tokenize_by_clex_fn()

    batch_size = 64
    epoches = 80
    ignore_id = -1
    max_length = 500
    epoch_ratio = 1.0
    sample_count = 1008
    check_error_task = True
    random_mask_position = True
    train_type = 'only_disc'

    from experiment.load_dataset import load_deepfix_masked_dataset
    datasets = load_deepfix_masked_dataset(is_debug=is_debug, vocabulary=vocabulary,
                                           random_mask_position=random_mask_position, train_type=train_type)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.generate_detect_model import create_parse_input_batch_data_fn
    from model.generate_detect_model import create_parse_target_batch_data_fn
    from model.generate_detect_model import create_loss_fn
    from model.generate_detect_model import create_output_ids_fn
    return {
        'name': 'pretrain_masked_model2',
        'save_name': 'pretrain_masked_model2.pkl',
        'load_model_name': 'pretrain_masked_model2.pkl',
        # 'logger_file_path': 'pretrain_masked_model1.log',

        'do_save_records_to_database': False,
        'db_path': DATA_RECORDS_DEEPFIX_PRETRAIN_GEN_DBPATH,
        'table_basename': 'test_masked_generator_model1',
        'change_output_records_to_batch_fn': None,
        'create_save_database_records_fn': None,
# [vocabulary, mask_language_model_param, detect_token_model_param, train_type, ignore_id, pad_id, check_error_task]
        'model_fn': PretrainMaskedCodeModel,
        'model_dict':{
            "vocabulary": vocabulary,
            "mask_language_model_param": {
                "vocab_size": vocabulary.vocabulary_size,
                "hidden_size": 400,
                "pad_idx": fill_value,
                "n_head": 8,
                "n_layers": 3,
                "max_length": max_length,
                'dropout': 0.2,
                'bidirectional_decoder': True,
                # 'model_type': 'only_encoder',
                # 'model_type': 'seq2seq',
            },
            # hidden_size, vocab_size, graph_embedding, graph_parameter, pointer_type='query', p2_type='static', p2_step_length=0, check_error_task=True
            "detect_token_model_param": {
                "hidden_size": 400,
                "vocab_size": vocabulary.vocabulary_size,
                "graph_embedding": "mixed",
                "graph_parameter": {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                      'max_len': max_length, 'input_size': 400,
                                                      'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                      'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                      'variable_lengths': False, 'embedding': None,
                                                      'update_embedding': True, },
                                    "graph_type": "ggnn",
                                    "graph_itr": 3,
                                    "dropout_p": 0.2,
                                    "mask_ast_node_in_rnn": False
                                    },
                "pointer_type": 'query',
                "p2_type": "step",
                "p2_step_length": 2,
                "check_error_task": check_error_task,
            },
            "train_type": train_type,
            "ignore_id": ignore_id,
            "pad_id": pad_id,
            "check_error_task": check_error_task,
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
        'create_output_ids_fn': create_output_ids_fn(train_type=train_type),
        'train_loss': create_loss_fn(ignore_id, check_error_task=check_error_task, train_type=train_type),
        'evaluate_object_list': [MaskedLanguageModelTokenAccuracy(ignore_token=ignore_id, train_type=train_type),
                                 GraphPositionPredictAccuracy(ignore_token=ignore_id)],

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