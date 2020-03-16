from common.constants import CACHE_DATA_PATH
from common.pycparser_util import tokenize_by_clex_fn
from common.util import disk_cache
from experiment.experiment_util import convert_c_code_fields_to_cpp_fields, add_masked_position_column
from experiment.load_data_vocabulary import load_deepfix_common_error_vocabulary
from experiment.parse_xy_util import parse_masked_code
from read_data.read_experiment_data import read_fake_common_deepfix_error_dataset_with_limit_length
from read_data.read_train_ids import read_deepfix_masked_position


@disk_cache(basename='load_deepfix_masked_datadict', directory=CACHE_DATA_PATH)
def load_deepfix_masked_datadict(sample_count=None):
    vocab = load_deepfix_common_error_vocabulary()
    train, valid, test = read_fake_common_deepfix_error_dataset_with_limit_length(500)

    tokenize_fn = tokenize_by_clex_fn()
    position_dict = read_deepfix_masked_position()

    def prepare_df(df):
        if sample_count is not None and sample_count > 0:
            df = df.sample(sample_count)
        df = convert_c_code_fields_to_cpp_fields(df, convert_include=False)
        df = add_masked_position_column(df, position_dict)
        return df

    train = prepare_df(train)
    valid = prepare_df(valid)
    test = prepare_df(test)

    parse_param = (vocab, tokenize_fn)

    train_data = parse_masked_code(train, *parse_param)
    valid_data = parse_masked_code(valid, *parse_param)
    test_data = parse_masked_code(test, *parse_param)

    train_dict = {**train_data, 'includes': train['includes'], 'id': train['id'],
                  'masked_positions': train['masked_positions']}
    valid_dict = {**valid_data, 'includes': valid['includes'], 'id': valid['id'],
                  'masked_positions': valid['masked_positions']}
    test_dict = {**test_data, 'includes': test['includes'], 'id': test['id'],
                 'masked_positions': test['masked_positions']}

    return train_dict, valid_dict, test_dict
