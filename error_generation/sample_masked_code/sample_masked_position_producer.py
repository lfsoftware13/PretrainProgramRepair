import json

from common.pycparser_util import tokenize_by_clex_fn
from common.util import ensure_file_path
from config import deepfix_masked_position_path
from error_generation.sample_masked_code.sample_masked_position import random_position
from experiment.experiment_util import convert_c_code_fields_to_cpp_fields
from experiment.parse_xy_util import create_name_list_by_LexToken
from read_data.read_filter_data_records import read_fake_deepfix_common_error_records


def tokenize_ac_code(df, tokenize_fn):
    df['res'] = ''
    df['ac_code_obj'] = df['ac_code'].map(tokenize_fn)
    df = df[df['ac_code_obj'].map(lambda x: x is not None)].copy()
    df['ac_code_obj'] = df['ac_code_obj'].map(list)
    print('after tokenize: ', len(df.index))
    df['ac_code_name'] = df['ac_code_obj'].map(create_name_list_by_LexToken)
    return df


def save_sample_masked_position_dict(data_dict, save_path):
    ensure_file_path(save_path)
    with open(save_path, mode='w') as f:
        data_str = json.dumps(data_dict)
        f.write(data_str)
    print('save {} ids to {}'.format(len(data_dict.keys()), save_path))


def sample_masked_position_main():
    data_df = read_fake_deepfix_common_error_records()
    data_df = convert_c_code_fields_to_cpp_fields(data_df)

    tokenize_fn = tokenize_by_clex_fn()
    data_df = tokenize_ac_code(data_df, tokenize_fn)

    data_df['ac_code_length'] = data_df['ac_code_name'].map(len)
    data_df['masked_positions'] = data_df['ac_code_length'].map(lambda l: random_position(l, frac=0.4))
    data_df['masked_positions_token'] = data_df.apply(lambda one: [one['ac_code_name'][pos] for pos in one['masked_positions']], axis=1)

    data_dict = {one['id']: (one['masked_positions'], one['masked_positions_token']) for i, one in data_df.iterrows()}
    # data_dict = {i: (masked_poses, masked_toks) for i, masked_poses, masked_toks in zip(
    #              data_df['id'].tolist(), data_df['masked_positions'].tolist(), data_df['masked_tokens'].tolist())}

    save_sample_masked_position_dict(data_dict, save_path=deepfix_masked_position_path)


if __name__ == '__main__':
    sample_masked_position_main()
