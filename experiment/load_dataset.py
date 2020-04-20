import pandas as pd
import random
from sklearn.utils import shuffle

from c_parser.pycparser.pycparser import c_ast
from common.util import CustomerDataSet, show_process_map
from error_generation.sample_masked_code.sample_masked_position import random_position
from read_data.read_experiment_data import read_fake_common_deepfix_error_dataset_with_limit_length
from vocabulary.word_vocabulary import Vocabulary

MAX_LENGTH = 500


class MaskedDataset(CustomerDataSet):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 MAX_LENGTH=500,
                 only_smaple=False,
                 random_mask_position=False,
                 mask_frac=0.2,
                 train_type='both'):
        # super().__init__(data_df, vocabulary, set_type, transform, no_filter)
        self.set_type = set_type
        self.vocabulary = vocabulary
        self.max_length = MAX_LENGTH
        self.only_sample = only_smaple
        self.masked_id = vocabulary.word_to_id('<MASK>')
        self.random_mask_position = random_mask_position
        self.mask_frac = mask_frac
        self.train_type = train_type

        if data_df is not None:
            self.data_df = self.filter_df(data_df)
            self._samples = [row for i, row in self.data_df.iterrows()]

    def filter_df(self, df):
        df = df[df['input_seq'].map(lambda x: x is not None)]
        df = df[df['input_seq'].map(lambda x: len(x) < self.max_length)]
        return df

    def init_epoch(self):
        if self.random_mask_position:
            for row in self._samples:
                row['masked_positions'] = random_position(row['ac_seq_len'], frac=self.mask_frac)

        if self.train_type == 'only_disc':
            for row in self._samples:
                self.replace_half_token_with_the_other(row)
        elif self.train_type == 'bert':
            for row in self._samples:
                self.replace_token_with_mask_bert(row, mask_name='mask')
        else:
            for row in self._samples:
                self.replace_token_with_mask(row)

        return

    def replace_token_with_mask(self, row, mask_name='<MASK>'):
        row['input_seq'] = row['ac_seq'].copy()
        row['input_seq_name'] = row['ac_code_name'].copy()
        row['target_seq'] = [-1 for _ in range(len(row['input_seq']))]
        for p in row['masked_positions']:
            row['input_seq'][p] = self.masked_id
            row['input_seq_name'][p] = mask_name
            row['target_seq'][p] = row['ac_seq'][p]
        return row

    def replace_token_with_mask_bert(self, row, mask_name='<MASK>', copy_frac=0.1, stay_frac=0.1):
        row['input_seq'] = row['ac_seq'].copy()
        row['input_seq_name'] = row['ac_code_name'].copy()
        row['target_seq'] = [-1 for _ in range(len(row['input_seq']))]

        mask_position, copy_position, stay_position = self.random_copy_and_stay_position(row['masked_positions'])

        for p in mask_position:
            row['input_seq'][p] = self.masked_id
            row['input_seq_name'][p] = mask_name
            row['target_seq'][p] = row['ac_seq'][p]
        copy_position = shuffle(list(copy_position))
        copy_replaced_p = set(random_position(row['ac_seq_len'], num=len(copy_position)))
        for p, p_t in zip(copy_position, copy_replaced_p):
            row['input_seq'][p] = row['ac_seq'][p_t]
            row['input_seq_name'][p] = row['ac_code_name'][p_t]
            row['target_seq'][p] = row['ac_seq'][p]
        for p in stay_position:
            row['target_seq'][p] = row['ac_seq'][p]
        return row

    def replace_half_token_with_the_other(self, row):
        positions = row['masked_positions']
        half_len = int(len(positions) / 2)
        row['target_seq'] = [-1 for _ in range(len(row['input_seq']))]
        for i, p in enumerate(positions[:half_len]):
            copy_p = positions[i + half_len]
            row['input_seq'][p] = row['ac_seq'][copy_p]
            row['input_seq_name'][p] = row['ac_code_name'][copy_p]
            row['target_seq'][p] = row['ac_seq'][p]
        for p in positions[half_len:]:
            row['input_seq'][p] = row['ac_seq'][p]
            row['input_seq_name'][p] = row['ac_code_name'][p]
            row['target_seq'][p] = row['ac_seq'][p]
        return row

    def random_copy_and_stay_position(self, all_position, copy_frac=0.1, stay_frac=0.1):
        all_position = set(all_position)
        position_num = len(all_position)
        copy_position = set(random.sample(all_position, int(position_num * copy_frac)))
        retain_position = all_position - copy_position
        stay_position = set(random.sample(retain_position, int(position_num * stay_frac)))
        mask_position = retain_position - stay_position
        return mask_position, copy_position, stay_position

    def _get_raw_sample(self, row):
        sample = {}
        sample['id'] = row['id']
        sample['includes'] = row['includes']
        sample['masked_positions'] = row['masked_positions']

        sample['ac_seq'] = row['ac_seq']
        sample['ac_seq_len'] = row['ac_seq_len']
        sample['ac_seq_name'] = row['ac_code_name']

        sample['input_seq'] = row['input_seq']
        sample['input_seq_name'] = row['input_seq_name']
        sample['input_seq_len'] = len(sample['input_seq'])

        sample['target_seq'] = row['target_seq']
        sample['target_seq_len'] = len(row['target_seq'])

        return sample

    def add_samples(self, df):
        df = self.filter_df(df)
        self._samples += [row for i, row in df.iterrows()]

    def remain_samples(self, count=0, frac=1.0):
        if count != 0:
            self._samples = random.sample(self._samples, count)
        elif frac != 1:
            count = int(len(self._samples) * frac)
            self._samples = random.sample(self._samples, count)

    def combine_dataset(self, dataset):
        d = MaskedDataset(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type)
        d._samples = self._samples + dataset._samples
        return d

    def remain_dataset(self, count=0, frac=1.0):
        d = MaskedDataset(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type)
        d._samples = self._samples
        d.remain_samples(count=count, frac=frac)
        return d

    def __getitem__(self, index):
        return self._get_raw_sample(self._samples[index])

    def __len__(self):
        return len(self._samples)


def load_graph_vocabulary(vocabulary):
    vocabulary.add_token("<Delimiter>")
    ast_node_dict = c_ast.__dict__
    for n in sorted(ast_node_dict):
        s_c = ast_node_dict[n]
        b_c = c_ast.Node
        try:
            if issubclass(s_c, b_c):
                vocabulary.add_token(n)
        except Exception as e:
            pass
    return vocabulary


def load_deepfix_masked_dataset(is_debug, vocabulary, random_mask_position=True, train_type='both'):
    from experiment.load_datadict import load_deepfix_masked_datadict
    if is_debug:
        data_dicts = load_deepfix_masked_datadict(100)
    else:
        data_dicts = load_deepfix_masked_datadict()

    datasets = [MaskedDataset(pd.DataFrame(dd), vocabulary, name, random_mask_position=random_mask_position,
                              train_type=train_type)
                for dd, name in zip(data_dicts, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "valid", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    return datasets
