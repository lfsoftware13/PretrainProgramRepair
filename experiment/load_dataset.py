import pandas as pd
import random
from common.util import CustomerDataSet, show_process_map
from read_data.read_experiment_data import read_fake_common_deepfix_error_dataset_with_limit_length
from vocabulary.word_vocabulary import Vocabulary

MAX_LENGTH = 500


class MaskedDataset(CustomerDataSet):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 MAX_LENGTH=500,
                 only_smaple=False):
        # super().__init__(data_df, vocabulary, set_type, transform, no_filter)
        self.set_type = set_type
        self.vocabulary = vocabulary
        self.max_length = MAX_LENGTH
        self.only_sample = only_smaple

        if data_df is not None:
            self.data_df = self.filter_df(data_df)
            self._samples = [row for i, row in self.data_df.iterrows()]

    def filter_df(self, df):
        df = df[df['input_seq'].map(lambda x: x is not None)]
        df = df[df['input_seq'].map(lambda x: len(x) < self.max_length)]
        return df

    def _get_raw_sample(self, row):
        sample = {}
        sample['id'] = row['id']
        sample['includes'] = row['includes']
        sample['masked_positions'] = row['masked_positions']

        sample['input_seq'] = row['input_seq']
        sample['input_seq_name'] = row['input_seq_name']
        sample['input_seq_len'] = len(sample['input_seq'])

        sample['target_seq'] = row['target_seq']
        sample['target_seq_len'] = len(row['target_seq'])
        sample['target_seq_name'] = row['ac_code_name']

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


def load_deepfix_masked_dataset(is_debug, vocabulary):
    from experiment.load_datadict import load_deepfix_masked_datadict
    if is_debug:
        data_dicts = load_deepfix_masked_datadict(100)
    else:
        data_dicts = load_deepfix_masked_datadict()

    datasets = [MaskedDataset(pd.DataFrame(dd), vocabulary, name)
                for dd, name in zip(data_dicts, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "valid", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    return datasets
