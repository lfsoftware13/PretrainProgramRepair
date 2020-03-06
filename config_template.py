import os

root = r'G:\Project\PretrainProgramRepair'
temp_code_write_path = r'tmp'
scrapyOJ_path = r'G:\Project\program_ai\data\scrapyOJ.db'
save_model_root = os.path.join(root, 'trained_model')
cache_path = os.path.join(root, 'data', 'cache_data')
summarization_source_code_to_method_name_path = r'G:\Project\dataset\json'

DEEPFIX_DB = r'G:\Project\deepfix_dataset\deepfix.db'
SLK_SAMPLE_DBPATH = os.path.join(root, 'data', 'slk_sample_data.db')
FAKE_DEEPFIX_ERROR_DATA_DBPATH = os.path.join(root, 'data', 'fake_deepfix_error_data.db')
num_processes = 2
DATA_RECORDS_DEEPFIX_DBPATH = os.path.join(root, 'data', 'data_records_deepfix.db')
DATA_RECORDS_DEEPFIX_CODEFORCES_TRAIN_DBPATH = os.path.join(root, 'data', 'data_records_deepfix_codeforces_train.db')

train_ids_path = r'.\text_file\train_ids.txt'
valid_ids_path = r'.\text_file\valid_ids.txt'
test_ids_path = r'.\text_file\test_ids.txt'