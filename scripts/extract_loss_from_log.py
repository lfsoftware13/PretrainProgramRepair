import re


loss_rule = r'epoch: ([0-9]+) loss: ([0-9]+\.[0-9]+)\.*'
masked_accuracy_rule = r'MaskedLanguageModelTokenAccuracy: ([0-9]+\.[0-9]+)\.*'
position_predict_rule = r'GraphPositionPredictAccuracy: ([0-9]+\.[0-9]+)\.*'
program_fix_rule = r'ErrorPositionAndValueAccuracy is_copy evaluate:([0-9]+\.[0-9]+), position evaluate:([0-9]+\.[0-9]+), output evaluate:([0-9]+\.[0-9]+), all evaluate: ([0-9]+\.[0-9]+)'


def create_extract_loss_re(rule, info_num):
    pattern = re.compile(rule)

    def extract_loss_re(line):
        res = pattern.search(line)
        if res is None:
            return None
        else:
            info = [res.group(i) for i in range(1, info_num+1)]
            return info
    return extract_loss_re


def extract_info_from_file_each_line(file_path, extract_fn_list):
    result = [[] for _ in range(len(extract_fn_list))]
    for line in open(file_path, 'r'):
        for i in range(len(extract_fn_list)):
            res = extract_fn_list[i](line)
            if res is not None:
                result[i].append(res)
    return result


def re_rule_test(extract_fn, line):
    res = extract_fn(line)
    if res is not None:
        print(res)
    else:
        print('not found')


if __name__ == '__main__':
    # line1 = 'epoch: 31 loss: 1.4095200300216675, train evaluator :'
    # line2 = 'MaskedLanguageModelTokenAccuracy: 0.867382934358914'
    # line3 = 'GraphPositionPredictAccuracy: 0.9864198315219196'
    # line4 = 'ErrorPositionAndValueAccuracy is_copy evaluate:0.9745495495495495, position evaluate:0.8807432432432433, output evaluate:0.46664712778429074, all evaluate: 0.8132882882882883'
    #
    extract_loss_fn = create_extract_loss_re(loss_rule, info_num=2)
    # re_rule_test(extract_loss_fn, line1)
    extract_masked_accuracy_fn = create_extract_loss_re(masked_accuracy_rule, info_num=1)
    # re_rule_test(extract_masked_accuracy_fn, line2)
    extract_position_predict_fn = create_extract_loss_re(position_predict_rule, info_num=1)
    # re_rule_test(extract_position_predict_fn, line3)
    extract_program_fix_fn = create_extract_loss_re(program_fix_rule, info_num=4)
    # re_rule_test(extract_program_fix_fn, line4)
    # file_path = r'/home/lf/Project/PretrainProgramRepair/log/pretrain_masked_model1.log'
    file_path = r'./pretrain_log/pretrain_encoder_sample_config7_pretrain45.log'
    res = extract_info_from_file_each_line(file_path, [extract_loss_fn, extract_program_fix_fn])
    print(res)

