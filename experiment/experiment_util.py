import json


# ---------------------------- convert c code fields to cpp fields ---------------------------------- #
from common.analyse_include_util import extract_include, replace_include_with_blank


def convert_one_action(one_action):
    CHANGE = 0
    INSERT = 1
    DELETE = 2
    from common.action_constants import ActionType
    if isinstance(one_action['act_type'], ActionType):
        one_action['act_type'] = one_action['act_type'].value
    if one_action['act_type'] == 4:
        one_action['act_type'] = CHANGE
    elif one_action['act_type'] == 3:
        one_action['act_type'] = DELETE
    elif one_action['act_type'] == 1:
        one_action['act_type'] = INSERT
    else:
        print('error_one_action', one_action)
        return None
    return one_action


def convert_action_map_to_old_action(actions):
    CHANGE = 0
    INSERT = 1
    DELETE = 2
    is_str = isinstance(actions, str)

    if is_str:
        new_actions_obj = json.loads(actions)
    else:
        new_actions_obj = actions
    old_actions_obj = [convert_one_action(one_action) for one_action in new_actions_obj]
    old_actions_obj = list(filter(lambda x: x is not None, old_actions_obj))
    if is_str:
        old_actions = json.dumps(old_actions_obj)
    else:
        old_actions = old_actions_obj
    return old_actions


def convert_c_code_fields_to_cpp_fields(df, convert_include=True):
    filter_macro_fn = lambda code: not (code.find('define') != -1 or code.find('defined') != -1 or
                                        code.find('undef') != -1 or code.find('pragma') != -1 or
                                        code.find('ifndef') != -1 or code.find('ifdef') != -1 or
                                        code.find('endif') != -1)
    df['action_character_list'] = df['modify_action_list'].map(convert_action_map_to_old_action)
    if convert_include:
        df['includes'] = df['similar_code'].map(extract_include)
    df['similar_code_without_include'] = df['similar_code'].map(replace_include_with_blank)
    df['ac_code'] = df['similar_code_without_include']
    df['error_count'] = df['distance']
    df = df[df['similar_code'].map(filter_macro_fn)]
    return df


def convert_deepfix_to_c_code(df):
    filter_macro_fn = lambda code: not (code.find('define') != -1 or code.find('defined') != -1 or
                                        code.find('undef') != -1 or code.find('pragma') != -1 or
                                        code.find('ifndef') != -1 or code.find('ifdef') != -1 or
                                        code.find('endif') != -1)
    df['includes'] = df['code'].map(extract_include)
    df['code_with_include'] = df['code']
    df['code'] = df['code_with_include'].map(replace_include_with_blank)
    df = df[df['code'].map(filter_macro_fn)]
    return df