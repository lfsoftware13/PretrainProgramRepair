import random


def replace_code_with_mask_label(tokens, frac=0.1):
    code_length = len(tokens)
    masked_position = random_position(code_length, frac=frac)
    return masked_position


def random_position(seq_length:int, num:int=0, frac:float=0):
    if seq_length <= 0:
        return []
    if num < 0 and frac < 0:
        return []
    if num == 0:
        num = min(int(seq_length * frac), seq_length)
    return list(random.sample([i for i in range(seq_length)], num))


if __name__ == '__main__':
    res = random_position(100, frac=0.1)
    print(res)
