import numpy as np

chars = " 'abcdefghijklmnopqrstuvwxyz"
int2char = {i: c for i, c in enumerate(chars)}
char2int = {c: i for i, c in enumerate(chars)}


def one_hot_encode(y):
    output = []
    if type(y) is np.array:
        for i in y:
            char_prob = np.zeros(29)
            char_prob[i] = 1
            output.append(char_prob)
    else:
        for char in y:
            char_prob = np.zeros(29)
            char_prob[char2int[char]] = 1
            output.append(char_prob)
    return np.array(output)

