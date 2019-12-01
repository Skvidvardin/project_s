import re
from collections import namedtuple
import numpy as np
import random

import torch

Example = namedtuple("Example", ["tokens", "label", "token_labels"])

def sst_reader(path, lower=False):
    """
    Reads in examples
    :param path:
    :param lower:
    :return:
    """
    file = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            file.append(line.strip().replace("\\", ""))

    for line in file:
        line = line.lower() if lower else line
        line = re.sub("\\\\", "", line)
        tokens = re.findall(r"\([0-9] ([^\(\)]+)\)", line)
        label = int(line[1])
        token_labels = list(map(int, re.findall(r"\(([0-9]) [^\(\)]", line)))
        assert len(tokens) == len(token_labels), "mismatch tokens/labels"
        yield Example(tokens=tokens, label=label, token_labels=token_labels)


def prepare_minibatch(mb, vocab, device=None, sort=True):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)
    reverse_map = None
    lengths = np.array([len(ex.tokens) for ex in mb])
    maxlen = lengths.max()

    # vocab returns 0 if the word is not there
    x = [[vocab.w2i.get(t, 0) for t in ex.tokens] + [1] * (maxlen - len([vocab.w2i.get(t, 0) for t in ex.tokens])) for ex in mb]
    y = [ex.label for ex in mb]

    x = np.array(x)
    y = np.array(y)

    if sort:  # required for LSTM
        sort_idx = np.argsort(lengths)[::-1]
        x = x[sort_idx]
        y = y[sort_idx]

        # to put back into the original order
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    return x, y, reverse_map


def get_minibatch(data, batch_size=25, shuffle=False):

    if shuffle:
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    for example in data:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch
        