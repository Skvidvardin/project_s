from collections import Counter, OrderedDict
import numpy as np


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                           OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        self.add_token("<unk>")
        self.add_token("<pad>")

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)


def load_glove(glove_path, vocab, glove_dim=300):
    """
    Load Glove embeddings and update vocab.
    :param glove_path:
    :param vocab:
    :param glove_dim:
    :return:
    """
    vectors = []
    w2i = {}
    i2w = []

    # Random embedding vector for unknown words
    vectors.append(np.random.uniform(
        -0.05, 0.05, glove_dim).astype(np.float32))
    w2i["<unk>"] = 0
    i2w.append("<unk>")

    # Zero vector for padding
    vectors.append(np.zeros(glove_dim).astype(np.float32))
    w2i["<pad>"] = 1
    i2w.append("<pad>")

    with open(glove_path, mode="r", encoding="utf-8") as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            w2i[word] = len(vectors)
            i2w.append(word)
            vectors.append(np.array(vec.split(), dtype=np.float32))

    # fix brackets
    w2i[u'-LRB-'] = w2i.pop(u'(')
    w2i[u'-RRB-'] = w2i.pop(u')')

    i2w[w2i[u'-LRB-']] = u'-LRB-'
    i2w[w2i[u'-RRB-']] = u'-RRB-'

    vocab.w2i = w2i
    vocab.i2w = i2w

    return np.stack(vectors)
