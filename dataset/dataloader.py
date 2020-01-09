import os

from .field import Field
from utils import shuffle_list

def load_data(path):
    src_vocab_path = os.path.join(path, 'vocab.en')
    tgt_vocab_path = os.path.join(path, 'vocab.de')
    src = {}
    tgt = {}

    with open(os.path.join(path, 'train.en.atok'), 'r') as f:
        src['train'] = [line.strip() for line in f.readlines()]
    with open(os.path.join(path, 'train.de.atok'), 'r') as f:
        tgt['train'] = [line.strip() for line in f.readlines()]
    with open(os.path.join(path, 'valid.en.atok'), 'r') as f:
        src['valid'] = [line.strip() for line in f.readlines()]
    with open(os.path.join(path, 'valid.de.atok'), 'r') as f:
        tgt['valid'] = [line.strip() for line in f.readlines()]
    with open(os.path.join(path, 'test.en.atok'), 'r') as f:
        src['test'] = [line.strip() for line in f.readlines()]
    with open(os.path.join(path, 'test.de.atok'), 'r') as f:
        tgt['test'] = [line.strip() for line in f.readlines()]

    if not os.path.exists(src_vocab_path):
        _make_vocab(src_vocab_path, src)
    if not os.path.exists(tgt_vocab_path):
        _make_vocab(tgt_vocab_path, tgt)

    return src, tgt

def _make_vocab(path, corpus, thres=2):
    word_dict = {}
    for line in corpus['train']:
        for token in line.split():
            if token not in word_dict:
                word_dict[token] = 0
            else:
                word_dict[token] += 1

    with open(path, 'w') as f:
        for word, count in word_dict.items():
            if count > thres:
                f.write('{}\n'.format(word))

class DataLoader:
    def __init__(self, src, tgt, batch_size, pad_idx, shuffle=False):
        assert len(src) == len(tgt), 'Number of sentences in source and target are different.'
        self.src = src
        self.tgt = tgt
        self.size = len(src)
        self.batch_size = batch_size
        self.pad_idx = pad_idx
        self.shuffle = shuffle

    def __iter__(self):
        self.index = 0

        if self.shuffle:
            self.src, self.tgt = shuffle_list(self.src, self.tgt)

        return self

    def pad(self, batch):
        max_len = 0
        for seq in batch:
            if max_len < len(seq):
                max_len = len(seq)

        for i in range(len(batch)):
            batch[i] += [self.pad_idx] * (max_len - len(batch[i]))

        return batch

    def __next__(self):
        if self.batch_size * self.index >= self.size:
            raise StopIteration

        src_batch = self.src[self.batch_size * self.index : self.batch_size * (self.index+1)]
        tgt_batch = self.tgt[self.batch_size * self.index : self.batch_size * (self.index+1)]

        src_batch = self.pad(src_batch)
        tgt_batch = self.pad(tgt_batch)

        self.index += 1

        return src_batch, tgt_batch

def get_loader(src, tgt, src_vocab, tgt_vocab, batch_size, shuffle=False):
    max_length = 50

    strip_func = lambda x: x[:max_length]
    src_field = Field(src_vocab,
                      preprocessing=None,
                      postprocessing=strip_func)
    tgt_field = Field(tgt_vocab,
                      preprocessing=lambda seq: ['<sos>'] + seq + ['<eos>'],
                      postprocessing=strip_func)

    src = [src_field(seq.split()) for seq in src]
    tgt = [tgt_field(seq.split()) for seq in tgt]

    data_loader = DataLoader(src, tgt, batch_size=batch_size, pad_idx=2, shuffle=shuffle)

    return data_loader
