import random

def seq2sen(batch, vocab):
    sen_list = []

    for seq in batch:
        seq_strip = seq[:seq.index(1)+1]
        sen = ' '.join([vocab.itow(token) for token in seq_strip[1:-1]])
        sen_list.append(sen)

    return sen_list

def shuffle_list(src, tgt):
    index = list(range(len(src)))
    random.shuffle(index)

    shuffle_src = []
    shuffle_tgt = []

    for i in index:
        shuffle_src.append(src[i])
        shuffle_tgt.append(tgt[i])

    return shuffle_src, shuffle_tgt
