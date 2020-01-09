class Vocab:
    def __init__(self, init_token, eos_token, pad_token, unk_token):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.vocab_list = []
        self.vocab_dict = None

    def load(self, path):
        self.vocab_list.append(self.init_token)
        self.vocab_list.append(self.eos_token)
        self.vocab_list.append(self.pad_token)
        self.vocab_list.append(self.unk_token)

        with open(path, 'r') as f:
            for token in f.readlines():
                token = token.strip()
                self.vocab_list.append(token)

        self.vocab_dict = {
            v: k for k, v in enumerate(self.vocab_list)
        }

    def __len__(self):
        return len(self.vocab_list)

    def itow(self, index):
        assert index < len(self.vocab_list), 'Invalid token accessed.'
        return self.vocab_list[index]

    def wtoi(self, word):
        if word in self.vocab_dict:
            return self.vocab_dict[word]
        else:
            return self.vocab_dict[self.unk_token]

class Field:
    def __init__(self, vocab, preprocessing, postprocessing):
        self.vocab = vocab
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        if self.preprocessing is not None:
            return self.preprocessing(x)
        return x

    def postprocess(self, x):
        if self.postprocessing is not None:
            return self.postprocessing(x)
        return x

    def numericalize(self, x):
        return [self.vocab.wtoi(token) for token in x]

    def __call__(self, x):
        return self.postprocess(
            self.numericalize(
                self.preprocess(x)
            )
        )
