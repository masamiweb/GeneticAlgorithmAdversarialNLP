"""
    Author: Manjinder Singh

"""

import os
from keras.preprocessing.text import Tokenizer

import pickle as pickle


def read_text(path):
    pos_path = path + '/pos'
    neg_path = path + '/neg'
    pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
    neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

    pos_list = [open(x, 'r', encoding="utf8").read().lower() for x in pos_files]
    neg_list = [open(x, 'r', encoding="utf8").read().lower() for x in neg_files]
    data_list = pos_list + neg_list  # create a list of lists, with positive at start and negative after

    # number of 1's that correspond to the num of positive reviews
    # number of 0's that correspond to negative reviews
    # i.e 1 = positive and = negative
    # make sure order is preserved here so the sentiments match up with the data_list
    labels_list = [1] * len(pos_list) + [0] * len(neg_list)
    return data_list, labels_list


class IMDBDataset(object):
    def __init__(self, path='imdb_dataset/imdb', max_vocab_size=None):  # constructor
        self.path = path
        self.train_path = path + '/train'
        self.test_path = path + '/test'
        #self.vocab_path = path + '/imdb.vocab'
        self.max_vocab_size = max_vocab_size

        # self._read_vocab()  # open the vocabulary file
        train_text, self.train_y = read_text(self.train_path)
        test_text, self.test_y = read_text(self.test_path)
        self.train_text = train_text
        self.test_text = test_text
        print('tokenizing...')

        # Tokenized text of training data
        self.tokenizer = Tokenizer()

        #  tokenizer.fit_on_texts method creates the vocabulary index based on word frequency
        self.tokenizer.fit_on_texts(self.train_text)

        # if we don't specify a vovalulary size then set the max size to (tokenizer.word_index + 1)
        if max_vocab_size is None:
            max_vocab_size = len(self.tokenizer.word_index) + 1

        self.dict = dict()
        self.train_seqs = self.tokenizer.texts_to_sequences(self.train_text)
        self.train_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.train_seqs]

        self.test_seqs = self.tokenizer.texts_to_sequences(self.test_text)
        self.test_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.test_seqs]

        self.dict['UNK'] = max_vocab_size
        self.inv_dict = dict()
        self.inv_dict[max_vocab_size] = 'UNK'
        self.full_dict = dict()
        self.inv_full_dict = dict()
        for word, idx in self.tokenizer.word_index.items():
            if idx < max_vocab_size:
                self.inv_dict[idx] = word
                self.dict[word] = idx
            self.full_dict[word] = idx
            self.inv_full_dict[idx] = word
        print('Dataset built !')

    def save(self, path='imdb'):
        with open(path + '_train_set.pickle', 'wb') as f:
            pickle.dump((self.train_text, self.train_seqs, self.train_y), f)

        with open(path + '_test_set.pickle', 'wb') as f:
            pickle.dump((self.test_text, self.test_seqs, self.test_y), f)

        with open(path + '_dictionary.pickle', 'wb') as f:
            pickle.dump((self.dict, self.inv_dict), f)

    def _read_vocab(self):
        with open(self.vocab_path, 'r', encoding="utf8") as f:
            vocab_words = f.read().split('\n')

            # create 2 dictionaries one with word (string) as key and the reversed with index(integer) as key
            self.vocab = dict([(w, i) for i, w in enumerate(vocab_words)])
            self.reverse_vocab = dict([(i, w) for w, i in self.vocab.items()])


    def build_text(self, text_seq):
        text_words = [self.inv_full_dict[x] for x in text_seq]
        return ' '.join(text_words)
