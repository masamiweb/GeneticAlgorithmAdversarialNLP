"""
    Author: Manjinder Singh

"""

import numpy as np
import pickle
import os
from keras.preprocessing.text import Tokenizer



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
    def __init__(self, path, max_vocab_size=None):  # constructor
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


def load_glove_model(glove_file):
    print("Loading Glove Model")
    f = open(glove_file, 'r', encoding="utf8")
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        # print(word)
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def save_glove_to_pickle(glove_model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(glove_model, f)


def load_glove_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def create_embeddings_matrix(glove_model, dictionary, full_dictionary, d=300):
    MAX_VOCAB_SIZE = len(dictionary)
    # Matrix size is 300
    embedding_matrix = np.zeros(shape=(d, MAX_VOCAB_SIZE + 1))
    cnt = 0
    unfound = []

    for w, i in dictionary.items():
        if not w in glove_model:
            cnt += 1

            unfound.append(i)
        else:
            embedding_matrix[:, i] = glove_model[w]
    print('Number of not found words = ', cnt)
    return embedding_matrix, unfound


def pick_most_similar_words(src_word, dist_mat, ret_count=10, threshold=None):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    dist_order = np.argsort(dist_mat[src_word, :])[1:1 + ret_count]
    dist_list = dist_mat[src_word][dist_order]
    if dist_list[-1] == 0:
        return [], []
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        return dist_order[mask], dist_list[mask]
    else:
        return dist_order, dist_list


'''
Large movie review database for binary sentiment classification
This is a dataset for binary sentiment classification containing substantially more data than previous 
benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 
There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. 
See the README file contained in the release for more details.
Download from:
https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

See this paper for more information:
https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf
'''



