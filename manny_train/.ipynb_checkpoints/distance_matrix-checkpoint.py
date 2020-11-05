mport numpy as np
import tensorflow as tf
import glove_utils
import pickle 
from keras.preprocessing.sequence import pad_sequences

MAX_VOCAB_SIZE = 134303 # pass this in as a parameter - after testing is done
embedding_matrix = np.load(('embeddings/imdb_counter_fitted_embeddings_%d.npy' %(MAX_VOCAB_SIZE)))

square_matrix = -2*np.dot(embedding_matrix.T , embedding_matrix)
a = np.sum(np.square(embedding_matrix), axis=0).reshape((1,-1))
b = a.T
dist = a + b + square_matrix

np.save(('embeddings/dist_counter_%d.npy' %(MAX_VOCAB_SIZE)), dist)

# Try an example
# dataset - is the dictionary created in the main jupyter notebook
with open('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)
src_word = dataset.dict['good']

neighbours, neighbours_dist = glove_utils.pick_most_similar_words(src_word, dist)
print('Closest words to `good` are :')
result_words = [dataset.inv_dict[x] for x in neighbours]
print(result_words)
