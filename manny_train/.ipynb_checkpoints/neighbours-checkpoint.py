"""
Author: Manjinder Singh
"""

import numpy as np
import pickle


def load_glove_embeddings(glove_file):
    print("Loading Glove Embeddings")
    f = open(glove_file, 'r', encoding="utf8")
    embeddings = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        # print(word)
        embedding = np.array([float(val) for val in row[1:]])
        embeddings[word] = embedding
    print("Done.", len(embeddings), " words loaded!")
    return embeddings




def return_all_neighbours(src_word, dist_mat, ret_count=10, threshold=None):
    """
    embeddings is a matrix with (dimensions, max_vocabulary_size)
    """
    neighbours_list = np.argsort(dist_mat[src_word, :])[1:1 + ret_count]
    neighbours_distance = dist_mat[src_word][neighbours_list]
    if neighbours_distance[-1] == 0:
        return [], []
    mask = np.ones_like(neighbours_distance)
    if threshold is not None:
        mask = np.where(neighbours_distance < threshold)
        return neighbours_list[mask], neighbours_distance[mask]
    else:
        return neighbours_list, neighbours_distance
