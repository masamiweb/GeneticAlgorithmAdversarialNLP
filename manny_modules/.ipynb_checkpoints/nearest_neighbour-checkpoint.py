import numpy as np


def closest_neighbours(input_word, distance_matrix, number_of_words_to_return=10, max_distance=None):
    """
    Retrun a list of the nearest neighbours to the input_word
    
    @param input_word
    @param distance_matrix
    @param number_of_words_to_return
    @param max_distance
    """
    # get indices of nearest words sorted in ascending order, start from index 1: so we exclude the source word
    nearest_neighbour_list = np.argsort(distance_matrix[input_word, :])[1:1 + number_of_words_to_return]
    
    # creat list with all the distance measures found
    distance_to_neighbour_list = distance_matrix[input_word][nearest_neighbour_list]
    
    # nothing found
    if distance_to_neighbour_list[-1] == 0:
        return [], []
    
    # creat a matrix with all zeros same size as the distance list created, to use as mask
    mask = np.zeros_like(distance_to_neighbour_list)

    
    # if we've set a max_distance - then we need to exclude all values over max_distance
    if max_distance is None:
        return nearest_neighbour_list, distance_to_neighbour_list
    
    mask = np.where(distance_to_neighbour_list < max_distance)
    return nearest_neighbour_list[mask], distance_to_neighbour_list[mask]
        




