import numpy as np
import random

def permutate_list(initial_list, permutation_list, noise_factor=0.05):
    value = random.uniform(0, 1)
    if value < noise_factor:
        change = np.random.choice(initial_list.shape[0])
        new_list = np.array(initial_list, copy=True)
        new_list[change] = permutation_list
        return new_list
    else:
        return initial_list

