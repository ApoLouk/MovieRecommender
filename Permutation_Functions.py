import numpy as np


def permutate_list(initial_list, permutation_list, noise_factor=0.05):

    # Generate noise numbers for count number of times.
    # This is where vectorization comes into the play.
    nums = np.random.choice(permutation_list, initial_list.shape)
    mask = np.random.choice([0, 1], size=initial_list.shape, p=[1 - noise_factor, noise_factor])
    new_list = np.array(initial_list, copy=True)
    np.place(new_list, mask == 1, nums)

    return new_list