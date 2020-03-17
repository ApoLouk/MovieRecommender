import pandas as pd
import os
import numpy as np
import random

from CollabFilter import filter_data, predict_user_item, new_id, get_recommendation_list, similarity
from dataset_setup import setup_data
from Permutation_Functions import permutate_list


def cost_calculation(original_list, new_list, predictions, baseline_score, list_lenght):
    sum_changes = 0

    result = np.subtract(original_list, new_list)
    index_of_changes = np.argwhere(result != 0)
    cost_changes = []
    for change in index_of_changes:
        sum_changes += predictions[change[0]][original_list[change[0]][change[1]]] - predictions[change[0]][new_list[change[0]][change[1]]]

    cost = baseline_score - (sum_changes / (list_lenght * predictions.shape[0]))
    return cost

def calculate_original_score(original_list, predictions, list_lenght):
    original_score = 0
    i=0
    j=0
    for x in np.nditer(original_list):
        original_score += predictions[i][x]
        if j < list_lenght - 1:
            j += 1
        else:
            i += 1
            j = 0
    original_score = original_score / (list_lenght * predictions.shape[0])
    return original_score

if __name__ == '__main__':
    LIST_LENGHT = 5
    KC = 0.15
    NUMBER_OF_CATEGORIES = 10

    data_path = os.getcwd() + '/ml-latest-small/'
    # configure file path
    movies_filename = 'movies.csv'
    ratings_filename = 'ratings.csv'# read data

    # Data Setup
    df_ratings, df_movies = setup_data(data_path, movies_filename, ratings_filename, NUMBER_OF_CATEGORIES)

    #Cleaning Dataset

    df_ratings_filtered = filter_data(df_ratings, 50, 50)

    n_users = df_ratings_filtered.userId.unique().shape[0]
    n_items = df_ratings_filtered.movieId.unique().shape[0]
    print(str(n_users) + ' users')
    print(str(n_items) + ' items')

    unique_movie_id = df_ratings_filtered.movieId.unique()
    unique_user_id = df_ratings_filtered.userId.unique()
    df_ratings_filtered['newMovieId'] = df_ratings_filtered['movieId'].apply(new_id, args=(unique_movie_id,))

    df_ratings_filtered['newUserId'] = df_ratings_filtered['userId'].apply(new_id, args=(unique_user_id,))

    # Setup new id [0, n_items] because movieIds have blank spaces
    df_movies['newMovieId'] = df_movies.index

    #Calculate user-item ratings matrix R

    item_prediction, final_rankings = predict_user_item(df_ratings_filtered, n_users, n_items)
    best_movies_index = np.argsort(item_prediction, axis=1)

    baseline_list = get_recommendation_list(LIST_LENGHT, best_movies_index, df_ratings_filtered)

    baseline_score = calculate_original_score(baseline_list, final_rankings, LIST_LENGHT)


    possible_changes = np.arange(20)

    new_list = permutate_list(baseline_list, possible_changes, noise_factor=0.05)

    cost = cost_calculation(baseline_list, new_list, final_rankings, baseline_score, LIST_LENGHT)