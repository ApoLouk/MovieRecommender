import numpy as np
import pandas as pd
import os
import numpy as np
import random
from numpy import linalg as LA

from CollabFilter import filter_data, predict_user_item, new_id, get_recommendation_list, similarity
from dataset_setup import setup_data
from Permutation_Functions import permutate_list
from CostCalc import calculate_original_score, cost_calculation
from Diversity_Coverage import calculate_diversity, calculate_coverage

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


    # possible_changes = np.arange(20)

    # new_list = permutate_list(baseline_list, possible_changes, noise_factor=0.95)

    # cost = cost_calculation(baseline_list, new_list, final_rankings, baseline_score, LIST_LENGHT)

    # Coverage calculation
    # coverage_dict = calculate_coverage(baseline_list, df_movies)

    # Calculating Dissimilarity
    ratings = np.zeros((n_users, n_items))
    for row in df_ratings_filtered.itertuples():
        ratings[row.newUserId - 1, row.newMovieId - 1] = row[3]
    user_dissimilarity = 1 - similarity(ratings, kind='user')
    # Diversity calculation
    # diversity_dict = calculate_diversity(baseline_list, df_movies, KC, user_dissimilarity)
    w, v = LA.eig(user_dissimilarity)
