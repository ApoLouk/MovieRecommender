import pandas as pd
import os
import numpy as np

from CollabFilter import filter_data, predict_user_item, new_movie_id, get_recommendation_list, similarity
from dataset_setup import setup_data

def calculate_coverage(recom_array, df_movies):
    #Find how many times does each move show up in the recommendations
    unique, counts = np.unique(recom_array, return_counts=True)
    movie_recom_dict = dict(zip(unique, counts))

    grouped = df_movies.groupby('category')
    category_coverage_dict = {}
    for group_num, group in grouped:
        # Calculate Ic
        category_items = len(group.index)
        category_list = []
        #for each movie
        for index, row in group.iterrows():
            #Check if it is recommended at all
            if index in movie_recom_dict:
                #If it is add to category list
                category_list.append(movie_recom_dict[row.newId])
        #If at least one movie was recommended
        if category_list:
            category_coverage_dict[row.category_name] = sum(category_list) / category_items
        else:
            category_coverage_dict[row.category_name] = 0


    return category_coverage_dict

def calculate_diversity(df, n_users, n_items, recom_array, df_movies, coverage):
    def get_user_dissimilarity(x, dis_arrray):
        return dis_arrray[x[0]][x[1]]


    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row.userId - 1, row.newId - 1] = row[3]
    print(ratings)

    user_similarity = 1 - similarity(ratings, kind='user')

    grouped = df_movies.groupby('category')
    category_diversity_dict = {}
    for group_num, group in grouped:
        # Calculate Ic
        category_items = len(group.index)
        category_list = []
        # for each movie
        for index, row in group.iterrows():
            users, _ = np.where(recom_array == index)
            if users.size > 0:
                user_grid = np.array(np.meshgrid(users,users)).T.reshape(-1,2)
                dissimilarity_per_user_duo = np.apply_along_axis(get_user_dissimilarity, 1, user_grid, user_similarity)
                category_list.append(np.sum(dissimilarity_per_user_duo))

        if category_list:
            normalisation_factor = (coverage[row.category_name] * category_items * (coverage[row.category_name] * category_items - 1))
            category_diversity_dict[row.category_name] = 2 * sum(category_list) / (category_items * normalisation_factor)
        else:
            category_diversity_dict[row.category_name] = 0


    return category_diversity_dict

if __name__ == '__main__':
    LIST_LENGHT = 10
    data_path = os.getcwd() + '/ml-latest-small/'
    # configure file path
    movies_filename = 'movies.csv'
    ratings_filename = 'ratings.csv'# read data

    # Data Setup
    df_ratings, df_movies = setup_data(data_path, movies_filename, ratings_filename)

    #Cleaning Dataset

    df_ratings_filtered = filter_data(df_ratings, 5, 5)

    n_users = df_ratings_filtered.userId.unique().shape[0]
    n_items = df_ratings_filtered.movieId.unique().shape[0]
    print(str(n_users) + ' users')
    print(str(n_items) + ' items')

    unique_movie_id = df_ratings_filtered.movieId.unique()
    df_ratings_filtered['newId'] = df_ratings_filtered['movieId'].apply(new_movie_id, args=(unique_movie_id,))

    # Setup new id [0, n_items] because movieIds have blank spaces
    df_movies['newId'] = df_movies.index

    #Calculate user-item ratings matrix R

    item_prediction = predict_user_item(df_ratings_filtered, n_users, n_items)
    best_movies_index = np.argsort(item_prediction, axis=1)

    baseline_list = get_recommendation_list(LIST_LENGHT, best_movies_index, df_ratings_filtered)

    #Coverage calculation
    coverage_dict = calculate_coverage(baseline_list, df_movies)

    print("The coverage of each category was:")
    for category in coverage_dict:
        print(category, ": ", coverage_dict[category])

    # Diversity calculation

    diversity_dict = calculate_diversity(df_ratings_filtered, n_users, n_items, baseline_list, df_movies, coverage_dict)

    print("The diversity of each category was:")
    for category in diversity_dict:
        print(category, ": ", diversity_dict[category])
