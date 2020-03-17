import pandas as pd
import os
import numpy as np

from CollabFilter import filter_data, predict_user_item, new_id, get_recommendation_list, similarity
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
                category_list.append(movie_recom_dict[row.newMovieId])
        #If at least one movie was recommended
        if category_list:
            category_coverage_dict[row.category] = sum(category_list) / category_items
        else:
            category_coverage_dict[row.category] = 0

    print('Mean Coverage', np.mean(np.array(list(category_coverage_dict.values()))))
    print("The coverage of each category was:")
    for category in category_coverage_dict:
        print(category, ": ", category_coverage_dict[category])

    return category_coverage_dict


def calculate_diversity(df, n_users, n_items, recom_array, df_movies, coverage):
    def get_user_dissimilarity(x, dis_arrray):
        return dis_arrray[x[0]][x[1]]

    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row.newUserId - 1, row.newMovieId - 1] = row[3]

    user_dissimilarity = 1 - similarity(ratings, kind='user')

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
                dissimilarity_per_user_duo = np.apply_along_axis(get_user_dissimilarity, 1, user_grid, user_dissimilarity)
                category_list.append(np.sum(dissimilarity_per_user_duo))

        if category_list:
            normalisation_factor = (coverage * category_items * (coverage * category_items - 1))
            category_diversity_dict[row.category] = 2 * sum(category_list) / (category_items * normalisation_factor)
        else:
            category_diversity_dict[row.category] = 0

    print("")
    print('Mean Diversity', np.mean(np.array(list(category_diversity_dict.values()))))
    print("The diversity of each category was:")
    for category in category_diversity_dict:
        print(category, ": ", category_diversity_dict[category])

    return category_diversity_dict

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

    #Coverage calculation
    coverage_dict = calculate_coverage(baseline_list, df_movies)



    # Diversity calculation

    diversity_dict = calculate_diversity(df_ratings_filtered, n_users, n_items, baseline_list, df_movies, KC)

