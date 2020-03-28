import pandas as pd
import os
import numpy as np

from CollabFilter import filter_data, predict_user_item, new_id, get_recommendation_list, similarity
from dataset_setup import setup_data
from CostCalc import calculate_original_score, cost_calculation
from Permutation_Functions import permutate_list
from Diversity_Coverage import calculate_coverage, calculate_diversity, get_user_dissimilarity


#
# possible_changes = np.arange(20)
#
# new_list = permutate_list(baseline_list, possible_changes, noise_factor=0.05)
#
# cost = cost_calculation(baseline_list, new_list, final_rankings, baseline_score, LIST_LENGHT)
def coverage_step(initial_list, df_movies, best_movies_index, min_div_category, LIST_LENGHT):
    #TODO turn diversity function to coverage
    possible_changes = []
    #Get movies of worst category
    possible_movies = df_movies.loc[df_movies['category'] == min_div_category, 'newMovieId']
    for movie in possible_movies:
        movies_in_ranking = np.argwhere(best_movies_index == movie)
        possible_changes.append(movies_in_ranking[movies_in_ranking[:, 1].argsort()][:5])

    total_best_movies = np.vstack(possible_changes)
    row_mask = (total_best_movies[:,1] > LIST_LENGHT)
    total_best_movies = total_best_movies[row_mask, :]
    sorted_total_best_movies = total_best_movies[total_best_movies[:, 1].argsort()][:15]
    idx = np.random.choice(len(sorted_total_best_movies), 10)
    possible_changes = sorted_total_best_movies[idx]
    pertubation_lists = []
    for j in range(5):
        new_list = np.array(initial_list, copy=True)
        for i in possible_changes:
            test = permutate_list(initial_list[i[0]], i[1], noise_factor=0.5)
            new_list[i[0]] = test
        pertubation_lists.append(new_list)
    return pertubation_lists

def get_biggest_dissimilarity(sorted_total_best_movies, user_dissimilarity, max_number=10):
    unique_elements, counts_elements = np.unique(sorted_total_best_movies[:,1], return_counts=True)
    unique_elements = np.stack((unique_elements, counts_elements), axis=-1)
    unique_elements = unique_elements[unique_elements[:,1] > 1]
    dissimilar_users = []
    for movie in unique_elements:

        possible_users = sorted_total_best_movies[sorted_total_best_movies[:,1] == movie[0]]
        user_grid = np.array(np.meshgrid(possible_users[:,0], possible_users[:,0])).T.reshape(-1, 2)
        dissimilarity_per_user_duo = np.apply_along_axis(get_user_dissimilarity, 1, user_grid, user_dissimilarity)
        user_grid = user_grid[dissimilarity_per_user_duo > 0]
        new_suggestions = user_grid.ravel()
        test = np.full(new_suggestions.shape, movie[0])
        new_suggestions = np.column_stack((new_suggestions,test))
        new_suggestions = np.unique(new_suggestions, axis=0)
        dissimilar_users.append(new_suggestions)

    final_suggestions = np.vstack(dissimilar_users)

    return final_suggestions

def diversity_step(initial_list, df_movies, best_movies_index, min_div_category, LIST_LENGHT, user_dissimilarity):
    possible_changes = []
    #Get movies of worst category
    possible_movies = df_movies.loc[df_movies['category'] == min_div_category, 'newMovieId']
    for movie in possible_movies:
        movies_in_ranking = np.argwhere(best_movies_index == movie)
        possible_changes.append(movies_in_ranking[movies_in_ranking[:, 1].argsort()][:20])

    total_best_movies = np.vstack(possible_changes)
    row_mask = (total_best_movies[:,1] > LIST_LENGHT)
    total_best_movies = total_best_movies[row_mask, :]
    sorted_total_best_movies = total_best_movies[total_best_movies[:, 1].argsort()]
    sorted_total_best_movies = get_biggest_dissimilarity(sorted_total_best_movies, user_dissimilarity)


    idx = np.random.choice(len(sorted_total_best_movies), 10)
    # possible_changes = sorted_total_best_movies[idx]
    possible_changes = sorted_total_best_movies
    pertubation_lists = []
    for j in range(5):
        new_list = np.array(initial_list, copy=True)
        for i in possible_changes:
            new_list[i[0]] = permutate_list(initial_list[i[0]], i[1], noise_factor=0.5)
        pertubation_lists.append(new_list)
    return pertubation_lists

def heuristic_iteration(initial_list, df_movies, best_movies_index, user_dissimilarity, KC, D, diversity_dict, coverage_dict, rankings, LIST_LENGHT):
    min_div_category = min(diversity_dict, key=diversity_dict.get)

    min_cov_category = min(coverage_dict, key=coverage_dict.get)
    min_div_value = diversity_dict[min_div_category]
    min_cov_value = coverage_dict[min_cov_category]

    if diversity_dict[min_div_category] >= D and coverage_dict[min_cov_category] >= KC:
        print("Here we minimize cost")
    elif diversity_dict[min_div_category] >= D and coverage_dict[min_cov_category] < KC:
        print("Here we cover the coverage contraint")
    else:
        pertubation_lists = diversity_step(initial_list, df_movies, best_movies_index, min_div_category, LIST_LENGHT, user_dissimilarity)
    results = {}

    for index, pertubation in enumerate(pertubation_lists):
        results[index] = {}
        results[index]['coverage'] = calculate_coverage(pertubation, df_movies)
        results[index]['diversity'] = calculate_diversity(pertubation, df_movies, KC,
                                             user_dissimilarity)
        intermediate_key1 = min(results[index]['coverage'], key=results[index]['coverage'].get)
        intermediate_key2 = min(results[index]['diversity'], key=results[index]['diversity'].get)
        results[index]['min_diversity'] = results[index]['diversity'][intermediate_key2]
        results[index]['min_coverage'] = results[index]['coverage'][intermediate_key1]


    return

def contrained_heuristic(baseline_list, final_rankings, df_movies, best_movies_index, kc, d, list_lenght, itterations):

    # Calculate score of the baseline list
    baseline_score = calculate_original_score(baseline_list, final_rankings, list_lenght)
    # Coverage calculation
    coverage_dict = calculate_coverage(baseline_list, df_movies)

    # Calculating Dissimilarity
    ratings = np.zeros((n_users, n_items))
    for row in df_ratings_filtered.itertuples():
        ratings[row.newUserId - 1, row.newMovieId - 1] = row[3]
    user_dissimilarity = 1 - similarity(ratings, kind='user')
    # Diversity calculation
    diversity_dict = calculate_diversity(baseline_list, df_movies, kc,
                                         user_dissimilarity)

    initial_list = np.array(baseline_list, copy=True)
    for itteration in range(itterations):
        new_list = heuristic_iteration(initial_list, df_movies, best_movies_index, user_dissimilarity, kc, d, diversity_dict, coverage_dict, final_rankings, LIST_LENGHT)



    return



if __name__ == '__main__':
    LIST_LENGHT = 5
    KC = 0.15
    D = 0.1
    NUMBER_OF_CATEGORIES = 100

    OPTIMIZATION_ITERATIONS = 10


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

    #Get initial recommendation list
    baseline_list = get_recommendation_list(LIST_LENGHT, best_movies_index, df_ratings_filtered)


    contrained_heuristic(baseline_list, final_rankings, df_movies, best_movies_index, KC, D, LIST_LENGHT, OPTIMIZATION_ITERATIONS)


