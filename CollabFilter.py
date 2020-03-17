import pandas as pd
import os
import numpy as np

from dataset_setup import setup_data

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test

def new_id(x, unique_id):
    new = np.where(unique_id == x)
    return new[0][0]

def slow_similarity(ratings, kind='user'):
    if kind == 'user':
        axmax = 1
        axmin = 0
    else:
        axmax = 0
        axmin = 1
    sim = np.zeros((ratings.shape[axmax], ratings.shape[axmax]))
    # print(ratings.shape[axmax])
    for u in range(ratings.shape[axmax]):
        for uprime in range(ratings.shape[axmax]):
            rui_sqrd = 0.
            ruprimei_sqrd = 0.
            for i in range(ratings.shape[axmin]):
                sim[u, uprime] = ratings[u, i] * ratings[uprime, i]
                rui_sqrd += ratings[u, i] ** 2
                ruprimei_sqrd += ratings[uprime, i] ** 2
            sim[u, uprime] /= rui_sqrd * ruprimei_sqrd
    return sim

def similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    else:
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


def filter_data(df_ratings, movie_rating_thres, user_rating_thres):
    # filter data
    df_movies_cnt = pd.DataFrame(
        df_ratings.groupby('movieId').size(),
        columns=['count'])
    popular_movies = list(set(df_movies_cnt.query('count >= @movie_rating_thres').index))  # noqa
    movies_filter = df_ratings.movieId.isin(popular_movies).values

    df_users_cnt = pd.DataFrame(
        df_ratings.groupby('userId').size(),
        columns=['count'])
    active_users = list(set(df_users_cnt.query('count >= @user_rating_thres').index))  # noqa
    users_filter = df_ratings.userId.isin(active_users).values

    df_ratings_filtered = df_ratings[movies_filter & users_filter].copy(deep=True)
    return df_ratings_filtered

def predict_user_item(df, n_users, n_items):

    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row.newUserId - 1, row.newMovieId - 1] = row[3]
    # print(ratings)

    item_similarity = similarity(ratings, kind='item')
    # print(item_similarity[:4, :4])

    item_prediction = predict_fast_simple(ratings, item_similarity, kind='item')
    minimum = item_prediction.min(axis=1, keepdims=True)
    initialSpan = item_prediction.max(axis=1, keepdims=True) - minimum
    finalSpan = 5

    valueScaled = (item_prediction - minimum) / initialSpan
    final_rankings = valueScaled * finalSpan
    return item_prediction, final_rankings



def get_recommendation_list(list_lenght, best_movies_index, df_ratings_filtered):
    total_recommendations = np.zeros(list_lenght, dtype=int)

    user=0
    for cell in best_movies_index:
        recommendation_list = []
        user += 1
        for movie in reversed(cell):
            if df_ratings_filtered[(df_ratings_filtered['newUserId'] == user)
                                       & (df_ratings_filtered['newMovieId'] == movie)].empty:
                if len(recommendation_list) < list_lenght:
                    recommendation_list.append(movie)
                else:
                    break
        total_recommendations = np.vstack((total_recommendations, np.array(recommendation_list)))


    return total_recommendations[1:]




if __name__ == '__main__':
    LIST_LENGHT = 5
    data_path = os.getcwd() + '/ml-latest-small/'
    # configure file path
    movies_filename = 'movies.csv'
    ratings_filename = 'ratings.csv'# read data

    #Data Setup
    df_ratings, df_movies = setup_data(data_path, movies_filename, ratings_filename)

    #Cleaning Dataset

    df_ratings_filtered = filter_data(df_ratings, 5, 5)

    #Calculate unique users and items
    n_users = df_ratings_filtered.userId.unique().shape[0]
    n_items = df_ratings_filtered.movieId.unique().shape[0]
    print(str(n_users) + ' users')
    print(str(n_items) + ' items')

    #Setup new id [0, n_items] because movieIds have blank spaces
    unique_movie_id = df_ratings_filtered.movieId.unique()
    df_ratings_filtered['newId'] = df_ratings_filtered['movieId'].apply(new_id, args=(unique_movie_id,))


    #Calculate user-item ratings matrix R

    item_prediction, final_rankings = predict_user_item(df_ratings_filtered, n_users, n_items)
    best_movies_index = np.argsort(item_prediction, axis=1)


    baseline_list = get_recommendation_list(LIST_LENGHT, best_movies_index, df_ratings_filtered)
