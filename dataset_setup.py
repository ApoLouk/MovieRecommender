import pandas as pd
import os
import numpy as np


def setup_data(data_path, movies_filename, ratings_filename, number_of_categories):
    df_movies = pd.read_csv(
        os.path.join(data_path, movies_filename))

    df_ratings = pd.read_csv(
        os.path.join(data_path, ratings_filename))
    df_movies = random_categories(df_movies, number_of_categories)
    # print(df_movies['category'].value_counts())
    return df_ratings, df_movies


def random_categories(movies, number_of_categories):
    np.random.seed(42)
    movies["category"] = np.random.randint(0, number_of_categories, size=movies.shape[0])
    movies.category = pd.Categorical(movies.category)
    return movies