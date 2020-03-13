import pandas as pd
import os


def setup_data(data_path, movies_filename, ratings_filename):
    df_movies = pd.read_csv(
        os.path.join(data_path, movies_filename))

    df_ratings = pd.read_csv(
        os.path.join(data_path, ratings_filename))
    df_movies.genres = df_movies.genres.str.split(pat = "|")
    df_movies.loc[:, 'category_name'] = df_movies.genres.map(lambda x: x[0])
    # list_of_categories = df_movies.category.unique()
    df_movies.category_name = pd.Categorical(df_movies.category_name)
    df_movies['category'] = df_movies.category_name.cat.codes

    return df_ratings, df_movies