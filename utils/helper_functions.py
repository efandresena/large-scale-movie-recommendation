"""
Helper functions for movie ID/index conversions and filtering
"""
import numpy as np
from typing import List
from utils.data_structure import CompactDatasetCSR


def index_to_movie_id(dataset: CompactDatasetCSR, indexes: List[int]) -> List[str]:
    """Convert movie indices to movie IDs"""
    ids = []
    for idx in indexes:
        movie_id = dataset.idx_to_movieId[idx]
        ids.append(movie_id)
    return ids


def movie_id_to_idx(dataset: CompactDatasetCSR, IDs: List) -> List[int]:
    """Convert movie IDs to indices"""
    indexes = []
    for movie_id in IDs:
        idx = dataset.movieId_to_idx[str(movie_id)]
        indexes.append(idx)
    return indexes


def check_if_less_rating(train: CompactDatasetCSR, bests: List[int], limit: int = 100) -> List[int]:
    """
    Check which movies have fewer than 'limit' ratings
    Returns list of movie indices to filter out
    """
    idx_to_remove = []
    count = 0
    
    for idx in bests:
        m = idx
        start = train.movie_indptr[m]
        end = train.movie_indptr[m + 1]
        if end - start <= limit:
            idx_to_remove.append(idx)
            count += 1
    
    if count > 0:
        print(f"Filtered {count} movies with fewer than {limit} ratings.")
    
    return idx_to_remove


def get_movie_title_and_genres(movies_df, movie_id: int) -> tuple:
    """Get movie title and genres from DataFrame"""
    try:
        row = movies_df.loc[movies_df['movieId'] == int(movie_id)].iloc[0]
        return row['title'], row['genres']
    except (IndexError, KeyError):
        return f"Movie ID {movie_id}", "Unknown"


def search_movies(movies_df, query: str, limit: int = 10):
    """
    Search for movies by title
    Returns DataFrame of matching movies
    """
    results = movies_df[
        movies_df['title'].str.contains(query, case=False, na=False, regex=False)
    ].head(limit)
    
    return results
