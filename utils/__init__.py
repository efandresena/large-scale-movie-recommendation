"""
Utils package for Movie Recommendation System
"""

from .data_structure import CompactDatasetCSR
from .dummy_user import DummyUser, predict
from .helper_functions import (
    index_to_movie_id,
    movie_id_to_idx,
    check_if_less_rating,
    get_movie_title_and_genres,
    search_movies
)
from .display import (
    print_header,
    print_section_header,
    print_movie_recommendations,
    print_search_results,
    print_success,
    print_error,
    print_info,
    get_menu_choice,
    get_rating_input
)

__all__ = [
    'CompactDatasetCSR',
    'DummyUser',
    'predict',
    'index_to_movie_id',
    'movie_id_to_idx',
    'check_if_less_rating',
    'get_movie_title_and_genres',
    'search_movies',
    'print_header',
    'print_section_header',
    'print_movie_recommendations',
    'print_search_results',
    'print_success',
    'print_error',
    'print_info',
    'get_menu_choice',
    'get_rating_input'
]
