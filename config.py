"""
Configuration file for Movie Recommendation System
Replace MODEL_FILE_ID with your actual Google Drive file ID
"""

# Google Drive File IDs
MODEL_FILE_ID = "1sLfM3eGW3Jp-a-qE6xkdqD_eJeIIB1ud"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

# MovieLens Dataset URLs
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

# Local paths
DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/final_model.npz"
MOVIES_CSV_PATH = f"{DATA_DIR}/movies.csv"
RATINGS_CSV_PATH = f"{DATA_DIR}/ratings.csv"

# Model hyperparameters (default for training)
DEFAULT_K = 15
DEFAULT_LAMBDA = 0.1
DEFAULT_GAMMA = 0.5
DEFAULT_EPOCHS = 20
DEFAULT_TEST_RATIO = 0.2
DEFAULT_SEED = 42

# Api key for The Movie Database (TMDb)
TMDB_API_KEY = "3789a3bf192094d1620d9da075649dde"