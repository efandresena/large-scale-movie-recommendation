"""
DummyUser class and prediction functions
"""
import numpy as np
from typing import List, Tuple
from utils.helper_functions import index_to_movie_id, check_if_less_rating


class DummyUser:
    """Represents a new user with their ratings and latent factors"""
    
    def __init__(self, rated_movies: List[Tuple[int, float]], k: int):
        """
        Args:
            rated_movies: list of tuples (movie_idx, rating)
            k: latent dimension
        """
        self.k = k

        # Ratings
        self.movie_idx = [movie_idx for movie_idx, _ in rated_movies]
        self.ratings = {movie_idx: rating for movie_idx, rating in rated_movies}

        # Latent vector and bias
        norm_k = 1.0 / np.sqrt(self.k)
        self.u = norm_k * np.random.normal(scale=0.1, size=(1, k)).astype(np.float32)
        self.bias = 0.0

    def score_for_item(self):
        """
        Predict ratings the user will give to all movies
        Returns: scores array
        """
        scores = self.u.dot(self.v.T) + 0.05 * self.item_biases  # regularize the bias term
        return scores

    def update(self, v, item_biases, lamda=0.1, gamma=0.1):
        """
        Update user's latent factors and bias
        
        Args:
            v: movie_factors array of shape (N, k)
            item_biases: array of shape (N,)
            lamda: regularization parameter
            gamma: regularization parameter
        """
        k = self.k
        self.v = v
        self.item_biases = item_biases
        
        # Get the number of movies the dummy user rated
        omega = len(self.movie_idx)
        if omega == 0:
            return
        
        # Update bias
        numerator = 0.0
        for m_idx in self.movie_idx:
            r = self.ratings[m_idx]
            uv = (self.u @ v[m_idx]).item()
            numerator += r - item_biases[m_idx] - uv
        self.bias = lamda * numerator / (lamda * omega + gamma)

        # Update latent vector
        A = np.zeros((k, k), dtype=np.float32)
        b = np.zeros(k, dtype=np.float32)
        
        for m_idx in self.movie_idx:
            r = self.ratings[m_idx]
            v_m = v[m_idx]
            A += np.outer(v_m, v_m)
            residual = r - self.bias - item_biases[m_idx]
            b += v_m * residual

        A = lamda * A + gamma * np.eye(k, dtype=np.float32)
        b = lamda * b
        self.u = np.linalg.solve(A, b)[None, :]


def predict(user: DummyUser, model_data: dict, train_dataset, num_recommendations: int = 20) -> List[str]:
    """
    Generate movie recommendations for a user
    
    Args:
        user: DummyUser instance
        model_data: Dictionary containing model parameters (v, item_biases, lamda, gamma)
        train_dataset: Training dataset for filtering low-rated movies
        num_recommendations: Number of recommendations to return
    
    Returns:
        List of movie IDs (as strings)
    """
    # Extract model parameters
    v = model_data['v']
    item_biases = model_data['item_biases']
    lamda = model_data['lamda']
    gamma = model_data['gamma']
    test_dataset = model_data['dataset']
    
    # Embed the new user in a latent trait vector
    user.update(v, item_biases, lamda, gamma)
    
    # Get scores for all movies
    scores = user.score_for_item()[0]
    
    # Get top recommendations
    sorted_idx = np.argsort(scores)
    best = sorted_idx[-(num_recommendations * 3):][::-1]  # Get more than needed for filtering
    
    # Filter movies with too few ratings
    less_rating = check_if_less_rating(train_dataset, best, limit=100) # type: ignore
    new_bests = [idx for idx in best if idx not in less_rating][:num_recommendations]
    
    # Convert indices to movie IDs
    movie_ids = index_to_movie_id(test_dataset, new_bests)
    
    return movie_ids
