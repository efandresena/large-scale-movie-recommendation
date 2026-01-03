"""
Compressed Sparse Row (CSR) data structure for efficient matrix operations
"""
import numpy as np
import csv


class CompactDatasetCSR:
    """CSR (Compressed Sparse Row) format for efficient matrix operations"""

    def __init__(self, shared_index=None):
        if shared_index is not None:
            self.userId_to_idx, self.idx_to_userId, self.movieId_to_idx, self.idx_to_movieId = shared_index
            self._owns_index = False
        else:
            self.userId_to_idx = {}
            self.idx_to_userId = []
            self.movieId_to_idx = {}
            self.idx_to_movieId = []
            self._owns_index = True

        self._temp_ratings = []
        self.user_indptr = None
        self.user_movie_ids = None
        self.user_ratings = None
        self.movie_indptr = None
        self.movie_user_ids = None
        self.movie_ratings = None
        self._finalized = False

    @property
    def usr_size(self):
        return len(self.idx_to_userId)

    @property
    def movie_size(self):
        return len(self.idx_to_movieId)

    def get_shared_index(self):
        return (self.userId_to_idx, self.idx_to_userId,
                self.movieId_to_idx, self.idx_to_movieId)

    def add_rating(self, userId, movieId, rating_value):
        if self._finalized:
            raise RuntimeError("Cannot add ratings after finalization")

        if self._owns_index:
            if userId not in self.userId_to_idx:
                self.userId_to_idx[userId] = len(self.idx_to_userId)
                self.idx_to_userId.append(userId)
            if movieId not in self.movieId_to_idx:
                self.movieId_to_idx[movieId] = len(self.idx_to_movieId)
                self.idx_to_movieId.append(movieId)

        user_pos = self.userId_to_idx.get(userId)
        movie_pos = self.movieId_to_idx.get(movieId)

        if user_pos is not None and movie_pos is not None:
            self._temp_ratings.append((user_pos, movie_pos, rating_value))

    def add_genres_features(self, filepath):
        """Build genre features in CSR format"""
        self.genre2id = {}
        movie_idx_to_genres = {}

        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                movieId, _, genres = row
                if movieId not in self.movieId_to_idx:
                    continue

                movie_idx = self.movieId_to_idx[movieId]
                genre_names = genres.strip().split('|')
                genre_ids = []

                for g in genre_names:
                    if g not in self.genre2id:
                        self.genre2id[g] = len(self.genre2id)
                    genre_ids.append(self.genre2id[g])

                movie_idx_to_genres[movie_idx] = genre_ids

        # Build CSR format for genres
        N = self.movie_size
        self.movie_genre_indptr = np.zeros(N + 1, dtype=np.int64)
        genre_list = []

        for movie_idx in range(N):
            genres = movie_idx_to_genres.get(movie_idx, [])
            self.movie_genre_indptr[movie_idx + 1] = self.movie_genre_indptr[movie_idx] + len(genres)
            genre_list.extend(genres)

        self.movie_genre_ids = np.array(genre_list, dtype=np.int32)
        self.num_genres = len(self.genre2id)

    def finalize(self):
        if self._finalized:
            return

        print(f"Finalizing dataset ({len(self._temp_ratings)} ratings)...")
        M, N = self.usr_size, self.movie_size

        # Build user CSR
        self._temp_ratings.sort(key=lambda x: (x[0], x[1]))
        self.user_indptr = np.zeros(M + 1, dtype=np.int64)
        self.user_movie_ids = np.zeros(len(self._temp_ratings), dtype=np.int32)
        self.user_ratings = np.zeros(len(self._temp_ratings), dtype=np.float32)

        current_user = -1
        for idx, (user_idx, movie_idx, rating) in enumerate(self._temp_ratings):
            while current_user < user_idx:
                current_user += 1
                self.user_indptr[current_user] = idx
            self.user_movie_ids[idx] = movie_idx
            self.user_ratings[idx] = rating
        self.user_indptr[M] = len(self._temp_ratings)

        # Build movie CSR
        self._temp_ratings.sort(key=lambda x: (x[1], x[0]))
        self.movie_indptr = np.zeros(N + 1, dtype=np.int64)
        self.movie_user_ids = np.zeros(len(self._temp_ratings), dtype=np.int32)
        self.movie_ratings = np.zeros(len(self._temp_ratings), dtype=np.float32)

        current_movie = -1
        for idx, (user_idx, movie_idx, rating) in enumerate(self._temp_ratings):
            while current_movie < movie_idx:
                current_movie += 1
                self.movie_indptr[current_movie] = idx
            self.movie_user_ids[idx] = user_idx
            self.movie_ratings[idx] = rating
        self.movie_indptr[N] = len(self._temp_ratings)

        self._temp_ratings = None
        self._finalized = True
        print(f"✓ Users={M}, Movies={N}, Ratings={len(self.user_ratings)}")
