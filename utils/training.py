"""
Training functions for ALS matrix factorization with genre features
"""

import random

import numpy as np
from numba import njit, prange # pyright: ignore[reportMissingImports]

from utils.data_structure import CompactDatasetCSR


def load_data(filepath: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Load dataset with train/test split

    Args:
        filepath: Path to ratings CSV file
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility

    Returns:
        train, test: CompactDatasetCSR objects
    """
    random.seed(seed)
    train = CompactDatasetCSR()
    test = CompactDatasetCSR(shared_index=train.get_shared_index())

    print("Loading data...")
    with open(filepath, "r") as file:
        next(file)  # Skip header
        for line in file:
            userId, movieId, rating, _ = line.strip().split(",")
            rating = float(rating)
            if random.random() < test_ratio:
                test.add_rating(userId, movieId, rating)
            else:
                train.add_rating(userId, movieId, rating)

    train.finalize()
    test.finalize()
    return train, test


@njit(parallel=True, fastmath=True)
def update_user_biases_and_factors(
    user_biases,
    u,
    item_biases,
    v,
    user_indptr,
    user_movie_ids,
    user_ratings,
    lamda,
    gamma,
    k,
):
    """Update user biases and latent factors"""
    M = len(user_biases)
    for m in prange(M):
        start = user_indptr[m]
        end = user_indptr[m + 1]
        omega = end - start

        if omega == 0:
            continue

        # Update user bias
        numerator = 0.0
        for idx in range(start, end):
            n = user_movie_ids[idx]
            r = user_ratings[idx]
            pred_no_bias = 0.0
            for d in range(k):
                pred_no_bias += u[m, d] * v[n, d]
            numerator += r - item_biases[n] - pred_no_bias
        user_biases[m] = lamda * numerator / (lamda * omega + gamma)

        # Update user latent factors
        A = np.zeros((k, k), dtype=np.float32)
        b = np.zeros(k, dtype=np.float32)

        for idx in range(start, end):
            n = user_movie_ids[idx]
            r = user_ratings[idx]
            for i in range(k):
                for j in range(k):
                    A[i, j] += v[n, i] * v[n, j]
            residual = r - user_biases[m] - item_biases[n]
            for i in range(k):
                b[i] += v[n, i] * residual

        for i in range(k):
            for j in range(k):
                A[i, j] *= lamda
            A[i, i] += gamma
            b[i] *= lamda

        u[m, :] = np.linalg.solve(A, b)


@njit(parallel=True, fastmath=True)
def update_item_biases_and_factors(
    item_biases,
    v,
    user_biases,
    u,
    f,
    movie_indptr,
    movie_user_ids,
    movie_ratings,
    movie_genre_indptr,
    movie_genre_ids,
    lamda,
    gamma,
    k,
):
    """Update item biases and latent factors"""
    N = len(item_biases)
    for n in prange(N):
        start = movie_indptr[n]
        end = movie_indptr[n + 1]
        omega = end - start

        if omega == 0:
            continue

        # Update item bias
        numerator = 0.0
        for idx in range(start, end):
            m = movie_user_ids[idx]
            r = movie_ratings[idx]
            pred_no_bias = 0.0
            for d in range(k):
                pred_no_bias += v[n, d] * u[m, d]
            numerator += r - user_biases[m] - pred_no_bias
        item_biases[n] = lamda * numerator / (lamda * omega + gamma)

        # Update item latent factors
        A = np.zeros((k, k), dtype=np.float32)
        b = np.zeros(k, dtype=np.float32)

        for idx in range(start, end):
            m = movie_user_ids[idx]
            r = movie_ratings[idx]
            for i in range(k):
                for j in range(k):
                    A[i, j] += u[m, i] * u[m, j]
            residual = r - user_biases[m] - item_biases[n]
            for i in range(k):
                b[i] += u[m, i] * residual

        for i in range(k):
            for j in range(k):
                A[i, j] *= lamda
            A[i, i] += gamma
            b[i] *= lamda

        # Add genre prior
        g_start = movie_genre_indptr[n]
        g_end = movie_genre_indptr[n + 1]
        F_n = g_end - g_start

        if F_n > 0:
            for idx_g in range(g_start, g_end):
                g = movie_genre_ids[idx_g]
                for i in range(k):
                    b[i] += gamma * f[g, i] / F_n

        v[n, :] = np.linalg.solve(A, b)


@njit(parallel=True, fastmath=True)
def update_genre_factors(f, v, movie_genre_indptr, movie_genre_ids, k):
    """Update genre factors"""
    num_genres = f.shape[0]
    num_movies = v.shape[0]
    f_new = np.zeros_like(f)

    for g in prange(num_genres):
        numerator = np.zeros(k, dtype=np.float32)
        denominator = 0.0

        for n in range(num_movies):
            start = movie_genre_indptr[n]
            end = movie_genre_indptr[n + 1]
            F_n = end - start
            if F_n == 0:
                continue

            sqrt_Fn = np.sqrt(F_n)
            contains_g = False
            sum_other_fk = np.zeros(k, dtype=np.float32)

            for idx in range(start, end):
                genre_id = movie_genre_ids[idx]
                if genre_id == g:
                    contains_g = True
                else:
                    sum_other_fk += f[genre_id, :]

            if contains_g:
                numerator += (v[n, :] - sum_other_fk / sqrt_Fn) / sqrt_Fn
                denominator += 1.0 / F_n

        f_new[g, :] = numerator / (denominator + 1e-10)

    for g in range(num_genres):
        for i in range(k):
            f[g, i] = f_new[g, i]


@njit(parallel=True, fastmath=True)
def compute_metrics_numba(
    user_indptr,
    user_movie_ids,
    user_ratings,
    user_biases,
    item_biases,
    u,
    v,
    f,
    movie_genre_indptr,
    movie_genre_ids,
    gamma,
):
    """Compute loss and RMSE metrics"""
    M = len(user_biases)
    k = u.shape[1]

    sq_errors = np.zeros(M, dtype=np.float64)
    counts = np.zeros(M, dtype=np.int64)

    for m in prange(M):
        start = user_indptr[m]
        end = user_indptr[m + 1]
        local_sq_err = 0.0
        for idx in range(start, end):
            n = user_movie_ids[idx]
            r = user_ratings[idx]
            pred = user_biases[m] + item_biases[n]
            for d in range(k):
                pred += u[m, d] * v[n, d]
            diff = r - pred
            local_sq_err += diff * diff
        sq_errors[m] = local_sq_err
        counts[m] = end - start

    sq_err = np.sum(sq_errors)
    count = np.sum(counts)

    reg_term = gamma * (
        np.sum(user_biases**2) + np.sum(item_biases**2) + np.sum(u**2) + np.sum(v**2)
    )

    # Genre prior term
    N = v.shape[0]
    genre_reg = 0.0
    for n in range(N):
        start_g = movie_genre_indptr[n]
        end_g = movie_genre_indptr[n + 1]
        F_n = end_g - start_g
        if F_n > 0:
            sum_fk = np.zeros(k, dtype=np.float32)
            for idx_g in range(start_g, end_g):
                g = movie_genre_ids[idx_g]
                sum_fk += f[g, :]
            diff_v = v[n, :] - sum_fk / F_n
            genre_reg += np.sum(diff_v**2)

    reg_term += gamma * (genre_reg + np.sum(f**2))
    loss = 0.5 * (sq_err + reg_term)
    rmse = np.sqrt(sq_err / count) if count > 0 else 0.0

    return loss, rmse


def train_als(train, test, k=15, lamda=0.1, gamma=0.5, epochs=20):
    """
    Train ALS model with genre features

    Args:
        train: Training dataset
        test: Test dataset
        k: Latent dimension
        lamda: Regularization parameter
        gamma: Regularization parameter
        epochs: Number of training epochs

    Returns:
        user_biases, item_biases, u, v, f: Model parameters
    """
    M, N = train.usr_size, train.movie_size
    G = train.num_genres

    # Initialize
    norm_f = 1.0 / np.sqrt(k)
    user_biases = np.zeros(M, dtype=np.float32)
    item_biases = np.zeros(N, dtype=np.float32)
    u = norm_f * np.random.randn(M, k).astype(np.float32)
    v = norm_f * np.random.randn(N, k).astype(np.float32)
    f = norm_f * np.random.randn(G, k).astype(np.float32)

    print(f"\nTraining: {M} users, {N} movies, {G} genres, k={k}")

    for epoch in range(epochs):
        update_user_biases_and_factors(
            user_biases,
            u,
            item_biases,
            v,
            train.user_indptr,
            train.user_movie_ids,
            train.user_ratings,
            lamda,
            gamma,
            k,
        )

        update_item_biases_and_factors(
            item_biases,
            v,
            user_biases,
            u,
            f,
            train.movie_indptr,
            train.movie_user_ids,
            train.movie_ratings,
            train.movie_genre_indptr,
            train.movie_genre_ids,
            lamda,
            gamma,
            k,
        )

        update_genre_factors(f, v, train.movie_genre_indptr, train.movie_genre_ids, k)

        loss_tr, rmse_tr = compute_metrics_numba(
            train.user_indptr,
            train.user_movie_ids,
            train.user_ratings,
            user_biases,
            item_biases,
            u,
            v,
            f,
            train.movie_genre_indptr,
            train.movie_genre_ids,
            gamma,
        )

        loss_te, rmse_te = compute_metrics_numba(
            test.user_indptr,
            test.user_movie_ids,
            test.user_ratings,
            user_biases,
            item_biases,
            u,
            v,
            f,
            test.movie_genre_indptr,
            test.movie_genre_ids,
            gamma,
        )

        print(
            f"Epoch {epoch + 1:2d}: Loss(tr)={loss_tr:.4f} Loss(te)={loss_te:.4f} "
            f"RMSE(tr)={rmse_tr:.4f} RMSE(te)={rmse_te:.4f}"
        )

    return user_biases, item_biases, u, v, f
