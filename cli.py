#!/usr/bin/env python3
"""
Movie Recommendation System CLI
Main entry point for the application
"""

import os
import sys
import zipfile

import gdown # type: ignore
import numpy as np
import pandas as pd # type: ignore

# Import configuration
import config
from utils.data_structure import CompactDatasetCSR

# Import utility modules
from utils.display import (
    get_menu_choice,
    get_rating_input,
    print_error,
    print_header,
    print_info,
    print_movie_recommendations,
    print_search_results,
    print_section_header,
    print_success,
)
from utils.dummy_user import DummyUser, predict
from utils.helper_functions import movie_id_to_idx, search_movies
from utils.training import load_data, train_als


def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)


def download_model():
    """Download pre-trained model from Google Drive"""
    print_info("Downloading pre-trained model from Google Drive...")
    print_info("This may take a few minutes (105MB)...")

    try:
        gdown.download(config.MODEL_URL, config.MODEL_PATH, quiet=False)
        print_success("Model downloaded successfully!")
        return True
    except Exception as e:
        print_error(f"Failed to download model: {e}")
        print_info("Please check your internet connection and try again.")
        return False


def download_movies_data():
    """Download MovieLens dataset if not present"""
    if os.path.exists(config.MOVIES_CSV_PATH):
        return True

    print_info("Downloading MovieLens dataset...")

    try:
        # Download zip file
        zip_path = os.path.join(config.DATA_DIR, "ml-25m.zip")

        import urllib.request

        urllib.request.urlretrieve(config.MOVIELENS_URL, zip_path)

        # Extract
        print_info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(config.DATA_DIR)

        # Move files to data directory
        ml_dir = os.path.join(config.DATA_DIR, "ml-25m")
        if os.path.exists(ml_dir):
            for file in ["movies.csv", "ratings.csv", "links.csv"]:
                src = os.path.join(ml_dir, file)
                dst = os.path.join(config.DATA_DIR, file)
                if os.path.exists(src) and not os.path.exists(dst):
                    os.rename(src, dst)

        # Cleanup
        os.remove(zip_path)
        print_success("MovieLens dataset downloaded successfully!")
        return True

    except Exception as e:
        print_error(f"Failed to download dataset: {e}")
        return False


def load_pretrained_model():
    """Load pre-trained model from file"""
    if not os.path.exists(config.MODEL_PATH):
        if not download_model():
            return None

    print_info("Loading pre-trained model...")
    try:
        data = np.load(config.MODEL_PATH, allow_pickle=True)
        model_data = {
            "v": data["v"],
            "item_biases": data["item_biases"],
            "f": data["f"],
            "k": int(data["k"]),
            "lamda": float(data["lamda"]),
            "gamma": float(data["gamma"]),
            "dataset": data["dataset"].item(),
        }
        print_success("Model loaded successfully!")
        return model_data
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return None


def load_movies_df():
    """Load movies DataFrame"""
    if not os.path.exists(config.MOVIES_CSV_PATH):
        if not download_movies_data():
            return None

    try:
        movies_df = pd.read_csv(config.MOVIES_CSV_PATH)
        return movies_df
    except Exception as e:
        print_error(f"Failed to load movies data: {e}")
        return None


def custom_recommendation_flow(model_data, movies_df, train_dataset):
    """Custom movie input flow with search"""
    print_section_header("🎬 CUSTOM RECOMMENDATION")

    num_movies = get_menu_choice(
        "How many movies would you like to rate? (1-10): ", list(range(1, 11))
    )

    rated_movies = []

    for i in range(num_movies):
        print(f"\n{'─' * 60}")
        print(f"  Movie {i + 1}/{num_movies}")
        print("─" * 60)

        while True:
            query = input("\nEnter movie title to search (or 'q' to cancel): ").strip()

            if query.lower() == "q":
                if len(rated_movies) > 0:
                    print_info(
                        f"Using {len(rated_movies)} movie(s) you've already rated."
                    )
                    break
                else:
                    print_info("No movies rated. Returning to menu.")
                    return

            # Search for movies
            results = search_movies(movies_df, query, limit=10)

            if not print_search_results(results):
                continue

            # Get user selection
            choice = get_menu_choice(
                f"Select movie (1-{len(results)}) or 0 to search again: ",
                list(range(0, len(results) + 1)),
            )

            if choice == 0:
                continue

            # Get the selected movie
            selected_movie = results.iloc[choice - 1]
            movie_id = selected_movie["movieId"]

            # Check if movie exists in dataset
            try:
                idx = movie_id_to_idx(model_data["dataset"], [movie_id])[0]
            except KeyError:
                print_error(
                    f"Movie '{selected_movie['title']}' not found in training data. Try another."
                )
                continue

            # Get rating
            rating = get_rating_input()

            rated_movies.append((idx, rating))
            print_success(f"Added: {selected_movie['title']} - Rating: {rating}⭐")
            break

    if len(rated_movies) == 0:
        print_info("No movies rated. Returning to menu.")
        return

    # Generate recommendations
    print("\n" + "═" * 60)
    print_info(
        f"Generating personalized recommendations based on {len(rated_movies)} movie(s)..."
    )

    user = DummyUser(rated_movies, model_data["k"])
    recommendations = predict(user, model_data, train_dataset, num_recommendations=20)

    # Display recommendations
    print_movie_recommendations(movies_df, recommendations)

    # Ask if user wants more
    print("\nWhat would you like to do?")
    print("1. Get 20 more recommendations")
    print("2. Start over with new movies")
    print("3. Return to main menu")

    choice = get_menu_choice("Your choice: ", [1, 2, 3])

    if choice == 1:
        recommendations = predict(
            user, model_data, train_dataset, num_recommendations=40
        )
        print_movie_recommendations(movies_df, recommendations[20:])
        input("\nPress Enter to continue...")
    elif choice == 2:
        custom_recommendation_flow(model_data, movies_df, train_dataset)


def use_pretrained_model():
    """Main flow for using pre-trained model"""
    ensure_directories()

    # Load model
    model_data = load_pretrained_model()
    if model_data is None:
        input("\nPress Enter to return to main menu...")
        return

    # Load movies database
    movies_df = load_movies_df()
    if movies_df is None:
        input("\nPress Enter to return to main menu...")
        return

    # Load training dataset for filtering
    print_info("Loading training dataset for recommendation filtering...")
    if not os.path.exists(config.RATINGS_CSV_PATH):
        if not download_movies_data():
            input("\nPress Enter to return to main menu...")
            return

    # We need to load the training data for filtering low-rated movies
    from utils.data_structure import CompactDatasetCSR

    train_dataset = CompactDatasetCSR(
        shared_index=model_data["dataset"].get_shared_index()
    )

    # Quick load without split for filtering purposes
    print_info("Preparing recommendation system...")
    with open(config.RATINGS_CSV_PATH, "r") as file:
        next(file)
        for line in file:
            userId, movieId, rating, _ = line.strip().split(",")
            train_dataset.add_rating(userId, movieId, float(rating))
    train_dataset.finalize()

    print_success("✅ System ready!")

    # Interactive recommendation loop
    while True:
        print_section_header("RECOMMENDATION OPTIONS")
        print("What would you like to do?")
        print("1. Get custom recommendations (search and rate movies)")
        print("2. Return to main menu")

        choice = get_menu_choice("\nYour choice: ", [1, 2])

        if choice == 1:
            custom_recommendation_flow(model_data, movies_df, train_dataset)
        elif choice == 2:
            break


def train_from_scratch():
    """Train model from scratch"""
    print_section_header("TRAIN FROM SCRATCH")

    ensure_directories()

    # Download data if needed
    if not os.path.exists(config.RATINGS_CSV_PATH):
        if not download_movies_data():
            input("\nPress Enter to return to main menu...")
            return

    print("\nUsing default hyperparameters:")
    print(f"  k (latent dimensions): {config.DEFAULT_K}")
    print(f"  lambda: {config.DEFAULT_LAMBDA}")
    print(f"  gamma: {config.DEFAULT_GAMMA}")
    print(f"  epochs: {config.DEFAULT_EPOCHS}")
    print(f"  test_ratio: {config.DEFAULT_TEST_RATIO}")

    proceed = input("\nProceed with training? (y/n): ").strip().lower()
    if proceed != "y":
        return

    # Load data
    print_info("Loading and splitting dataset...")
    train, test = load_data(
        config.RATINGS_CSV_PATH,
        test_ratio=config.DEFAULT_TEST_RATIO,
        seed=config.DEFAULT_SEED,
    )

    # Add genre features
    print_info("Loading genre features...")
    train.add_genres_features(config.MOVIES_CSV_PATH)
    test.add_genres_features(config.MOVIES_CSV_PATH)

    # Train
    print_section_header("TRAINING MODEL")
    user_biases, item_biases, u, v, f = train_als(
        train,
        test,
        k=config.DEFAULT_K,
        lamda=config.DEFAULT_LAMBDA,
        gamma=config.DEFAULT_GAMMA,
        epochs=config.DEFAULT_EPOCHS,
    )

    # Save model
    print_info("Saving model...")
    np.savez(
        config.MODEL_PATH,
        user_biases=user_biases,
        item_biases=item_biases,
        u=u,
        v=v,
        f=f,
        k=config.DEFAULT_K,
        lamda=config.DEFAULT_LAMBDA,
        gamma=config.DEFAULT_GAMMA,
        dataset=test, # pyright: ignore[reportArgumentType]
    )

    print_success(f"Model saved to {config.MODEL_PATH}")
    print_info("You can now use this model for recommendations!")

    input("\nPress Enter to return to main menu for inference...")


def main_menu():
    """Display main menu and get user choice"""
    print_section_header("MAIN MENU")
    print("Choose an option:")
    print("1. Use pre-trained model (quick start)")
    print("2. Train model from scratch")
    print("3. Exit")

    return get_menu_choice("\nYour choice: ", [1, 2, 3])


def main():
    """Main application entry point"""
    print_header()

    # Check if gdown is installed
    try:
        import gdown # pyright: ignore[reportMissingImports]
    except ImportError:
        print_error("Required package 'gdown' not found.")
        print_info("Please install it with: pip install gdown")
        sys.exit(1)

    while True:
        choice = main_menu()

        if choice == 1:
            use_pretrained_model()
        elif choice == 2:
            train_from_scratch()
        elif choice == 3:
            print("\n" + "=" * 60)
            print(" " * 20 + "👋 Thank you for using")
            print(" " * 15 + "Movie Recommendation System!")
            print("=" * 60 + "\n")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
