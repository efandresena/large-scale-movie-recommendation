"""
Display and formatting utilities
"""
from typing import List
import pandas as pd # type: ignore


def print_header():
    """Print application header"""
    print("\n" + "=" * 60)
    print(" " * 15 + "🎬 MOVIE RECOMMENDATION SYSTEM 🎬")
    print("=" * 60 + "\n")


def print_section_header(title: str):
    """Print a section header"""
    print("\n" + "─" * 60)
    print(f"  {title}")
    print("─" * 60 + "\n")


def print_movie_recommendations(movies_df: pd.DataFrame, movie_ids: List[str]):
    """
    Pretty print movie recommendations
    
    Args:
        movies_df: DataFrame containing movie information
        movie_ids: List of movie IDs to display
    """
    print("\n" + "=" * 60)
    print(" " * 18 + "🎯 YOUR RECOMMENDATIONS")
    print("=" * 60 + "\n")
    
    for idx, movie_id in enumerate(movie_ids, start=1):
        try:
            row = movies_df.loc[movies_df['movieId'] == int(movie_id)].iloc[0]
            
            # Tag for top 3 recommendations
            tag = " 🔥 MUST WATCH!" if idx <= 3 else ""
            
            print(f"{idx:2d}. {row['title']}{tag}")
            print(f"    Genres: {row['genres']}")
            
            if idx < len(movie_ids):
                print()
        except (IndexError, KeyError, ValueError):
            print(f"{idx:2d}. Movie ID {movie_id} (Not found in database)")
            print()
    
    print("=" * 60 + "\n")


def print_search_results(results_df: pd.DataFrame):
    """Print search results in a formatted way"""
    if len(results_df) == 0:
        print("❌ No movies found. Try a different search term.\n")
        return False
    
    print("\n📽️  Search Results:\n")
    for idx, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"  {idx}. {row['title']}")
        print(f"     ID: {row['movieId']} | Genres: {row['genres']}")
    
    print()
    return True


def print_progress(message: str, bar_length: int = 40):
    """Print a simple progress indicator"""
    print(f"\n{message}")


def print_success(message: str):
    """Print success message"""
    print(f"✓ {message}")


def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")


def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")


def get_menu_choice(prompt: str, valid_choices: List[int]) -> int:
    """
    Get a valid menu choice from user
    
    Args:
        prompt: Prompt to display
        valid_choices: List of valid integer choices
    
    Returns:
        Valid integer choice
    """
    while True:
        try:
            choice = int(input(prompt))
            if choice in valid_choices:
                return choice
            else:
                print(f"❌ Please enter a number between {min(valid_choices)} and {max(valid_choices)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)


def get_rating_input() -> float:
    """Get a valid rating from user (0.5 to 5.0)"""
    while True:
        try:
            rating = float(input("Your rating (0.5-5.0): "))
            if 0.5 <= rating <= 5.0:
                return rating
            else:
                print("❌ Rating must be between 0.5 and 5.0")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit(0)
