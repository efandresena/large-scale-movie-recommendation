import os
import random
import time
from typing import List

import numpy as np
import pandas as pd  # type: ignore
import requests
import streamlit as st  # type: ignore

import config
from utils.data_structure import CompactDatasetCSR
from utils.dummy_user import DummyUser, predict
from utils.helper_functions import (
    get_movie_title_and_genres,
    index_to_movie_id,  # noqa: F401
    movie_id_to_idx,
    search_movies,
)


@st.cache_resource
def load_model():
    """Load model once and cache it"""
    try:
        import cli as cli_module
        model_data = cli_module.load_pretrained_model()
        if model_data is not None:
            return model_data
    except Exception:
        pass

    # Fallback: load directly from file
    data = np.load(config.MODEL_PATH, allow_pickle=True)
    model_data = {
        "v": data["v"],
        "item_biases": data["item_biases"],
        "f": data.get("f", None),
        "k": int(data["k"]),
        "lamda": float(data["lamda"]),
        "gamma": float(data["gamma"]),
        "dataset": data["dataset"].item(),
    }
    return model_data


@st.cache_resource
def load_train_dataset_for_filtering():
    """Load only the movie rating counts for filtering - much lighter than full dataset"""
    print("DEBUG: Loading movie rating counts from ratings.csv...")
    
    # Count ratings per movie
    from collections import defaultdict
    movie_rating_counts = defaultdict(int)
    
    with open(config.RATINGS_CSV_PATH, "r") as file:
        next(file)  # Skip header
        for line in file:
            _, movieId, _, _ = line.strip().split(",")
            movie_rating_counts[movieId] += 1
    
    print(f"DEBUG: Loaded rating counts for {len(movie_rating_counts)} movies")
    return dict(movie_rating_counts)


@st.cache_resource
def load_data():
    """Ensure data is downloaded once"""
    try:
        import cli as cli_module 
        cli_module.download_movies_data()
    except Exception:
        pass


@st.cache_data
def load_movies_df(path: str = config.MOVIES_CSV_PATH):
    return pd.read_csv(path)


@st.cache_data
def load_links_df(path: str = os.path.join(config.DATA_DIR, "links.csv")):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])  # type: ignore
    return pd.read_csv(path, dtype={"movieId": str, "imdbId": str, "tmdbId": str})


@st.cache_data
def tmdb_get_poster_url(tmdb_id: str, tmdb_api_key: str):
    if not tmdb_api_key or not tmdb_id or pd.isna(tmdb_id) or tmdb_id == "nan":
        return None
    try:
        url = (
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={tmdb_api_key}"
        )
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        j = r.json()
        poster_path = j.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w300{poster_path}"
    except Exception:
        return None
    return None


@st.cache_data
def get_poster_for_movieid(movie_id: str, links_df: pd.DataFrame, tmdb_key: str = None):  # type: ignore
    row = links_df.loc[links_df["movieId"] == str(movie_id)]
    if row.empty:
        return None
    tmdb_id = str(row.iloc[0].get("tmdbId", ""))

    p = None
    if tmdb_key:
        p = tmdb_get_poster_url(tmdb_id, tmdb_key)
    return p


def poster_placeholder(width: int = 180, height: int = 270) -> str:
    """Generate a consistent placeholder image as data URI"""
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
        <rect width='100%' height='100%' fill='#e8e8e8'/>
        <rect x='50' y='100' width='80' height='100' rx='5' fill='#d0d0d0'/>
        <circle cx='90' cy='140' r='15' fill='#c0c0c0'/>
        <polygon points='75,165 90,150 105,165' fill='#c0c0c0'/>
        <text x='50%' y='85%' dominant-baseline='middle' text-anchor='middle' fill='#999' font-size='12' font-family='Arial'>
            No Poster
        </text>
    </svg>"""
    return "data:image/svg+xml;utf8," + svg.replace("\n", "").replace("  ", "")


def render_movie_card(
    col, idx, movie_id, movies_df, links_df, tmdb_key, dataset, key_prefix
):
    """Render a single movie card with fixed dimensions"""
    with col:
        # Get poster with fallback
        poster = get_poster_for_movieid(movie_id, links_df, tmdb_key)

        # Get title and genres
        title, genres = get_movie_title_and_genres(movies_df, int(movie_id))

        # Truncate title if too long
        display_title = title if len(title) <= 40 else title[:37] + "..."
        display_genres = genres if len(genres) <= 50 else genres[:47] + "..."

        # Render poster
        if poster:
            st.markdown(
                f'<div class="poster-container"><img src="{poster}" alt="{display_title}"></div>',
                unsafe_allow_html=True,
            )
        else:
            placeholder = poster_placeholder()
            st.markdown(
                f'<div class="poster-container"><img src="{placeholder}" alt="No poster"></div>',
                unsafe_allow_html=True,
            )

        # Title and genres
        st.markdown(
            f'<div class="movie-title">{display_title}</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="movie-genres">{display_genres}</div>', unsafe_allow_html=True
        )

        # Rating widget
        key = f"rate_{key_prefix}_{idx}"
        if key not in st.session_state:
            st.session_state[key] = 4.0
        _ = st.select_slider(
            "Rate",
            options=[0.5 * j for j in range(1, 11)],
            value=st.session_state[key],
            key=key,
            label_visibility="collapsed",
        )

        # Add button
        if st.button("Add", key=f"add_{key_prefix}_{idx}", use_container_width=True):
            st.session_state["user_ratings"][int(idx)] = float(st.session_state[key])
            st.session_state["last_update"] = time.time()
            st.session_state["need_update"] = True
            st.rerun()


def idxs_to_titles(
    dataset: CompactDatasetCSR, idxs: List[int], movies_df: pd.DataFrame
) -> List[str]:
    titles = []
    for idx in idxs:
        movie_id = dataset.idx_to_movieId[idx]
        row = movies_df.loc[movies_df["movieId"] == int(movie_id)]
        if not row.empty:
            titles.append(row.iloc[0]["title"])
        else:
            titles.append(str(movie_id))
    return titles


def top_by_bias(item_biases: np.ndarray, top_n: int = 10) -> List[int]:
    idxs = np.argsort(item_biases)[-top_n:][::-1]
    return idxs.tolist()


def random_movies(dataset: CompactDatasetCSR, count: int = 10) -> List[int]:
    N = dataset.movie_size
    if N == 0:
        return []
    return random.sample(range(N), min(count, N))


def genre_top_movies(
    dataset: CompactDatasetCSR, v: np.ndarray, genre_name: str, top_k: int = 5
) -> List[int]:
    genre2id = getattr(dataset, "genre2id", None)
    if genre2id is None or genre_name not in genre2id:
        return []
    g = genre2id[genre_name]
    
    indptr = dataset.movie_genre_indptr
    ids = dataset.movie_genre_ids
    candidates = []
    for n in range(dataset.movie_size):
        start = indptr[n]
        end = indptr[n + 1]
        if start == end:
            continue
        found = False
        for i in range(start, end):
            if ids[i] == g:
                found = True
                break
        if found:
            candidates.append(n)

    if len(candidates) == 0:
        return []

    norms = [(n, np.linalg.norm(v[n])) for n in candidates]
    norms.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in norms[:top_k]]


def generate_recommendations(model_data, movie_rating_counts, movies_df, links_df, tmdb_key):
    """Generate recommendations based on current user ratings"""
    if len(st.session_state["user_ratings"]) == 0:
        return []
    
    print(f"DEBUG: Generating recommendations for {len(st.session_state['user_ratings'])} rated movies")
    print(f"DEBUG: User ratings: {st.session_state['user_ratings']}")
    
    rated_movies = [
        (int(idx), float(r))
        for idx, r in st.session_state["user_ratings"].items()
    ]
    
    print(f"DEBUG: Creating DummyUser with k={model_data['k']}")
    user = DummyUser(rated_movies, model_data["k"])
    
    # Extract model parameters
    v = model_data['v']
    item_biases = model_data['item_biases']
    lamda = model_data['lamda']
    gamma = model_data['gamma']
    dataset = model_data['dataset']
    
    # Update user embedding
    user.update(v, item_biases, lamda, gamma)
    
    # Get scores for all movies
    scores = user.score_for_item()[0]
    
    # Get top candidates (3x what we need for filtering)
    sorted_idx = np.argsort(scores)
    best = sorted_idx[-(20 * 3):][::-1]
    best.remo
    
    print(f"DEBUG: Got {len(best)} candidate indices")
    
    # Filter movies with fewer than 100 ratings
    filtered_recommendations = []
    for idx in best:
        try:
            movie_id = dataset.idx_to_movieId[int(idx)]
            rating_count = movie_rating_counts.get(str(movie_id), 0)
            
            if rating_count >= 100:
                filtered_recommendations.append(idx)
                
            if len(filtered_recommendations) >= 20:
                break
        except Exception as e:
            print(f"ERROR processing idx {idx}: {e}")
            continue
    
    print(f"DEBUG: After filtering, got {len(filtered_recommendations)} recommendations")
    
    # Convert to movie details
    rec_titles = []
    for idx in filtered_recommendations:
        try:
            movie_id = dataset.idx_to_movieId[int(idx)]
            movie_id_int = int(movie_id)
            
            # Get title
            row = movies_df[movies_df["movieId"] == movie_id_int]
            if not row.empty:
                title = row.iloc[0]["title"]
            else:
                title = str(movie_id)
            
            # Get poster
            poster = get_poster_for_movieid(str(movie_id_int), links_df, tmdb_key)
            
            rec_titles.append((title, poster, movie_id_int, idx))
        except Exception as e:
            print(f"ERROR processing recommendation idx {idx}: {e}")
            continue
    
    print(f"DEBUG: Returning {len(rec_titles)} recommendations")
    return rec_titles


def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")

    st.markdown(
        """
    <style>
        .movie-card {
            height: 420px;
            display: flex;
            flex-direction: column;
            margin-bottom: 10px;
        }
        .poster-container {
            width: 180px;
            height: 270px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #f0f0f0;
            margin-bottom: 8px;
            overflow: hidden;
        }
        .poster-container img {
            width: 180px;
            height: 270px;
            object-fit: cover;
        }
        .movie-title {
            height: 48px;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            font-weight: bold;
            font-size: 14px;
            line-height: 1.4;
            margin-bottom: 4px;
        }
        .movie-genres {
            height: 32px;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            font-size: 11px;
            color: #666;
            margin-bottom: 8px;
        }
        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
        }
        .stSlider {
            margin-top: 5px;
            margin-bottom: 5px;
        }
        .stButton button {
            width: 100%;
            margin-top: 5px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🎬 Movie Recommender System")

    # Load assets ONCE with caching
    load_data()
    model_data = load_model()
    movies_df = load_movies_df()
    dataset = model_data["dataset"]
    v = model_data["v"]
    item_biases = model_data["item_biases"]
    
    # Load movie rating counts for filtering (lightweight)
    movie_rating_counts = load_train_dataset_for_filtering()
    
    links_df = load_links_df()
    tmdb_key = getattr(config, "TMDB_API_KEY", None)

    # Initialize session state
    if "user_ratings" not in st.session_state:
        st.session_state["user_ratings"] = {}
        st.session_state["last_update"] = 0
        st.session_state["recommendations"] = []
        st.session_state["need_update"] = False

    # Auto-generate recommendations when ratings change
    if st.session_state["need_update"] and len(st.session_state["user_ratings"]) > 0:
        print("DEBUG: Auto-updating recommendations...")
        recs = generate_recommendations(model_data, movie_rating_counts, movies_df, links_df, tmdb_key)
        st.session_state["recommendations"] = recs
        st.session_state["need_update"] = False
        print(f"DEBUG: Stored {len(recs)} recommendations in session state")

    # SIDEBAR
    st.sidebar.header("🔍 Search & Rate Movies")
    query = st.sidebar.text_input("Search movie title")
    if query:
        results = search_movies(movies_df, query, limit=10)
        if not results.empty:
            selected = st.sidebar.selectbox(
                "Select movie to rate", options=results["title"].tolist()
            )
            if selected:
                row = results[results["title"] == selected].iloc[0]
                movie_id = int(row["movieId"])
                try:
                    idx = movie_id_to_idx(dataset, [movie_id])[0]
                    rating = st.sidebar.slider("Your rating", 0.5, 5.0, 4.0, 0.5)
                    if st.sidebar.button("Add rating", use_container_width=True):
                        st.session_state["user_ratings"][int(idx)] = float(rating)
                        st.session_state["last_update"] = time.time()
                        st.session_state["need_update"] = True
                        st.sidebar.success(f"Added {row['title']} → {rating}⭐")
                        st.rerun()
                except KeyError:
                    st.sidebar.error("Movie not in model dataset.")

    # Show current ratings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Last Watched")
    if st.session_state["user_ratings"]:
        for idx, r in list(st.session_state["user_ratings"].items())[:10]:
            try:
                movie_id = dataset.idx_to_movieId[int(idx)]
                title = idxs_to_titles(dataset, [int(idx)], movies_df)[0]
                display_title = title if len(title) <= 30 else title[:27] + "..."
                st.sidebar.write(f"• {display_title}: {r}⭐")
            except Exception:
                st.sidebar.write(f"• Movie {idx}: {r}⭐")

        if len(st.session_state["user_ratings"]) > 10:
            st.sidebar.write(
                f"... and {len(st.session_state['user_ratings']) - 10} more"
            )
    else:
        st.sidebar.info("No ratings yet. Rate some movies!")

    if st.sidebar.button("Clear All Ratings", use_container_width=True):
        st.session_state["user_ratings"] = {}
        st.session_state["recommendations"] = []
        st.session_state["need_update"] = False
        st.rerun()

    # MAIN CONTENT
    # Row 1: Personalized recommendations
    st.subheader("You Might Like These")
    
    print(f"DEBUG: Rendering recommendations section. Count: {len(st.session_state['recommendations'])}")
    
    if st.session_state["recommendations"]:
        recs = st.session_state["recommendations"][:10]
        print(f"DEBUG: Displaying {len(recs)} recommendations")
        cols = st.columns(5)
        for i, (title, poster, movie_id_int, idx) in enumerate(recs):
            col = cols[i % 5]
            movie_id = str(movie_id_int)
            print(f"DEBUG: Rendering rec {i}: {title} (idx={idx})")
            render_movie_card(
                col, idx, movie_id, movies_df, links_df, tmdb_key, dataset, f"rec_{i}"
            )
    else:
        print("DEBUG: No recommendations to display")
        st.info("⭐ Rate some movies to see personalized recommendations!")

    st.markdown("---")

    # Row 2: Top movies
    st.subheader("⭐ Top Rated Movies")
    top_idxs = top_by_bias(item_biases, top_n=10)
    cols = st.columns(5)
    for i, idx in enumerate(top_idxs):
        col = cols[i % 5]
        movie_id = dataset.idx_to_movieId[int(idx)]
        render_movie_card(
            col, idx, movie_id, movies_df, links_df, tmdb_key, dataset, f"top_{i}"
        )

    st.markdown("---")

    # Row 3: Random picks
    st.subheader("🎲 Random Picks For You")
    rand_idxs = random_movies(dataset, 10)
    cols = st.columns(5)
    for i, idx in enumerate(rand_idxs):
        col = cols[i % 5]
        movie_id = dataset.idx_to_movieId[int(idx)]
        render_movie_card(
            col, idx, movie_id, movies_df, links_df, tmdb_key, dataset, f"rand_{i}"
        )

    st.markdown("---")

    # Row 4: Genre highlights
    st.subheader("🎭 Genre Highlights")
    genre_list = ["Animation", "Comedy", "Action", "Drama", "Horror"]
    for genre in genre_list:
        with st.expander(f"📁 {genre}", expanded=False):
            g_top = genre_top_movies(dataset, v, genre, top_k=5)
            if g_top:
                cols = st.columns(5)
                for i, idx in enumerate(g_top):
                    col = cols[i % 5]
                    movie_id = dataset.idx_to_movieId[int(idx)]
                    render_movie_card(
                        col,
                        idx,
                        movie_id,
                        movies_df,
                        links_df,
                        tmdb_key,
                        dataset,
                        f"genre_{genre}_{i}",
                    )
            else:
                st.info(f"No movies found for {genre}")


if __name__ == "__main__":
    main()
