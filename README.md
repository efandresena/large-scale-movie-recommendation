# Large-scale-movie-recommendation
A from-scratch implementation of a scalable movie recommendation system. 


🎬 Movie Recommendation System
A collaborative filtering movie recommendation system using ALS (Alternating Least Squares) matrix factorization with genre features.

The notebook ipynb documents each development step. The last notebook is the last version and used as basis in this application.


Features :

Pre-trained Model: Get recommendations instantly with our pre-trained model (105MB)

Custom Training: Train your own model from scratch on the MovieLens 25M dataset (with cli only)

Interactive CLI: Search movies, rate them, and get personalized recommendations

Genre-aware: Incorporates movie genres for better recommendations

Fast & Efficient: Optimized with Numba for high performance

Streamlit interface: Web based interface and interactive easy to use

## **Quick Start**

## Clone the repository
```bash
git clone https://github.com/efandresena/large-scale-movie-recommendation.git
cd movie-recommender
```

## Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run the application
```bash
python cli.py
```

## For interface using Streamlit
```bash
streamlit run recommender_system.py
```

## Warning 
For cli or interface the application may take some time to run due to the size of the data to be downloaded and it depends on the user internet speed.

During training some movie didn't have ratings which is the cold start problem so the model can not infer for it yet. This is the main problem of collaborative filtering known as the Cold start problem.