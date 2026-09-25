import io
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="MovieLens New User Recommender", page_icon="🎬", layout="wide")

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ASPECTS = [
    "Story", "Acting", "Visuals", "Soundtrack", "Comedy", "Action",
    "Romance", "Emotion", "Sci-Fi", "Horror", "Family", "Crime"
]

ASPECT_KEYWORDS = {
    "Story": ["story", "plot", "narrative", "screenplay", "script", "writing", "twist", "ending", "dialogue"],
    "Acting": ["acting", "actor", "actress", "cast", "performance", "performances", "character", "characters"],
    "Visuals": ["visual", "visuals", "cinematography", "effects", "cgi", "animation", "beautiful", "scenery", "color"],
    "Soundtrack": ["music", "soundtrack", "score", "song", "songs", "sound", "audio"],
    "Comedy": ["comedy", "funny", "humor", "humour", "hilarious", "laugh", "satire", "parody"],
    "Action": ["action", "fight", "fighting", "battle", "war", "explosion", "adventure", "superhero", "chase"],
    "Romance": ["romance", "romantic", "love", "relationship", "wedding", "couple"],
    "Emotion": ["emotional", "emotion", "sad", "touching", "heart", "moving", "inspiring", "tear", "drama"],
    "Sci-Fi": ["sci-fi", "scifi", "science fiction", "space", "future", "robot", "alien", "technology", "time travel"],
    "Horror": ["horror", "scary", "fear", "ghost", "zombie", "monster", "creepy", "slasher", "thriller"],
    "Family": ["family", "children", "kids", "disney", "pixar", "animated", "cartoon"],
    "Crime": ["crime", "detective", "mystery", "murder", "police", "gangster", "mafia", "noir"]
}

GENRE_TO_ASPECT = {
    "Action": "Action", "Adventure": "Action", "Animation": "Family", "Children": "Family",
    "Comedy": "Comedy", "Crime": "Crime", "Documentary": "Story", "Drama": "Emotion",
    "Fantasy": "Visuals", "Film-Noir": "Crime", "Horror": "Horror", "Musical": "Soundtrack",
    "Mystery": "Crime", "Romance": "Romance", "Sci-Fi": "Sci-Fi", "Thriller": "Horror",
    "War": "Action", "Western": "Action", "IMAX": "Visuals"
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "best", "classic", "masterpiece", "beautiful",
    "funny", "hilarious", "clever", "smart", "touching", "inspiring", "favorite", "favourite",
    "awesome", "perfect", "brilliant", "cool", "interesting", "entertaining", "fun", "strong"
}
NEGATIVE_WORDS = {
    "bad", "boring", "worst", "weak", "poor", "terrible", "awful", "dull", "slow",
    "stupid", "waste", "annoying", "predictable", "disappointing", "overrated"
}


def normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("|", " ")).strip()


def min_max(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


@st.cache_data(show_spinner="Downloading MovieLens dataset...")
def download_movielens():
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        movies = pd.read_csv(zf.open("ml-latest-small/movies.csv"))
        ratings = pd.read_csv(zf.open("ml-latest-small/ratings.csv"))
        tags = pd.read_csv(zf.open("ml-latest-small/tags.csv"))
    return movies, ratings, tags


def tag_aspect_scores(tags: pd.DataFrame) -> pd.DataFrame:
    if tags.empty:
        return pd.DataFrame(columns=["movieId"] + [f"tag_{a}" for a in ASPECTS] + ["tag_sentiment"])

    tag_rows = []
    for _, row in tags.iterrows():
        text = str(row.get("tag", "")).lower()
        tokens = set(re.findall(r"[a-zA-Z\-]+", text))
        sentiment = 0.0
        if tokens & POSITIVE_WORDS:
            sentiment += 1.0
        if tokens & NEGATIVE_WORDS:
            sentiment -= 1.0
        if sentiment == 0.0:
            sentiment = 0.2  # neutral tags still indicate relevance, but less sentiment strength

        aspect_values = {}
        for aspect, keywords in ASPECT_KEYWORDS.items():
            match = any(k in text for k in keywords)
            aspect_values[f"tag_{aspect}"] = sentiment if match else 0.0
        aspect_values["movieId"] = row["movieId"]
        aspect_values["tag_sentiment"] = sentiment
        tag_rows.append(aspect_values)

    tag_df = pd.DataFrame(tag_rows)
    tag_cols = [c for c in tag_df.columns if c.startswith("tag_")]
    return tag_df.groupby("movieId", as_index=False)[tag_cols].mean()


@st.cache_data(show_spinner="Building movie profiles...")
def build_profiles():
    movies, ratings, tags = download_movielens()
    movies["title"] = movies["title"].apply(normalise_title)
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")

    rating_stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    profiles = movies.merge(rating_stats, on="movieId", how="left")
    profiles["avg_rating"] = profiles["avg_rating"].fillna(0)
    profiles["rating_count"] = profiles["rating_count"].fillna(0)
    profiles["quality_score"] = min_max(profiles["avg_rating"]) * 0.75 + min_max(np.log1p(profiles["rating_count"])) * 0.25

    for aspect in ASPECTS:
        profiles[aspect] = 0.0

    for idx, row in profiles.iterrows():
        genres = str(row["genres"]).split("|") if row["genres"] else []
        for genre in genres:
            aspect = GENRE_TO_ASPECT.get(genre)
            if aspect:
                profiles.at[idx, aspect] += 0.7

        title_text = str(row["title"]).lower()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(k in title_text for k in keywords):
                profiles.at[idx, aspect] += 0.3

    tag_scores = tag_aspect_scores(tags)
    profiles = profiles.merge(tag_scores, on="movieId", how="left")
    for aspect in ASPECTS:
        tag_col = f"tag_{aspect}"
        if tag_col in profiles.columns:
            profiles[tag_col] = profiles[tag_col].fillna(0)
            profiles[aspect] = profiles[aspect] + profiles[tag_col].clip(lower=0) * 0.8
    profiles["tag_sentiment"] = profiles.get("tag_sentiment", pd.Series(np.zeros(len(profiles)))).fillna(0)

    # Convert raw aspect relevance into a sentiment-weighted aspect profile using real rating quality.
    # Higher-rated movies with matching genres/tags receive stronger aspect scores.
    for aspect in ASPECTS:
        profiles[aspect] = profiles[aspect] * (0.5 + profiles["quality_score"])
        profiles[aspect] = min_max(profiles[aspect])

    return movies, ratings, tags, profiles


def cosine_similarity_matrix(candidate_matrix: np.ndarray, user_vector: np.ndarray) -> np.ndarray:
    user_norm = np.linalg.norm(user_vector)
    cand_norm = np.linalg.norm(candidate_matrix, axis=1)
    denom = cand_norm * user_norm
    denom[denom == 0] = 1e-9
    return np.dot(candidate_matrix, user_vector) / denom


def recommend_movies(profiles: pd.DataFrame, selected_movie_ids, aspect_weights, top_n=10):
    aspect_cols = ASPECTS
    user_vector = np.zeros(len(aspect_cols))

    # Preference signal from movies selected by the new user.
    if selected_movie_ids:
        selected_profiles = profiles[profiles["movieId"].isin(selected_movie_ids)]
        if not selected_profiles.empty:
            user_vector += selected_profiles[aspect_cols].mean().values * 0.55

    # Preference signal from aspect sliders.
    slider_vector = np.array([aspect_weights.get(a, 0) / 5.0 for a in aspect_cols])
    user_vector += slider_vector * 0.45

    # Avoid zero profile if the user selects very little.
    if np.linalg.norm(user_vector) == 0:
        user_vector = np.ones(len(aspect_cols)) / len(aspect_cols)

    candidates = profiles[~profiles["movieId"].isin(selected_movie_ids)].copy()
    matrix = candidates[aspect_cols].values.astype(float)
    similarity = cosine_similarity_matrix(matrix, user_vector)
    aspect_bonus = matrix.dot(slider_vector) / (np.linalg.norm(slider_vector) + 1e-9) if np.linalg.norm(slider_vector) > 0 else 0
    candidates["recommendation_score"] = (
        similarity * 0.60
        + candidates["quality_score"].values * 0.30
        + np.array(aspect_bonus) * 0.10
    )
    return candidates.sort_values("recommendation_score", ascending=False).head(top_n)


def run_small_evaluation(profiles, ratings, sample_users=75, k=10):
    rng = np.random.default_rng(42)
    eligible = ratings.groupby("userId").filter(lambda x: (x["rating"] >= 4).sum() >= 6)
    users = eligible["userId"].unique()
    if len(users) == 0:
        return None
    users = rng.choice(users, size=min(sample_users, len(users)), replace=False)

    hits_model, hits_pop = 0, 0
    total = 0
    popular = set(profiles.sort_values("quality_score", ascending=False).head(k)["movieId"].tolist())

    for user_id in users:
        liked = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4)]["movieId"].tolist()
        liked = [m for m in liked if m in set(profiles["movieId"])]
        if len(liked) < 6:
            continue
        rng.shuffle(liked)
        train = liked[:5]
        test = set(liked[5:])
        recs = recommend_movies(profiles, train, {a: 0 for a in ASPECTS}, top_n=k)
        rec_set = set(recs["movieId"].tolist())
        if rec_set & test:
            hits_model += 1
        if popular & test:
            hits_pop += 1
        total += 1

    if total == 0:
        return None
    return {
        "sample_users": total,
        "model_hit_rate_at_k": hits_model / total,
        "popularity_hit_rate_at_k": hits_pop / total,
        "k": k,
    }


st.title("🎬 MovieLens New User Movie Recommender")
st.caption("Prototype artefact for an MSc Computing dissertation: new-user movie recommendation using real MovieLens data, aspect preferences, ratings and user-generated tags.")

try:
    movies_df, ratings_df, tags_df, profiles_df = build_profiles()
except Exception as exc:
    st.error("Dataset could not be loaded. Please check internet access or upload the MovieLens CSV files.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Dataset")
    st.write("Source: MovieLens latest-small")
    st.metric("Movies", f"{len(movies_df):,}")
    st.metric("Ratings", f"{len(ratings_df):,}")
    st.metric("Tags", f"{len(tags_df):,}")
    st.divider()
    top_n = st.slider("Number of recommendations", 5, 20, 10)

st.subheader("1. New user input")
st.write("Select a few movies you already like, then rate how important different movie aspects are to you.")

popular_titles = profiles_df[profiles_df["rating_count"] >= 20].sort_values("avg_rating", ascending=False)
movie_options = dict(zip(popular_titles["title"], popular_titles["movieId"]))
selected_titles = st.multiselect(
    "Select 3–8 movies you like",
    options=list(movie_options.keys()),
    default=[]
)
selected_movie_ids = [movie_options[t] for t in selected_titles]

st.subheader("2. Aspect preferences")
cols = st.columns(3)
aspect_weights = {}
for i, aspect in enumerate(ASPECTS):
    with cols[i % 3]:
        aspect_weights[aspect] = st.slider(aspect, 0, 5, 3, help="0 = not important, 5 = very important")

if st.button("Generate recommendations", type="primary"):
    results = recommend_movies(profiles_df, selected_movie_ids, aspect_weights, top_n=top_n)
    st.subheader("3. Recommended movies")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        aspect_summary = sorted([(a, row[a]) for a in ASPECTS], key=lambda x: x[1], reverse=True)[:3]
        aspect_text = ", ".join([f"{a}" for a, _ in aspect_summary])
        st.markdown(
            f"**{rank}. {row['title']}**  \n"
            f"Genres: {row['genres']}  \n"
            f"Average rating: {row['avg_rating']:.2f} from {int(row['rating_count'])} ratings  \n"
            f"Strong matching aspects: {aspect_text}  \n"
            f"Recommendation score: {row['recommendation_score']:.3f}"
        )
        st.divider()

st.subheader("Evaluation evidence")
st.write("The button below runs a small offline evaluation using real MovieLens users as historical test records. These users are not contacted; their anonymised ratings are only used to test whether the recommendation logic can recover held-out liked movies.")
if st.button("Run quick evaluation"):
    with st.spinner("Running evaluation..."):
        evaluation = run_small_evaluation(profiles_df, ratings_df, sample_users=75, k=10)
    if evaluation:
        c1, c2, c3 = st.columns(3)
        c1.metric("Sample users", evaluation["sample_users"])
        c2.metric(f"Prototype Hit Rate@{evaluation['k']}", f"{evaluation['model_hit_rate_at_k']:.2%}")
        c3.metric(f"Popularity Baseline@{evaluation['k']}", f"{evaluation['popularity_hit_rate_at_k']:.2%}")
        st.info("Use these values as example evaluation evidence in your dissertation, but also explain the limitations of a small offline test.")
    else:
        st.warning("Not enough eligible users for evaluation.")

with st.expander("How this prototype works"):
    st.write("""
    This prototype uses the real MovieLens latest-small dataset. Because MovieLens does not include full movie review paragraphs, the artefact uses structured ratings, genres and user-generated free-text tags as available MovieLens evidence. Movie aspect profiles are created from genres, tags and rating quality. A new user profile is created from selected liked movies and explicit aspect preferences. Recommendations are generated by comparing the new user profile to movie profiles and ranking suitable movies.
    """)
