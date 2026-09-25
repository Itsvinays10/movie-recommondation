"""
Aspect-Based Sentiment Analysis Movie Recommendation System for New Users
Author: Vinay (MSc Computing Dissertation Artefact)

This Streamlit application demonstrates a prototype recommendation system for new users.
The user gives initial preferences by selecting movies and choosing movie aspects they like.
The system extracts aspect-level sentiment signals from review text and recommends movies
that best match the new user's initial profile.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
MOVIES_FILE = DATA_DIR / "movies.csv"
REVIEWS_FILE = DATA_DIR / "reviews.csv"

# Movie aspects used by the prototype. These match the dissertation scope.
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "storyline": ["story", "storyline", "plot", "narrative", "writing", "ending", "twist", "dialogue"],
    "acting": ["acting", "performance", "performances", "characters", "character", "cast", "voice acting"],
    "visuals": ["visual", "visuals", "cinematography", "animation", "effects", "style", "world-building", "design"],
    "soundtrack": ["music", "soundtrack", "sound", "sound design", "score"],
    "action": ["action", "fight", "battle", "battles", "choreography", "thriller", "suspense", "tension"],
    "comedy": ["comedy", "funny", "humour", "humor", "jokes", "quirky"],
    "emotion": ["emotional", "emotion", "heartwarming", "moving", "touching", "romantic", "romance", "inspiring"],
}

# Simple lexicon used for sentiment scoring. This avoids relying on a black-box API.
POSITIVE_WORDS = {
    "excellent", "great", "good", "strong", "powerful", "beautiful", "impressive", "outstanding",
    "memorable", "enjoyable", "effective", "clever", "creative", "fascinating", "brilliant", "warm",
    "charming", "amazing", "engaging", "satisfying", "intense", "original", "smart", "sharp",
    "rewarding", "inspiring", "heartwarming", "moving", "touching", "energetic", "stylish", "fun",
    "funny", "colourful", "colorful", "magical", "gripping", "pleasant", "sincere", "suspenseful",
}
NEGATIVE_WORDS = {
    "weak", "slow", "confusing", "unclear", "predictable", "minimal", "limited", "dark", "violent",
    "excessive", "disturbing", "conventional", "familiar", "crowded", "long", "simple", "heavy",
    "divided", "predictable", "not", "minimal", "too", "weakness", "weaknesses",
}


def clean_text(text: str) -> str:
    """Lowercase and normalise text for simple NLP processing."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_split(text: str) -> List[str]:
    """Split review text into basic sentence-like chunks."""
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def sentiment_score(sentence: str) -> float:
    """
    Calculate a simple lexicon-based sentiment score for a sentence.
    Output is approximately in the range [-1, 1].
    """
    words = clean_text(sentence).split()
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def extract_aspect_sentiments(review_text: str) -> Dict[str, float]:
    """
    Aspect-Based Sentiment Analysis prototype.
    For each sentence, detect mentioned aspects using keywords and assign sentiment.
    The final score for each aspect is the average sentence sentiment mentioning that aspect.
    """
    aspect_scores: Dict[str, List[float]] = {aspect: [] for aspect in ASPECT_KEYWORDS}
    for sentence in sentence_split(review_text):
        cleaned = clean_text(sentence)
        s_score = sentiment_score(sentence)
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(keyword in cleaned for keyword in keywords):
                aspect_scores[aspect].append(s_score)

    # Return average score; if aspect not mentioned, score is 0.
    return {
        aspect: float(np.mean(scores)) if scores else 0.0
        for aspect, scores in aspect_scores.items()
    }


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(MOVIES_FILE)
    reviews = pd.read_csv(REVIEWS_FILE)

    rows = []
    for _, row in reviews.iterrows():
        scores = extract_aspect_sentiments(row["review_text"])
        scores["movie_id"] = row["movie_id"]
        rows.append(scores)

    aspect_review_scores = pd.DataFrame(rows)
    aspect_profile = aspect_review_scores.groupby("movie_id").mean().reset_index()
    movie_profiles = movies.merge(aspect_profile, on="movie_id", how="left").fillna(0.0)
    return movies, reviews, movie_profiles


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two numeric vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalise_rating(rating: float) -> float:
    """Convert 1-5 star rating to 0-1 weight."""
    return max(0.0, min(1.0, (rating - 1.0) / 4.0))


def build_user_profile(
    movie_profiles: pd.DataFrame,
    selected_ratings: Dict[int, float],
    aspect_preferences: Dict[str, float],
    preferred_genres: List[str],
) -> Dict[str, float]:
    """
    Create the new user's initial preference profile from:
    - explicit aspect preferences selected by the user
    - aspect profiles of initially selected/rated movies
    - preferred genres
    """
    profile = {aspect: aspect_preferences.get(aspect, 0.0) for aspect in ASPECT_KEYWORDS}

    # Add signals from rated/selected movies.
    for movie_id, rating in selected_ratings.items():
        weight = normalise_rating(rating)
        movie_row = movie_profiles[movie_profiles["movie_id"] == movie_id]
        if movie_row.empty:
            continue
        movie_row = movie_row.iloc[0]
        for aspect in ASPECT_KEYWORDS:
            profile[aspect] += weight * float(movie_row[aspect])

    # Add genre preferences as separate keys.
    all_genres = sorted({g for gs in movie_profiles["genres"] for g in gs.split("|")})
    for genre in all_genres:
        profile[f"genre::{genre}"] = 1.0 if genre in preferred_genres else 0.0

    return profile


def recommend_movies(
    movie_profiles: pd.DataFrame,
    user_profile: Dict[str, float],
    selected_movie_ids: List[int],
    top_k: int = 10,
) -> pd.DataFrame:
    """Generate movie recommendations based on aspect similarity, genre match and average rating."""
    aspect_cols = list(ASPECT_KEYWORDS.keys())
    all_genres = sorted({g for gs in movie_profiles["genres"] for g in gs.split("|")})

    user_aspect_vec = np.array([user_profile.get(a, 0.0) for a in aspect_cols], dtype=float)
    user_genre_vec = np.array([user_profile.get(f"genre::{g}", 0.0) for g in all_genres], dtype=float)

    results = []
    for _, row in movie_profiles.iterrows():
        if int(row["movie_id"]) in selected_movie_ids:
            continue

        movie_aspect_vec = np.array([float(row[a]) for a in aspect_cols], dtype=float)
        aspect_similarity = vector_cosine(user_aspect_vec, movie_aspect_vec)

        movie_genres = row["genres"].split("|")
        movie_genre_vec = np.array([1.0 if g in movie_genres else 0.0 for g in all_genres], dtype=float)
        genre_similarity = vector_cosine(user_genre_vec, movie_genre_vec)

        rating_score = float(row["avg_rating"]) / 5.0

        # Weighted hybrid score. Weights can be justified in dissertation as design decision.
        final_score = (0.55 * aspect_similarity) + (0.25 * genre_similarity) + (0.20 * rating_score)

        top_aspects = sorted(
            [(a, float(row[a])) for a in aspect_cols], key=lambda x: x[1], reverse=True
        )[:3]
        why = ", ".join([a for a, score in top_aspects if score > 0]) or "overall profile match"

        results.append({
            "movie_id": int(row["movie_id"]),
            "title": row["title"],
            "year": int(row["year"]),
            "genres": row["genres"],
            "avg_rating": float(row["avg_rating"]),
            "aspect_similarity": round(aspect_similarity, 3),
            "genre_similarity": round(genre_similarity, 3),
            "recommendation_score": round(final_score, 3),
            "why_recommended": why,
        })

    recs = pd.DataFrame(results).sort_values("recommendation_score", ascending=False).head(top_k)
    return recs


def run_demo_evaluation(movie_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Simple repeatable evaluation for dissertation screenshots.
    This creates three test profiles and checks whether recommended movies match expected genres/aspects.
    It is not a full academic benchmark, but it provides artefact-level evaluation evidence.
    """
    test_cases = [
        {
            "name": "Action and visuals user",
            "aspects": {"action": 5, "visuals": 5, "storyline": 3, "acting": 3, "soundtrack": 2, "comedy": 1, "emotion": 1},
            "genres": ["Action", "Sci-Fi", "Adventure"],
            "expected_genres": {"Action", "Sci-Fi", "Adventure"},
        },
        {
            "name": "Drama and acting user",
            "aspects": {"acting": 5, "storyline": 5, "emotion": 4, "visuals": 2, "soundtrack": 2, "action": 1, "comedy": 1},
            "genres": ["Drama", "Crime"],
            "expected_genres": {"Drama", "Crime"},
        },
        {
            "name": "Animation and comedy user",
            "aspects": {"comedy": 5, "visuals": 4, "emotion": 4, "storyline": 3, "acting": 2, "soundtrack": 2, "action": 1},
            "genres": ["Animation", "Comedy", "Family"],
            "expected_genres": {"Animation", "Comedy", "Family"},
        },
    ]

    rows = []
    for case in test_cases:
        profile = build_user_profile(
            movie_profiles,
            selected_ratings={},
            aspect_preferences=case["aspects"],
            preferred_genres=case["genres"],
        )
        recs = recommend_movies(movie_profiles, profile, selected_movie_ids=[], top_k=10)
        hits = 0
        for genres in recs["genres"]:
            if case["expected_genres"].intersection(set(genres.split("|"))):
                hits += 1
        precision_at_10 = hits / 10
        rows.append({
            "Test Profile": case["name"],
            "Relevant Recommendations in Top 10": hits,
            "Precision@10": round(precision_at_10, 2),
        })
    return pd.DataFrame(rows)


# ------------------------- Streamlit user interface -------------------------

st.set_page_config(
    page_title="ABSA Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

movies_df, reviews_df, movie_profiles_df = load_data()

st.title("🎬 Aspect-Based Sentiment Movie Recommender")
st.caption("Prototype artefact for MSc Computing dissertation: recommendations for new users only")

with st.expander("About this prototype", expanded=False):
    st.write(
        "This artefact is designed for new users. The user provides initial movie ratings and aspect preferences. "
        "The system analyses review text using a simple aspect-based sentiment method and recommends movies that best match the new user's profile."
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Select movies you already like")
    movie_options = {f"{row.title} ({row.year})": int(row.movie_id) for row in movies_df.itertuples()}
    selected_movie_labels = st.multiselect(
        "Choose 3 to 5 movies you like. This simulates a new user's initial input.",
        options=list(movie_options.keys()),
        default=["Inception (2010)", "The Matrix (1999)", "Interstellar (2014)"],
    )

    selected_ratings: Dict[int, float] = {}
    for label in selected_movie_labels:
        movie_id = movie_options[label]
        rating = st.slider(f"Your rating for {label}", 1.0, 5.0, 4.0, 0.5)
        selected_ratings[movie_id] = rating

with col2:
    st.subheader("2. Select what matters to you")
    aspect_preferences: Dict[str, float] = {}
    for aspect in ASPECT_KEYWORDS:
        aspect_preferences[aspect] = st.slider(
            f"Importance of {aspect}", 0, 5, 3, help="0 = not important, 5 = very important"
        )

    all_genres = sorted({g for gs in movies_df["genres"] for g in gs.split("|")})
    preferred_genres = st.multiselect(
        "Preferred genres",
        options=all_genres,
        default=["Action", "Sci-Fi"],
    )

st.divider()

if st.button("Generate Recommendations", type="primary"):
    if not selected_ratings and not any(v > 0 for v in aspect_preferences.values()):
        st.warning("Please select at least one movie or aspect preference.")
    else:
        user_profile = build_user_profile(movie_profiles_df, selected_ratings, aspect_preferences, preferred_genres)
        recs = recommend_movies(
            movie_profiles_df,
            user_profile,
            selected_movie_ids=list(selected_ratings.keys()),
            top_k=10,
        )

        st.subheader("Recommended Movies")
        st.dataframe(
            recs[["title", "year", "genres", "avg_rating", "recommendation_score", "why_recommended"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Generated New User Profile")
        profile_display = pd.DataFrame({
            "Aspect": list(ASPECT_KEYWORDS.keys()),
            "Profile Weight": [round(float(user_profile.get(a, 0.0)), 3) for a in ASPECT_KEYWORDS],
        })
        st.bar_chart(profile_display.set_index("Aspect"))

        st.info(
            "The recommendations are generated for this new user session only. External dataset users are not contacted or treated as live users."
        )

st.divider()
with st.expander("Show prototype evaluation results"):
    eval_df = run_demo_evaluation(movie_profiles_df)
    st.write("A simple profile-based evaluation checks whether the top 10 recommendations match expected user interests.")
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

with st.expander("Show calculated aspect sentiment profiles"):
    display_cols = ["title", "genres", "avg_rating"] + list(ASPECT_KEYWORDS.keys())
    st.dataframe(movie_profiles_df[display_cols].round(3), use_container_width=True, hide_index=True)
