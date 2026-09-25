# Aspect-Based Sentiment Analysis Movie Recommendation System

This project is a prototype artefact for an MSc Computing dissertation titled:

**Developing and Evaluating an Aspect-Based Sentiment Analysis Recommendation System for New User Movie Recommendations**

## Purpose

The system is designed for **new users only**. A new user gives a small amount of initial information by selecting/rating movies and choosing the aspects they care about. The system then uses aspect-based sentiment signals from movie review text to recommend movies.

## How to Run

1. Install Python 3.10 or later.
2. Open the project folder in a terminal.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

## Main Features

- New user movie preference input
- Aspect preference input, including storyline, acting, visuals, soundtrack, action, comedy and emotion
- Simple aspect-based sentiment analysis from movie review text
- User profile creation
- Movie recommendation generation
- Explanation of why each movie is recommended
- Basic evaluation using Precision@10 style testing

## Data Files

- `data/movies.csv`: movie metadata and average ratings
- `data/reviews.csv`: original sample review text used for aspect-sentiment extraction

## Important Dissertation Note

This artefact is a functional prototype. The sample dataset can be replaced with larger datasets such as MovieLens combined with publicly available movie review data. The implemented version demonstrates the workflow and technical feasibility of new-user recommendations using aspect-based sentiment information.
