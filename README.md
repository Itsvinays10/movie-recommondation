# MovieLens New User Movie Recommender

This is a Streamlit web app artefact for an MSc Computing dissertation.

## Dataset

The app uses the real MovieLens `ml-latest-small` dataset from GroupLens:
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

The app automatically downloads the dataset at runtime. It uses:
- `movies.csv`
- `ratings.csv`
- `tags.csv`

MovieLens does not contain full review paragraphs, so this prototype uses movie genres, ratings and user-generated free-text tags as the available textual/aspect evidence. This limitation should be stated clearly in the dissertation.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

Upload these files to GitHub:
- `app.py`
- `requirements.txt`
- `README.md`

Then deploy the repository on Streamlit Community Cloud using `app.py` as the main file.

## Artefact description

The artefact is a prototype web-based recommendation system for new users. A new user selects a small number of movies they like and indicates preferred aspects such as story, acting, visuals, soundtrack, comedy and action. The system creates an initial profile and recommends movies using real MovieLens ratings, movie genres and user-generated tags.
