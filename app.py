from flask import Flask, render_template, request, session, redirect, url_for, flash
from Recommendation_System.hybrid_recommender import (
    recommend_movies as hybrid_recommend_movies,
    recommend_similar_movies)
import pandas as pd
import os
import requests
import json
import re
from dotenv import load_dotenv
import sqlite3
import secrets

app = Flask(__name__)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "Processed",
    "cinescope_master.csv"
)

POSTER_CACHE_FILE = os.path.join(
    BASE_DIR,
    "poster_cache.json"
)

# ==========================================
# USER MOVIE RATINGS DATABASE
# ==========================================

INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)

RATINGS_DB = os.path.join(INSTANCE_DIR, "user_ratings.db")

# Persistent Flask secret key
SECRET_KEY_FILE = os.path.join(INSTANCE_DIR, ".flask_secret_key")

if os.getenv("FLASK_SECRET_KEY"):
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")
else:
    if os.path.exists(SECRET_KEY_FILE):
        with open(SECRET_KEY_FILE, "r", encoding="utf-8") as file:
            app.config["SECRET_KEY"] = file.read().strip()
    else:
        generated_key = secrets.token_hex(32)

        with open(SECRET_KEY_FILE, "w", encoding="utf-8") as file:
            file.write(generated_key)

        app.config["SECRET_KEY"] = generated_key


def initialize_ratings_database():

    connection = sqlite3.connect(RATINGS_DB)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS user_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER NOT NULL,
            visitor_id TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(movie_id, visitor_id)
        )
    """)

    connection.commit()
    connection.close()


initialize_ratings_database()

# ==========================================
# OMDb API CONFIGURATION
# ==========================================

load_dotenv(
    dotenv_path=os.path.join(BASE_DIR, ".env"),
    override=True
)

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Alternative titles for MovieLens movies
TITLE_ALIASES = {
    "Independence Day (ID4)": "Independence Day",
    "Twelve Monkeys": "12 Monkeys",
    "Willy Wonka and the Chocolate Factory":
        "Willy Wonka & the Chocolate Factory",
    "Monty Python and the Holy Grail":
        "Monty Python's Life of Brian",
}


# ==========================================
# POSTER CACHE
# ==========================================

if os.path.exists(POSTER_CACHE_FILE):
    try:
        with open(
            POSTER_CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            poster_cache = json.load(file)

    except (json.JSONDecodeError, OSError):
        poster_cache = {}

else:
    poster_cache = {}


def save_poster_cache():

    with open(
        POSTER_CACHE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            poster_cache,
            file,
            indent=4
        )


# ==========================================
# MOVIE POSTER FUNCTION
# ==========================================

def get_movie_poster(title):

    if not title:
        return None

    # Return existing valid cached poster
    if title in poster_cache and poster_cache[title]:
        return poster_cache[title]

    if not OMDB_API_KEY:
        print("OMDb API key missing!")
        return None

    try:

        # Extract year
        year_match = re.search(
            r"\((\d{4})\)$",
            title
        )

        year = (
            year_match.group(1)
            if year_match
            else None
        )

        # Remove year
        clean_title = re.sub(
            r"\s*\(\d{4}\)$",
            "",
            title
        ).strip()

        # Convert MovieLens article format
        # Example: Godfather, The -> The Godfather

        article_match = re.match(
            r"^(.*),\s*(The|A|An)$",
            clean_title,
            re.IGNORECASE
        )

        if article_match:

            clean_title = (
                article_match.group(2)
                + " "
                + article_match.group(1)
            )

        # Apply known alternative titles
        search_title = TITLE_ALIASES.get(
            clean_title,
            clean_title
        )

        # Try multiple title formats
        search_titles = list(dict.fromkeys([
            search_title,
            clean_title,
            title
        ]))

        for current_title in search_titles:

            params = {
                "apikey": OMDB_API_KEY,
                "t": current_title,
                "type": "movie"
            }

            if year:
                params["y"] = year

            response = requests.get(
                "https://www.omdbapi.com/",
                params=params,
                timeout=10
            )

            response.raise_for_status()

            data = response.json()

            if data.get("Response") == "True":

                poster = data.get("Poster")

                if poster and poster != "N/A":

                    poster_cache[title] = poster

                    save_poster_cache()

                    print("Poster found:", title)

                    return poster

        print("Poster not found:", title)

    except (
        requests.RequestException,
        ValueError
    ) as error:

        print("OMDb error:", title, error)

    return None


# ==========================================
# LOAD PROCESSED DATASET
# ==========================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        "Processed dataset not found. "
        "Run prepare_dataset.py first."
    )

df = pd.read_csv(DATASET_PATH, sep=";", encoding="cp1252")

# ==========================================
# PREPARE MOVIE INFORMATION
# ==========================================

movies = (
    df.drop_duplicates(subset="movie_id")
    .copy()
    .reset_index(drop=True)
)

movies["title"] = movies["title"].fillna(
    "Unknown Movie"
)

movies["genres"] = movies["genres"].fillna(
    "Unknown"
)


# ==========================================
# CALCULATE MOVIE STATISTICS
# ==========================================

movie_stats = (
    df.groupby("movie_id")
    .agg(
        average_rating=("rating", "mean"),
        total_ratings=("rating", "count")
    )
    .reset_index()
)

movies = movies.merge(
    movie_stats,
    on="movie_id",
    how="left"
)

movies["average_rating"] = (
    movies["average_rating"].round(2)
)


# ==========================================
# POPULAR MOVIES
# ==========================================

popular_movies = (
    movies[movies["total_ratings"] >= 20]
    .sort_values(
        by=["average_rating", "total_ratings"],
        ascending=[False, False]
    )
)


# ==========================================
# EXTRACT GENRES
# ==========================================

genres = sorted({

    genre.strip()

    for value in movies["genres"]

    for genre in str(value).split(",")

    if genre.strip()

})


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    search = request.args.get(
        "search",
        ""
    ).strip()

    selected_genre = request.args.get(
        "genre",
        ""
    ).strip()

    results = movies.copy()

    # Search movies
    if search:

        results = results[
            results["title"].str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ]

    # Filter by genre
    if selected_genre:

        results = results[
            results["genres"].str.contains(
                selected_genre,
                case=False,
                na=False,
                regex=False
            )
        ]

    # Sort by popularity
    results = results.sort_values(
        by="total_ratings",
        ascending=False
    )

    # Fetch Explore Movies posters
    display_movies = results.head(30).copy()

    display_movies["poster"] = (
        display_movies["title"].apply(
            get_movie_poster
        )
    )

    # Fetch Popular Movies posters
    display_popular = popular_movies.head(10).copy()

    display_popular["poster"] = (
        display_popular["title"].apply(
            get_movie_poster
        )
    )

    return render_template(

        "index.html",

        movies=display_movies.to_dict("records"),

        popular=display_popular.to_dict("records"),

        genres=genres,

        selected_genre=selected_genre,

        search=search,

        total_movies=len(movies),

        total_users=df["user_id"].nunique(),

        total_ratings=len(df)

    )

# ==========================================
# SUBMIT OR UPDATE MOVIE RATING
# ==========================================

@app.route("/movie/<int:movie_id>/rate", methods=["POST"])
def submit_rating(movie_id):

    # Verify movie exists
    result = movies[movies["movie_id"] == movie_id]

    if result.empty:
        return "Movie not found", 404

    # Get submitted rating
    try:
        rating = int(request.form.get("rating", 0))
    except (ValueError, TypeError):
        flash("Please select a valid rating.")
        return redirect(url_for("movie_details", movie_id=movie_id))

    if rating < 1 or rating > 5:
        flash("Rating must be between 1 and 5.")
        return redirect(url_for("movie_details", movie_id=movie_id))

    # Identify visitor
    if "visitor_id" not in session:
        session["visitor_id"] = secrets.token_urlsafe(24)

    visitor_id = session["visitor_id"]

    # Save or update rating
    connection = sqlite3.connect(RATINGS_DB)

    connection.execute("""
        INSERT INTO user_ratings (movie_id, visitor_id, rating)
        VALUES (?, ?, ?)

        ON CONFLICT(movie_id, visitor_id)
        DO UPDATE SET
            rating = excluded.rating,
            updated_at = CURRENT_TIMESTAMP
    """, (movie_id, visitor_id, rating))

    connection.commit()
    connection.close()

    flash("Your rating has been saved successfully!")

    return redirect(url_for("movie_details", movie_id=movie_id))

# ==========================================
# MOVIE DETAILS PAGE
# ==========================================

@app.route("/movie/<int:movie_id>")
def movie_details(movie_id):

    result = movies[
        movies["movie_id"] == movie_id
    ]

    if result.empty:
        return "Movie not found", 404

    movie = result.iloc[0].to_dict()

    # ==========================================
    # USER RATING INFORMATION
    # ==========================================

    if "visitor_id" not in session:
        session["visitor_id"] = secrets.token_urlsafe(24)

    visitor_id = session["visitor_id"]

    connection = sqlite3.connect(RATINGS_DB)

    # Average rating and total votes
    rating_data = connection.execute("""
        SELECT AVG(rating), COUNT(*)
        FROM user_ratings
        WHERE movie_id = ?
    """, (movie_id,)).fetchone()

    average_user_rating = rating_data[0] or 0
    user_rating_count = rating_data[1]

    # Current visitor's rating
    user_rating_data = connection.execute("""
        SELECT rating
        FROM user_ratings
        WHERE movie_id = ? AND visitor_id = ?
    """, (movie_id, visitor_id)).fetchone()

    user_rating = user_rating_data[0] if user_rating_data else 0

    connection.close()
    # Selected movie poster
    movie["poster"] = get_movie_poster(
        movie["title"]
    )

    # Generate similar movies
    similar_movies = recommend_similar_movies(
        movie_id,
        number=10
    )

    # Fetch posters for similar movies
    for similar in similar_movies:

        similar["poster"] = get_movie_poster(
            similar["title"]
        )

    return render_template(
        "movie.html",
        movie=movie,
        similar_movies=similar_movies
    )


# ==========================================
# RECOMMENDATION SYSTEM
# ==========================================

# ==========================================
# RECOMMENDATION SYSTEM
# ==========================================

@app.route(
    "/recommendations",
    methods=["GET", "POST"]
)
def recommendations():

    recommended = []

    user_id = ""

    error = None

    if request.method == "POST":

        user_id = request.form.get(
            "user_id",
            ""
        ).strip()

        try:

            user_id = int(user_id)

            if user_id not in df["user_id"].values:

                error = (
                    "User ID not found. "
                    "Try another ID."
                )

            else:

                # Generate hybrid recommendations
                recommended = hybrid_recommend_movies(
                    user_id,
                    number=20
                )

                if not recommended:

                    error = (
                        "No recommendations available "
                        "for this user."
                    )

                else:

                    # Fetch movie posters
                    for movie in recommended:

                        movie["poster"] = get_movie_poster(
                            movie["title"]
                        )

        except ValueError:

            error = (
                "Please enter a valid numeric User ID."
            )

    return render_template(

        "recommendations.html",

        recommended=recommended,

        user_id=user_id,

        error=error

    )

# ==========================================
# ANALYTICS DASHBOARD
# ==========================================

@app.route("/analytics")
def analytics():

    data = df.copy()

    # Unique movies
    unique_movies = data.drop_duplicates(
        subset="movie_id"
    ).copy()

    # Genre columns
    genre_columns = [

        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"

    ]

    genre_columns = [

        genre for genre in genre_columns

        if genre in data.columns

    ]

    # --------------------------------------
    # 1. RATING DISTRIBUTION
    # --------------------------------------

    rating_distribution = (
        data["rating"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------
    # 2. GENRE POPULARITY
    # --------------------------------------

    genre_popularity = (
        unique_movies[genre_columns]
        .sum()
        .sort_values(ascending=False)
    )

    # --------------------------------------
    # 3. AVERAGE RATING BY GENRE
    # --------------------------------------

    average_genre_ratings = {}

    for genre in genre_columns:

        genre_movie_ids = unique_movies.loc[
            unique_movies[genre] == 1,
            "movie_id"
        ]

        genre_ratings = data[
            data["movie_id"].isin(
                genre_movie_ids
            )
        ]

        average_genre_ratings[genre] = round(
            genre_ratings["rating"].mean(),
            2
        )

    # --------------------------------------
    # 4. TOP 10 HIGHEST-RATED MOVIES
    # --------------------------------------

    top_movie_stats = (
        data.groupby(
            ["movie_id", "title"]
        )
        .agg(
            average_rating=("rating", "mean"),
            total_ratings=("rating", "count")
        )
        .reset_index()
    )

    top_movies = (

        top_movie_stats[
            top_movie_stats["total_ratings"] >= 20
        ]

        .sort_values(
            by=[
                "average_rating",
                "total_ratings"
            ],
            ascending=[False, False]
        )

        .head(10)

    )

    top_movie_labels = top_movies[
        "title"
    ].tolist()

    top_movie_values = (
        top_movies["average_rating"]
        .round(2)
        .tolist()
    )

    # --------------------------------------
    # 5. RATINGS BY GENDER
    # --------------------------------------

    gender_ratings = (
        data.dropna(subset=["gender"])
        .groupby("gender")["rating"]
        .mean()
        .sort_index()
    )

    gender_labels = gender_ratings.index.tolist()

    gender_values = (
        gender_ratings.round(2).tolist()
    )

    # --------------------------------------
    # DASHBOARD STATISTICS
    # --------------------------------------

    total_movies = unique_movies[
        "movie_id"
    ].nunique()

    total_ratings = len(data)

    total_users = data[
        "user_id"
    ].nunique()

    # --------------------------------------
    # FASTCLUSTER RESULTS (OPTIONAL)
    # --------------------------------------

    fastcluster_root = os.path.join(
        BASE_DIR, "Data_Mining", "Fastcluster"
    )

    def find_fastcluster_file(filename):
        if not os.path.isdir(fastcluster_root):
            return None
        for root, _, files in os.walk(fastcluster_root):
            if filename in files:
                return os.path.join(root, filename)
        return None

    fastcluster_metrics = []
    fastcluster_profiles = []
    fastcluster_genres = []

    metrics_file = find_fastcluster_file(
        "fastcluster_evaluation_metrics.csv"
    )
    profiles_file = find_fastcluster_file(
        "fastcluster_cluster_profiles.csv"
    )
    genres_file = find_fastcluster_file(
        "fastcluster_genre_profiles.csv"
    )

    try:
        if metrics_file:
            metrics_data = pd.read_csv(metrics_file)
            fastcluster_metrics = metrics_data.fillna("").to_dict("records")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        fastcluster_metrics = []

    try:
        if profiles_file:
            profiles_data = pd.read_csv(profiles_file)
            # The cluster number may be saved as the CSV index.
            if "cluster" not in profiles_data.columns:
                first_col = profiles_data.columns[0]
                if str(first_col).lower().startswith("unnamed"):
                    profiles_data = profiles_data.rename(
                        columns={first_col: "cluster"}
                    )
            fastcluster_profiles = profiles_data.fillna("").to_dict("records")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        fastcluster_profiles = []

    try:
        if genres_file:
            genres_data = pd.read_csv(genres_file)
            if "cluster" not in genres_data.columns:
                first_col = genres_data.columns[0]
                if str(first_col).lower().startswith("unnamed"):
                    genres_data = genres_data.rename(
                        columns={first_col: "cluster"}
                    )
            fastcluster_genres = genres_data.fillna("").to_dict("records")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        fastcluster_genres = []

    fastcluster_chart_dir = os.path.join(
        BASE_DIR, "static", "images", "analytics", "Fastcluster"
    )
    fastcluster_chart_names = [
        "fastcluster_dendrogram.png",
        "fastcluster_cluster_distribution.png",
        "fastcluster_genre_profiles.png"
    ]
    fastcluster_charts = [
        name for name in fastcluster_chart_names
        if os.path.exists(os.path.join(fastcluster_chart_dir, name))
    ]

    # --------------------------------------
    # RENDER ANALYTICS PAGE
    # --------------------------------------

    return render_template(

        "analytics.html",

        # Chart 1
        rating_labels=rating_distribution.index.tolist(),

        rating_values=rating_distribution.values.tolist(),

        # Chart 2
        genre_labels=genre_popularity.index.tolist(),

        genre_values=genre_popularity.values.tolist(),

        # Chart 3
        average_labels=list(
            average_genre_ratings.keys()
        ),

        average_values=list(
            average_genre_ratings.values()
        ),

        # Chart 4
        top_movie_labels=top_movie_labels,

        top_movie_values=top_movie_values,

        # Chart 5
        gender_labels=gender_labels,

        gender_values=gender_values,

        # Statistics
        total_movies=total_movies,

        total_ratings=total_ratings,

        total_users=total_users,

        # Fastcluster analysis (kept separate from WEKA)
        fastcluster_metrics=fastcluster_metrics,
        fastcluster_profiles=fastcluster_profiles,
        fastcluster_genres=fastcluster_genres,
        fastcluster_charts=fastcluster_charts

    )

    return render_template(
    "movie.html",
    movie=movie,
    similar_movies=similar_movies,
    user_rating=user_rating,
    average_user_rating=round(average_user_rating, 1),
    user_rating_count=user_rating_count
)
# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)