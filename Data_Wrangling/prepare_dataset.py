import pandas as pd
import os

print("=" * 55)
print("       CINESCOPE - DATA PREPROCESSING")
print("=" * 55)

# Dataset paths
RAW_PATH = "Dataset/Raw/ml-100k"
OUTPUT_PATH = "Dataset/Processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Column names
rating_columns = [
    "user_id",
    "movie_id",
    "rating",
    "timestamp"
]

user_columns = [
    "user_id",
    "age",
    "gender",
    "occupation",
    "zip_code"
]

genre_columns = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

movie_columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url"
] + genre_columns


# 1. Load ratings
print("\nLoading ratings dataset...")

ratings = pd.read_csv(
    f"{RAW_PATH}/u.data",
    sep="\t",
    names=rating_columns
)


# 2. Load movies
print("Loading movie dataset...")

movies = pd.read_csv(
    f"{RAW_PATH}/u.item",
    sep="|",
    names=movie_columns,
    encoding="latin-1"
)


# 3. Load users
print("Loading user dataset...")

users = pd.read_csv(
    f"{RAW_PATH}/u.user",
    sep="|",
    names=user_columns
)


# 4. Merge datasets
print("\nCombining datasets...")

master = ratings.merge(
    movies,
    on="movie_id",
    how="left"
)

master = master.merge(
    users,
    on="user_id",
    how="left"
)


# 5. Create readable genre column
print("Processing movie genres...")

def get_genres(row):
    return ", ".join(
        genre for genre in genre_columns
        if row[genre] == 1
    )

master["genres"] = master.apply(get_genres, axis=1)


# 6. Convert timestamp
master["rating_date"] = pd.to_datetime(
    master["timestamp"],
    unit="s"
)


# 7. Extract release year
master["release_year"] = pd.to_datetime(
    master["release_date"],
    errors="coerce"
).dt.year


# 8. Check data quality
print("\nChecking data quality...")

print("Total records:", len(master))
print("Missing values:", master.isnull().sum().sum())
print("Duplicate records:", master.duplicated().sum())


# 9. Save processed dataset
output_file = os.path.join(
    OUTPUT_PATH,
    "cinescope_master.csv"
)

master.to_csv(output_file, index=False)


# 10. Display summary
print("\n" + "=" * 55)
print("         CINESCOPE DATASET SUMMARY")
print("=" * 55)

print("Total Ratings:", len(ratings))
print("Unique Users:", ratings["user_id"].nunique())
print("Unique Movies:", movies["movie_id"].nunique())
print("Total Genres:", len(genre_columns))

print("\nMaster Dataset Shape:", master.shape)

print("\nSample Records:")
print(master[[
    "user_id",
    "movie_id",
    "title",
    "rating",
    "genres",
    "age",
    "gender"
]].head())

print("\nDataset saved successfully!")
print("Location:", output_file)

print("\nCINESCOPE DATA PREPROCESSING COMPLETED!")