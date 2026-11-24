import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 55)
print("       CINESCOPE - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# Load processed dataset
df = pd.read_csv(
    "Dataset/Processed/cinescope_master.csv",
    sep=";",
    encoding="cp1252",
    encoding_errors="replace"
)
# Create output folder
os.makedirs("Visualizations", exist_ok=True)

sns.set_theme(style="whitegrid")

# Basic dataset information
print("\nDATASET OVERVIEW")
print("-" * 40)

print("Total Ratings:", len(df))
print("Unique Users:", df["user_id"].nunique())
print("Unique Movies:", df["movie_id"].nunique())

print("\nRating Statistics:")
print(df["rating"].describe())

print("\nTop 10 Most Rated Movies:")
print(
    df.groupby("title")["rating"]
    .count()
    .sort_values(ascending=False)
    .head(10)
)


# VISUALIZATION 1: Rating Distribution

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="rating"
)

plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.tight_layout()
plt.savefig("Visualizations/01_rating_distribution.png")
plt.close()


# VISUALIZATION 2: Genre Popularity

genre_columns = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

genre_counts = df[genre_columns].sum().sort_values(
    ascending=False
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=genre_counts.values,
    y=genre_counts.index
)

plt.title("Movie Genre Popularity")
plt.xlabel("Number of Ratings")
plt.ylabel("Genre")

plt.tight_layout()
plt.savefig("Visualizations/02_genre_popularity.png")
plt.close()


# VISUALIZATION 3: Average Rating by Genre

genre_ratings = {}

for genre in genre_columns:
    genre_data = df[df[genre] == 1]
    genre_ratings[genre] = genre_data["rating"].mean()

genre_avg = pd.Series(genre_ratings).sort_values(
    ascending=False
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=genre_avg.values,
    y=genre_avg.index
)

plt.title("Average Rating by Movie Genre")
plt.xlabel("Average Rating")
plt.ylabel("Genre")

plt.tight_layout()
plt.savefig("Visualizations/03_average_rating_by_genre.png")
plt.close()


# VISUALIZATION 4: Ratings by Gender

gender_ratings = df.groupby(
    "gender"
)["rating"].mean().reset_index()

plt.figure(figsize=(7, 5))

sns.barplot(
    data=gender_ratings,
    x="gender",
    y="rating"
)

plt.title("Average Movie Ratings by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Rating")

plt.tight_layout()
plt.savefig("Visualizations/04_gender_ratings.png")
plt.close()


# VISUALIZATION 5: Top 10 Movies by Rating Count

top_movies = (
    df.groupby("title")["rating"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_movies.values,
    y=top_movies.index
)

plt.title("Top 10 Most Rated Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie")

plt.tight_layout()
plt.savefig("Visualizations/05_top_movies.png")
plt.close()


print("\n" + "=" * 55)
print("EDA COMPLETED SUCCESSFULLY!")
print("Five visualisations generated.")
print("Check the Visualizations folder.")
print("=" * 55)