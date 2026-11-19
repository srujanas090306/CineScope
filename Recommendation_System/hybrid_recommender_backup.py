import pandas as pd
import numpy as np
import math
import re
from collections import Counter

print("=" * 65)
print("          CINESCOPE - HYBRID RECOMMENDATION SYSTEM")
print("=" * 65)

# Load dataset
df = pd.read_csv("Dataset/Processed/cinescope_master.csv")

# Movie information
movies = (
    df.drop_duplicates(subset="movie_id")
    .set_index("movie_id")
)

movies["genres"] = movies["genres"].fillna("")
movies["title"] = movies["title"].fillna("")

# User-movie rating matrix
rating_matrix = df.pivot_table(
    index="movie_id",
    columns="user_id",
    values="rating"
).fillna(0)

movie_ids = rating_matrix.index.to_numpy()

print("\nMovies loaded:", len(movies))
print("Users loaded:", len(rating_matrix.columns))


# -------------------------------------------------
# PART 1: CONTENT-BASED FILTERING
# -------------------------------------------------

print("\nBuilding content-based model...")


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


documents = []

for movie_id in movie_ids:

    movie = movies.loc[movie_id]

    text = movie["title"] + " " + movie["genres"]

    documents.append(tokenize(text))


# Calculate document frequency
document_frequency = Counter()

for document in documents:
    document_frequency.update(set(document))


total_documents = len(documents)

content_vectors = []

for document in documents:

    term_counts = Counter(document)
    total_terms = len(document)

    vector = {}

    for term, count in term_counts.items():

        tf = count / total_terms

        idf = math.log(
            (1 + total_documents) /
            (1 + document_frequency[term])
        ) + 1

        vector[term] = tf * idf

    content_vectors.append(vector)


def cosine_similarity(vector1, vector2):

    common = set(vector1) & set(vector2)

    dot = sum(
        vector1[t] * vector2[t]
        for t in common
    )

    norm1 = math.sqrt(
        sum(v ** 2 for v in vector1.values())
    )

    norm2 = math.sqrt(
        sum(v ** 2 for v in vector2.values())
    )

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot / (norm1 * norm2)


# Create content similarity matrix
content_similarity = np.zeros(
    (len(movie_ids), len(movie_ids))
)

for i in range(len(movie_ids)):

    for j in range(i, len(movie_ids)):

        score = cosine_similarity(
            content_vectors[i],
            content_vectors[j]
        )

        content_similarity[i, j] = score
        content_similarity[j, i] = score


print("Content-based model ready!")


# -------------------------------------------------
# PART 2: COLLABORATIVE FILTERING
# -------------------------------------------------

print("\nBuilding collaborative filtering model...")

ratings = rating_matrix.to_numpy(dtype=np.float32)

dot_product = ratings @ ratings.T

norms = np.sqrt(
    np.sum(ratings ** 2, axis=1)
)

denominator = np.outer(norms, norms)

collaborative_similarity = np.divide(
    dot_product,
    denominator,
    out=np.zeros_like(dot_product),
    where=denominator != 0
)

print("Collaborative model ready!")


# -------------------------------------------------
# PART 3: HYBRID RECOMMENDATION
# -------------------------------------------------

def normalize(scores):

    scores = np.asarray(scores, dtype=float)

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def recommend_movies(user_id, number=10):

    if user_id not in rating_matrix.columns:

        print("\nUser ID not found!")
        return

    user_ratings = rating_matrix[user_id]

    watched = user_ratings[user_ratings > 0]

    if watched.empty:

        print("\nNo ratings found for this user.")
        return

    # Give more weight to movies rated above 3
    preference_weights = np.maximum(
        watched.to_numpy(dtype=float) - 3,
        0
    )

    if preference_weights.sum() == 0:
        preference_weights = watched.to_numpy(dtype=float)

    watched_positions = [
        rating_matrix.index.get_loc(movie_id)
        for movie_id in watched.index
    ]

    # Content-based scores
    content_scores = (
        content_similarity[:, watched_positions]
        @ preference_weights
    ) / preference_weights.sum()

    # Collaborative scores
    similarity_values = collaborative_similarity[
        :, watched_positions
    ]

    collaborative_scores = (
        similarity_values @ preference_weights
    )

    similarity_weights = np.abs(
        similarity_values
    ).sum(axis=1)

    collaborative_scores = np.divide(
        collaborative_scores,
        similarity_weights,
        out=np.zeros_like(collaborative_scores),
        where=similarity_weights != 0
    )

    # Normalize both scores
    content_scores = normalize(content_scores)

    collaborative_scores = normalize(
        collaborative_scores
    )

    # Hybrid weighted score
    hybrid_scores = (
        0.5 * content_scores +
        0.5 * collaborative_scores
    )

    watched_ids = set(watched.index)

    recommendations = []

    for position, movie_id in enumerate(movie_ids):

        if movie_id in watched_ids:
            continue

        recommendations.append((
            movie_id,
            content_scores[position],
            collaborative_scores[position],
            hybrid_scores[position]
        ))

    recommendations.sort(
        key=lambda x: x[3],
        reverse=True
    )

    print("\n" + "=" * 65)
    print("RECOMMENDATIONS FOR USER:", user_id)
    print("=" * 65)

    for rank, (movie_id, content, collaborative, hybrid) in enumerate(
        recommendations[:number],
        start=1
    ):

        title = movies.loc[movie_id, "title"]

        print(f"\n{rank}. {title}")
        print(f"   Content Score:       {content:.4f}")
        print(f"   Collaborative Score: {collaborative:.4f}")
        print(f"   Hybrid Score:        {hybrid:.4f}")


# -------------------------------------------------
# PART 4: INTERACTIVE SYSTEM
# -------------------------------------------------

while True:

    print("\nEnter a User ID between 1 and 943.")
    print("Type 'exit' to stop.")

    choice = input("\nUser ID: ").strip()

    if choice.lower() == "exit":

        print("\nThank you for using CineScope!")
        break

    try:

        user_id = int(choice)

        recommend_movies(user_id)

    except ValueError:

        print("Please enter a valid numeric User ID.")