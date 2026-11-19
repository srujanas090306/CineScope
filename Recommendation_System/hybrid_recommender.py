import pandas as pd
import numpy as np
import math
import re
from collections import Counter


# ==========================================
# CINESCOPE HYBRID RECOMMENDATION SYSTEM
# ==========================================

DATASET_PATH = "Dataset/Processed/cinescope_master.csv"

print("Loading CineScope Hybrid Recommendation Model...")


# Load dataset
df = pd.read_csv(DATASET_PATH)


# Movie information
movies = (
    df.drop_duplicates(subset="movie_id")
    .set_index("movie_id")
    .copy()
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


# ==========================================
# PART 1: CONTENT-BASED FILTERING
# ==========================================

print("Building content-based model...")


def tokenize(text):

    return re.findall(
        r"[a-z0-9]+",
        str(text).lower()
    )


documents = []

for movie_id in movie_ids:

    movie = movies.loc[movie_id]

    text = movie["title"] + " " + movie["genres"]

    documents.append(tokenize(text))


document_frequency = Counter()

for document in documents:

    document_frequency.update(set(document))


total_documents = len(documents)

content_vectors = []

for document in documents:

    term_counts = Counter(document)

    total_terms = max(len(document), 1)

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


print("Calculating content similarity...")

content_similarity = np.zeros(
    (len(movie_ids), len(movie_ids)),
    dtype=np.float32
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


# ==========================================
# PART 2: COLLABORATIVE FILTERING
# ==========================================

print("Building collaborative filtering model...")


ratings = rating_matrix.to_numpy(
    dtype=np.float32
)

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


# ==========================================
# PART 3: NORMALIZATION
# ==========================================

def normalize(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    minimum = scores.min()

    maximum = scores.max()

    if maximum == minimum:

        return np.zeros_like(scores)

    return (
        (scores - minimum) /
        (maximum - minimum)
    )


# ==========================================
# PART 4: HYBRID RECOMMENDATION
# ==========================================

def recommend_movies(user_id, number=20):

    user_id = int(user_id)

    if user_id not in rating_matrix.columns:

        return []

    user_ratings = rating_matrix[user_id]

    watched = user_ratings[
        user_ratings > 0
    ]

    if watched.empty:

        return []

    # Give more importance to movies rated above 3
    preference_weights = np.maximum(
        watched.to_numpy(dtype=float) - 3,
        0
    )

    if preference_weights.sum() == 0:

        preference_weights = watched.to_numpy(
            dtype=float
        )

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
        :,
        watched_positions
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


    # Normalize scores
    content_scores = normalize(
        content_scores
    )

    collaborative_scores = normalize(
        collaborative_scores
    )


    # Hybrid weighted score
    hybrid_scores = (

        0.5 * content_scores +

        0.5 * collaborative_scores

    )


    # Exclude already watched movies
    watched_ids = set(watched.index)

    recommendations = []

    for position, movie_id in enumerate(movie_ids):

        if movie_id in watched_ids:

            continue

        recommendations.append({

            "movie_id": int(movie_id),

            "title": movies.loc[
                movie_id,
                "title"
            ],

            "genres": movies.loc[
                movie_id,
                "genres"
            ],

            "content_score": round(
                float(content_scores[position]),
                4
            ),

            "collaborative_score": round(
                float(collaborative_scores[position]),
                4
            ),

            "hybrid_score": round(
                float(hybrid_scores[position]),
                4
            )

        })


    recommendations.sort(

        key=lambda x: x["hybrid_score"],

        reverse=True

    )


    return recommendations[:number]


print("CineScope Hybrid Recommendation Model READY!")

# ==========================================
# PART 5: SIMILAR MOVIES RECOMMENDATION
# ==========================================

def recommend_similar_movies(movie_id, number=10):

    try:
        movie_id = int(movie_id)
    except (ValueError, TypeError):
        return []

    if movie_id not in movie_ids:
        return []

    # Get selected movie position
    movie_position = np.where(
        movie_ids == movie_id
    )[0][0]

    # Get similarity scores
    similarity_scores = content_similarity[
        movie_position
    ]

    # Sort movies by similarity
    similar_positions = np.argsort(
        similarity_scores
    )[::-1]

    recommendations = []

    for position in similar_positions:

        recommended_id = int(movie_ids[position])

        # Exclude selected movie
        if recommended_id == movie_id:
            continue

        recommendations.append({

            "movie_id": recommended_id,

            "title": movies.loc[
                recommended_id,
                "title"
            ],

            "genres": movies.loc[
                recommended_id,
                "genres"
            ],

            "similarity_score": round(
                float(similarity_scores[position]),
                4
            )

        })

        if len(recommendations) >= number:
            break

    return recommendations