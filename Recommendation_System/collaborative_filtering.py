import pandas as pd
import numpy as np

print("=" * 60)
print("     CINESCOPE - COLLABORATIVE FILTERING")
print("=" * 60)

# Load dataset
df = pd.read_csv("Dataset/Processed/cinescope_master.csv")

# Create user-movie rating matrix
print("\nCreating user-movie rating matrix...")

ratings_matrix = df.pivot_table(
    index="movie_id",
    columns="user_id",
    values="rating"
).fillna(0)

# Store movie information
movies = df.drop_duplicates(
    subset="movie_id"
).set_index("movie_id")

# Convert matrix to NumPy array
rating_values = ratings_matrix.to_numpy(dtype=np.float32)

print("Rating matrix created:", ratings_matrix.shape)


# STEP 1: Calculate item similarity

print("\nCalculating movie similarities...")

dot_product = rating_values @ rating_values.T

norms = np.sqrt(
    np.sum(rating_values ** 2, axis=1)
)

denominator = np.outer(norms, norms)

similarity_matrix = np.divide(
    dot_product,
    denominator,
    out=np.zeros_like(dot_product),
    where=denominator != 0
)

print("Movie similarity calculated successfully!")


# STEP 2: Recommendation function

def recommend_for_user(user_id, number=10):

    if user_id not in ratings_matrix.columns:

        print("\nUser ID not found!")
        print("Enter a user ID between 1 and 943.")
        return

    user_ratings = ratings_matrix[user_id]

    watched_movies = user_ratings[
        user_ratings > 0
    ]

    if watched_movies.empty:

        print("\nThis user has no movie ratings.")
        return

    # Get positions of movies rated by user
    watched_positions = [
        ratings_matrix.index.get_loc(movie_id)
        for movie_id in watched_movies.index
    ]

    # Calculate weighted recommendation scores
    user_rating_values = watched_movies.values

    scores = (
        similarity_matrix[:, watched_positions]
        @ user_rating_values
    )

    # Calculate similarity weights
    weights = similarity_matrix[:, watched_positions].sum(axis=1)

    predicted_ratings = np.divide(
        scores,
        weights,
        out=np.zeros_like(scores),
        where=weights != 0
    )

    # Exclude movies already watched
    watched_ids = set(watched_movies.index)

    results = []

    for position, movie_id in enumerate(ratings_matrix.index):

        if movie_id in watched_ids:
            continue

        if weights[position] == 0:
            continue

        results.append((
            movie_id,
            predicted_ratings[position]
        ))

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRecommendations for User:", user_id)
    print("-" * 60)

    for rank, (movie_id, score) in enumerate(
        results[:number],
        start=1
    ):

        title = movies.loc[movie_id, "title"]

        print(
            f"{rank}. {title} | Predicted Rating: {score:.2f}"
        )


# STEP 3: Interactive system

while True:

    print("\nEnter a User ID (1-943).")
    print("Type 'exit' to stop.")

    user_input = input("\nUser ID: ").strip()

    if user_input.lower() == "exit":

        print("\nThank you for using CineScope!")
        break

    try:

        user_id = int(user_input)

        recommend_for_user(user_id)

    except ValueError:

        print("Please enter a valid numeric User ID.")