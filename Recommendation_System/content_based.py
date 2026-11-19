import pandas as pd
import math
import re
from collections import Counter, defaultdict

print("=" * 60)
print("       CINESCOPE - CONTENT BASED RECOMMENDATION")
print("=" * 60)

# Load dataset
df = pd.read_csv("Dataset/Processed/cinescope_master.csv")

# Keep one record per movie
movies = df.drop_duplicates(
    subset="movie_id"
).copy().reset_index(drop=True)

movies["genres"] = movies["genres"].fillna("")
movies["title"] = movies["title"].fillna("")

print("\nMovies loaded:", len(movies))


# STEP 1: Tokenize movie information

def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


documents = []

for _, movie in movies.iterrows():

    text = movie["title"] + " " + movie["genres"]

    documents.append(tokenize(text))


# STEP 2: Calculate TF-IDF

document_frequency = Counter()

for document in documents:
    document_frequency.update(set(document))


total_documents = len(documents)

tfidf_vectors = []

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

    tfidf_vectors.append(vector)


# STEP 3: Calculate cosine similarity

def cosine_similarity(vector1, vector2):

    common_terms = set(vector1) & set(vector2)

    dot_product = sum(
        vector1[term] * vector2[term]
        for term in common_terms
    )

    magnitude1 = math.sqrt(
        sum(value ** 2 for value in vector1.values())
    )

    magnitude2 = math.sqrt(
        sum(value ** 2 for value in vector2.values())
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


print("\nTF-IDF model created successfully!")
print("Cosine similarity calculation ready!")


# STEP 4: Recommendation function

def recommend_movies(movie_title, number=10):

    matches = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:

        print("\nMovie not found!")
        return

    movie_index = matches.index[0]

    target_vector = tfidf_vectors[movie_index]

    similarity_scores = []

    for index, vector in enumerate(tfidf_vectors):

        if index == movie_index:
            continue

        score = cosine_similarity(
            target_vector,
            vector
        )

        similarity_scores.append((index, score))

    similarity_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRecommendations for:")
    print(movies.iloc[movie_index]["title"])

    print("\n" + "-" * 60)

    for rank, (index, score) in enumerate(
        similarity_scores[:number],
        start=1
    ):

        recommended_movie = movies.iloc[index]

        print(
            f"{rank}. {recommended_movie['title']}"
            f" | Similarity: {score:.4f}"
        )


# STEP 5: Interactive system

while True:

    print("\nEnter a movie name.")
    print("Type 'exit' to stop.")

    user_input = input("\nMovie: ").strip()

    if user_input.lower() == "exit":

        print("\nThank you for using CineScope!")
        break

    if user_input:

        recommend_movies(user_input)