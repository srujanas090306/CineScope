import pandas as pd
import numpy as np
import os

# ==========================================
# CINESCOPE - WEKA DATASET PREPARATION
# ==========================================

INPUT_FILE = "Dataset/Processed/cinescope_master.csv"

OUTPUT_FILE = (
    "Data_Mining/WEKA/movie_rating_classification.arff"
)

print("=" * 55)
print(" CINESCOPE - WEKA DATA PREPARATION")
print("=" * 55)


# Load dataset
df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    encoding="cp1252",
    encoding_errors="replace"
)
print("\nOriginal dataset:", df.shape)


# ==========================================
# SELECT FEATURES
# ==========================================

genre_columns = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

columns = [
    "age",
    "gender",
    "occupation",
    "release_year"
] + genre_columns + ["rating"]

data = df[columns].copy()


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

data["release_year"] = data["release_year"].fillna(
    data["release_year"].median()
)

data["gender"] = data["gender"].fillna("Unknown")
data["occupation"] = data["occupation"].fillna("Unknown")


# ==========================================
# CREATE RATING CLASSES
# ==========================================

def classify_rating(rating):

    if rating <= 2:
        return "Low"

    elif rating == 3:
        return "Medium"

    return "High"


data["rating_class"] = data["rating"].apply(
    classify_rating
)

data.drop(columns=["rating"], inplace=True)


# ==========================================
# CONVERT DATA TYPES
# ==========================================

for column in genre_columns:
    data[column] = data[column].fillna(0).astype(int)

data["age"] = data["age"].astype(int)
data["release_year"] = data["release_year"].astype(int)


# ==========================================
# GENERATE ARFF FILE
# ==========================================

def arff_quote(value):

    value = str(value)

    value = value.replace("\\", "\\\\")
    value = value.replace("'", "\\'")

    return "'" + value + "'"


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("@RELATION cinescope_movie_ratings\n\n")

    file.write("@ATTRIBUTE age NUMERIC\n")
    file.write("@ATTRIBUTE gender {'F','M','Unknown'}\n")

    occupations = sorted(
        data["occupation"].astype(str).unique()
    )

    occupation_values = ",".join(
        arff_quote(value)
        for value in occupations
    )

    file.write(
        f"@ATTRIBUTE occupation {{{occupation_values}}}\n"
    )

    file.write("@ATTRIBUTE release_year NUMERIC\n")

    for genre in genre_columns:
        file.write(f"@ATTRIBUTE {genre} NUMERIC\n")

    file.write(
        "@ATTRIBUTE rating_class {Low,Medium,High}\n\n"
    )

    file.write("@DATA\n")

    for row in data.itertuples(index=False, name=None):

        values = []

        for index, value in enumerate(row):

            if index == 1 or index == 2:
                values.append(arff_quote(value))

            else:
                values.append(str(value))

        file.write(",".join(values) + "\n")


# ==========================================
# SUMMARY
# ==========================================

print("\nWEKA dataset created successfully!")

print("Records:", len(data))

print("\nRating class distribution:")

print(data["rating_class"].value_counts())

print("\nOutput file:")

print(OUTPUT_FILE)

print("\nCINESCOPE WEKA DATASET READY!")