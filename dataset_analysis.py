import pandas as pd
import os

# Dataset location
DATA_PATH = "Dataset/Raw/ml-100k"

# Load ratings dataset
ratings = pd.read_csv(
    os.path.join(DATA_PATH, "u.data"),
    sep="\t",
    names=["User_ID", "Movie_ID", "Rating", "Timestamp"]
)

# Display basic information
print("\n========== CINESCOPE ==========")

print("\nDataset loaded successfully!")

print("\nFirst 10 Records:")
print(ratings.head(10))

print("\nDataset Information:")
print(ratings.info())

print("\nStatistical Summary:")
print(ratings.describe())

print("\nTotal Ratings:", len(ratings))

print("Total Users:", ratings["User_ID"].nunique())

print("Total Movies:", ratings["Movie_ID"].nunique())

print("\nRating Distribution:")
print(ratings["Rating"].value_counts().sort_index())

print("\n================================")