import pandas as pd

from helpers.test_cases import TestCases

# # Define file paths (adjust if your files are elsewhere)
# ratings_file = "data/movielens-1m/ratings.dat"
# users_file   = "data/movielens-1m/users.dat"
# movies_file  = "data/movielens-1m/movies.dat"

# # Read the ratings data
# ratings = pd.read_csv(
#     ratings_file,
#     sep="::",
#     engine="python",
#     names=["UserID", "MovieID", "Rating", "Timestamp"],
#     encoding="latin-1"
# )

# # Read the users data
# users = pd.read_csv(
#     users_file,
#     sep="::",
#     engine="python",
#     names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
#     encoding="latin-1"
# )

# # Read the movies data
# movies = pd.read_csv(
#     movies_file,
#     sep="::",
#     engine="python",
#     names=["MovieID", "Title", "Genres"],
#     encoding="latin-1"
# )

# # Merge ratings with users on UserID
# merged = pd.merge(ratings, users, on="UserID")

# # Merge the above with movies on MovieID
#full_table = pd.merge(merged, movies, on="MovieID")
# full_table = "/data/MATHE/output_table.csv"

# # Save to CSV
# #full_table.to_csv("movielens-1m-full.csv", index=False)

# print("Full MovieLens table saved as 'movielens-1m-full.csv'")
# print("Shape of table:", full_table.shape)
# print("Columns:", full_table.columns.tolist())
# print(full_table.head())



############################################33
# import pandas as pd

# movielens = pd.read_csv("movielens-1m-full.csv")

# sampled_movielens = movielens.sample(n=200_000, random_state=42)  # Set a seed for reproducibility

# sampled_movielens.to_csv("movielens-200k.csv", index=False)

# print("Shape:", sampled_movielens.shape)


test_cases = TestCases()
T_input, UR = test_cases.get_case(22) 
print(UR)

