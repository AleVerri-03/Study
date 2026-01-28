# Rating cinema films, many users and poor number of evaluations -> so sparse matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

dataset = pd.read_csv(
    "./movielens.csv",
    sep="\t",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
)
dataset

n_people = np.unique(dataset.user_id).size # Nuber of people
n_movies = np.unique(dataset.item_id).size
n_ratings = len(dataset)

print(f"{n_people} people")
print(f"{n_movies} movies")
print(f"{n_ratings} ratings")

# Shuffle data
np.random.seed(1)
idxs = np.arange(n_ratings) # index array with all the evaluations
np.random.shuffle(idxs)
rows_dupes = dataset.user_id[idxs] # shuffle row and column using these indexs
cols_dupes = dataset.item_id[idxs]
vals = dataset.rating[idxs]

# Free rows and columns (example users that doesn't evaluate)
_, rows = np.unique(rows_dupes, return_inverse=True)
_, cols = np.unique(cols_dupes, return_inverse=True)

print(rows.min(), rows.max(), n_people)
print(cols.min(), cols.max(), n_movies)

# Split dataset into training subset and testing rating
training_data = int(0.8 * n_ratings)
rows_train = rows[:training_data]
cols_train = cols[:training_data]
vals_train = vals[:training_data]
# ---
rows_test = rows[training_data:]
cols_test = cols[training_data:]
vals_test = vals[training_data:]

# Sparse matrix to check if E j-movie for i-user
X_sparse = csr_matrix((vals_train, (rows_train, cols_train)), shape=(n_people, n_movies))
X_full = X_sparse.toarray()

# Trivial (banale) solution
avg_ratings = np.empty((n_people, ))
for user_id in range(n_people):
    avg_ratings[user_id] = np.mean(vals_train[rows_train == user_id])

vals_trivial = avg_ratings[rows_test] # for every prevision it returns the avg value of that user

errors_trivial = vals_test - vals_trivial

RMSE_trivial = np.sqrt(np.mean(errors_trivial**2))
rho_trivial = pearsonr(vals_test, vals_trivial)[0]
print(f"RMSE: {RMSE_trivial:1.3f}")
print(f"rho : {rho_trivial:1.3f}")

n_max_iter = 100
threshold = 100.0
increment_tol = 1e-6

RMSE_list = list()
rho_list = list()

A = X_full.copy()

print("Iter | Increment |  RMSE |  Corr ")
for i in range(n_max_iter):
    A_old = A.copy()
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    s[s < threshold] = 0
    A = U * np.diag(s) @ VT

    A[rows_train, cols_train] = vals_train # Re-Write training set
    increment = np.linalg.norm(A - A_old)

    vals_predicted = A[rows_test, cols_test]
    errors = vals_test - vals_predicted

    RMSE_list.append(np.sqrt(np.mean(errors**2))) # average quadratic error
    rho_list.append(pearsonr(vals_test, vals_predicted)[0]) # Pearson correlation

    print(f"{i+1:04} | {increment:.3e} | {RMSE_list[-1]:1.3f} | {rho_list[-1]:1.3f}")
    if increment < increment_tol:
        break

# Plot RMSE and rho
fig, axs = plt.subplot(2, 1, figsize=(12, 16))
axs[0].plot(RMSE_list, "o-")
axs[0].axhline(RMSE_trivial, color="red")
axs[0].legend(["RMSE", "RMSE trivial"])

axs[1].plot(rho_list, "o-")
axs[1].axhline(rho_trivial, color="red")
axs[1].legend([r"$\rho$", r"$rho$ trivial"])