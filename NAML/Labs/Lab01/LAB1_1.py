import scipy.linalg as la
import numpy as np
np.random.seed(0)

A = np.random.rand(5,4)
A

U, s, VT = np.linalg.svd(A)
#U, s, VT = la.svd(A)
print('U shape: ', U.shape)
print('s shape: ', s.shape)
print('VT shape: ', VT.shape)

S = np.zeros(A.shape)
for i in range(len(s)):
    S[i, i] = s[i]
S

S = la.diagsvd(s, A.shape[0], A.shape[1])
S

A_svd = np.matmul(U, np.matmul(S,VT))
# equivalently: A_svd = U @ S @ VT
print(f"err: {(la.norm(A - A_svd) / la.norm(A))}")

## NOW USING REDUCED SVD
U, s, VT = la.svd(A, full_matrices=False)
print('U shape: ', U.shape)
print('s shape: ', s.shape)
print('VT shape: ', VT.shape)

S = np.diag(s)
S

A_svd = np.matmul(U, np.matmul(S,VT))
print(f"err: {la.norm(A - A_svd) / la.norm(A)}")

# [Summary]
# In this lab, we explored SVD (Singular Value Decomposition) using NumPy and SciPy, computing both full and reduced forms.
# We verified the decomposition by reconstructing the original matrix and measuring the relative reconstruction error.

# --------------------------------------------

# Misure time taken for this operation for matrix A (larger)

import time

A = np.random.rand(1000, 1500)
U, s, VT = la.svd(A, full_matrices=False)
S = np.diag(s)

start_time = time.time()

# Loop time
A_reconstructed_loop = np.zeros_like(A)
for i in range(len(s)):
    A_reconstructed_loop += s[i] * np.outer(U[:, i], VT[i, :])

loop_time = time.time() - start_time
print(f"Loop reconstruction time: {loop_time} seconds")

# Matrix multiplication time
start_time = time.time()

A_reconstructed_matmult = U @ S @ VT

matmult_time = time.time() - start_time
print(f"Matrix multiplication reconstruction time: {matmult_time} seconds")

start_time = time.time()

# here we are using broadcasting to avoid the creation of a diagonal matrix
# see: https://numpy.org/doc/stable/user/basics.broadcasting.html
A_reconstructed_vectorized = (U * s) @ VT

vectorized_time = time.time() - start_time
print(f"Vectorized reconstruction time: {vectorized_time} seconds")

# [Summary]
# Abbiamo confrontato tre metodi di ricostruzione della matrice: loop, moltiplicazione matriciale e broadcasting.
# Il metodo vettorizzato con broadcasting è il più efficiente, evitando la creazione della matrice diagonale S.