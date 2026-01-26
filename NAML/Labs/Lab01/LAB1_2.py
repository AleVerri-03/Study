from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.ndimage import rotate

image_path = "./TarantulaNebula.jpg"

# Caricamento dell'immagine RGB e conversione in scala di grigi
A = imread(image_path)
# A = rotate(A, 20, reshape=False)

img = plt.imshow(A)
plt.axis("off")

X = np.mean(A, axis=2)

img = plt.imshow(X)
plt.axis("off")
img.set_cmap("gray")
# plt.show()

X.shape # Picture size

# Calcolo della SVD ridotta dell'immagine in scala di grigi
U, s, VT = la.svd(X, full_matrices=False)
S = np.diag(s)

# Visualizzazione dei valori singolari: decadimento, somma cumulativa e varianza spiegata
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
axes[0].semilogy(s, "o-")
axes[0].set_title("Singluar values")

axes[1].plot(np.cumsum(s) / np.sum(s), "o-")
axes[1].set_title("Cumulate fraction of singular values")

axes[2].plot(np.sqrt(np.cumsum(s**2) / np.sum(s**2)), "o-")
axes[2].set_title("Explained variance")

plt.show()

# Best rank-k matrix for k = 1, 2, 5, 10, 15, 50
# Approssimazione low-rank: A_k = U[:,:k] @ S[:k,:k] @ VT[:k,:] minimizza ||A - A_k||_F
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axs = axs.reshape((-1,))
idxs = [1, 2, 5, 10, 15, 50]
for i in range(len(idxs)):
    k = idxs[i]
    # SOLUTION-BEGIN
    # Ricostruzione usando i primi k valori/vettori singolari
    A_k = np.matmul(U[:, :k], np.matmul(np.diag(s[:k]), VT[:k, :]))
    axs[i].imshow(A_k, cmap="gray")
    # SOLUTION-END
    axs[i].set_title(f"k = {k}")
    axs[i].axis("off")
plt.show()

# Best k-th rank-1 matrix for k = 1, 2, 3, 4, 5, 6
# Ogni matrice rank-1 è data dal prodotto esterno u_k * v_k^T (pesato da sigma_k)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axs = axs.reshape((-1,))
idxs = [1, 2, 3, 4, 5, 6]
for i, k in enumerate(idxs):
    # SOLUTION-BEGIN
    # Il k-esimo contributo rank-1 cattura un pattern specifico dell'immagine
    ukvk = np.outer(U[:, k - 1], VT[k - 1, :])
    axs[i].imshow(ukvk, cmap="gray")
    # SOLUTION-END
    axs[i].set_title(f"{k}-th rank-1 matrix")
    axs[i].axis("off")

plt.show()

# [Summary]
# Abbiamo applicato la SVD ad un'immagine per analizzare i valori singolari e la varianza spiegata.
# Abbiamo ricostruito l'immagine usando approssimazioni di rango k e visualizzato le singole matrici di rango 1.
# --------------------------------------------

# SVD randomizzata: algoritmo efficiente per matrici grandi
# Usa proiezione random per ridurre la dimensionalità prima di calcolare la SVD
def randomized_SVD(A, k):
    # SOLUTION-BEGIN
    _, n = A.shape
    P = np.random.randn(n, k)
    Z = A @ P
    Q, _ = np.linalg.qr(Z)
    Y = Q.T @ A
    Uy, sy, VTy = np.linalg.svd(Y, full_matrices=False)
    U = Q @ Uy
    # SOLUTION-END
    return U, sy, VTy

k = 100
# SOLUTION-BEGIN
U_rand, s_rand, VT_rand = randomized_SVD(X, k)
# SOLUTION-END

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# SOLUTION-BEGIN
axs[0].loglog(s, "o-")
axs[0].loglog(s_rand, "+-")
axs[0].set_title("singular values")

axs[1].semilogx(np.cumsum(s), "o-")
axs[1].semilogx(np.cumsum(s_rand), "+-")
axs[1].set_title("cumulative fraction")

axs[2].semilogx(np.cumsum(s**2), "o-")
axs[2].semilogx(np.cumsum(s_rand**2), "+-")
axs[2].set_title("explained variance")

plt.show()
# SOLUTION-END

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# SOLUTION-BEGIN
axs[0].imshow(X, cmap="gray")
axs[1].imshow(U[:, :k] @ np.diag(s[:k]) @ VT[:k, :], cmap="gray")
axs[2].imshow(U_rand @ np.diag(s_rand) @ VT_rand, cmap="gray")

plt.show()
# SOLUTION-END

# [Summary]
# La SVD randomizzata approssima bene la SVD esatta con costo computazionale O(mnk) invece di O(mn*min(m,n)).
# Il confronto visivo mostra che l'approssimazione randomizzata è quasi indistinguibile dall'originale.