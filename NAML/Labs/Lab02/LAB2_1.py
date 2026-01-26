import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

theta1 = np.pi / 6 # 30 degrees in radians
theta2 = theta1 + np.pi / 2 # 120 degrees in radians

z1 = np.array((np.cos(theta1), np.sin(theta1)))
z2 = np.array((np.cos(theta2), np.sin(theta2)))
# Quindi z1 e z2 sono ortogonali

b = np.array((20, 30)) # Vettore di traslazione (bias)

rho1 = 12.0
rho2 = 3.0
n_points = 1000

seeds = np.random.randn(2, n_points) # Generazione di 1000 punti casuali
X = np.column_stack((rho1 * z1, rho2 * z2)) @ seeds + b[:, None]
# Costruzione matrice 2x2 Z = [rho1 * z1, rho2 * z2]
# Poi trasformazione lineare dei punti casuali con Z e aggiunta del bias b


X.shape
# Ottenfo una nuvola di punti centrati in 20, 30 con distribuzione gaussiana allungata

plt.figure(figsize=(8, 8))
plt.scatter(X[0, :], X[1, :], alpha=0.5)
plt.axis('equal')
plt.title('Nuvola di punti con distribuzione gaussiana allungata')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
#    plt.show()

# Plot dei vettori z1 e z2 scalati da rho1 e rho2
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(X[0, :], X[1, :])
ax.arrow(
    b[0] - z1[0] * rho1,
    b[1] - z1[1] * rho1,
    2 * z1[0] * rho1,
    2 * z1[1] * rho1,
    color="black",
)
ax.arrow(
    b[0] - z2[0] * rho2,
    b[1] - z2[1] * rho2,
    2 * z2[0] * rho2,
    2 * z2[1] * rho2,
    color="black",
)
ax.set_aspect("equal")

ax.set_title("Vettori principali della distribuzione")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
plt.grid(True)
# plt.show()

# PCA
X_mean = np.mean(X, axis=1) # Calcolo della media dei dati (axis 1 per media lungo le colonne)
U, s, VT = np.linalg.svd(X - X_mean[:, None], full_matrices=False)
# X - X_mean[:, None] centra i dati sottraendo la media
# SVD per ottenere i vettori principali (U) e i valori singolari (s)

u1 = U[:, 0] # Prima e seconda componente principale
u2 = U[:, 1]

r = s / np.sqrt(n_points - 1) # Conversione dei valori singolari in deviazioni standard

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(X[0, :], X[1, :])
plt.arrow(
    X_mean[0] - u1[0] * r[0], # Freccia centrata sulla media
    X_mean[1] - u1[1] * r[0], # si estende di 2*r[0] lungo u1
    2 * u1[0] * r[0],
    2 * u1[1] * r[0],
    color="red",
)
plt.arrow(
    X_mean[0] - u2[0] * r[1],
    X_mean[1] - u2[1] * r[1],
    2 * u2[0] * r[1],
    2 * u2[1] * r[1],
    color="red",
)
ax.set_aspect("equal")
ax.set_title("Vettori principali stimati tramite PCA")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
plt.grid(True)
#   plt.show()

#    print(z1, z2)
#    print(u1, u2)
# Simili a meno di cambio segno -> PCA ha correttamente identificato le direzioni principali della distribuzione

Phi = np.matmul(U.transpose(), X - X_mean[:, None])
# Proiezione dei dati centrati sul nuovo sistema di riferimento definito dai vettori principali U

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(Phi[0, :] / r[0], Phi[1, :] / r[1])
ax.set_aspect("equal")
ax.set_title("Dati proiettati sul sistema di riferimento PCA")
ax.set_xlabel("Componente principale 1 (normalizzata)")
ax.set_ylabel("Componente principale 2 (normalizzata)")
plt.grid(True)
#    plt.show()


# Riassumendo:
# 1. Generazione dati non geometria nota (gaussiana allungata)
# 2. Applicazione PCA per identificare direzioni principali della distribuzione e varianza