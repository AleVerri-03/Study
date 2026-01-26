import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

ovariancancer_obs_path = "./ovariancancer_obs.csv"
ovariancancer_grp_path = "./ovariancancer_grp.csv"

A = np.genfromtxt(ovariancancer_obs_path, delimiter=",").transpose()
with open(ovariancancer_grp_path, "r") as fp:
    grp = np.array(fp.read().split("\n"))
grp = grp[grp != ""] #holds the index infrormation as to which samples rapresente cancer patients and which ones represente normal patienta

n_features = A.shape[0]
n_patients = A.shape[1]
print(f"{n_features} features")
print(f"{n_patients} patients")

print(f"{np.sum(grp == 'Cancer')} Cancer")
print(f"{np.sum(grp == 'Normal')} Normal")
print(f"Total: {grp.size}")
is_sane = np.where(grp == "Normal", 1, 0)

protein_x = 0
protein_y = 1

plt.scatter(A[protein_x, grp == "Cancer"], A[protein_y, grp == "Cancer"], label = "Cancer")
plt.scatter(A[protein_x, grp == "Normal"], A[protein_y, grp == "Normal"], label = "Normal")
plt.legend()
# plt.show()

# Now with 3 proteins
fig = plt.figure() # prepare a new figure for 3d
ax = fig.add_subplot(111, projection='3d') # create a 3d axis

protein_x = 1000
protein_y = 2000
protein_z = 3000

for label in ("Cancer", "Normal"):
    ax.scatter(
        A[protein_x, grp == label],
        A[protein_y, grp == label],
        A[protein_z, grp == label],
        label=label,
        marker = "x",
        s=50,
    )
plt.legend()
ax.view_init(25,20)
# plt.show()

# PCA on data
A_mean = np.mean(A, axis=1)
A_bar = A - A_mean[:, None] # centered data matrix
U, s, VT = np.linalg.svd(A_bar, full_matrices=False)

plt.close('all')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

axes[0].semilogy(s, "o-")
axes[0].set_title("Singluar values")
axes[1].plot(np.cumsum(s) / np.sum(s), "o-")
axes[1].set_title("Cumulate fraction of singular values")
axes[2].plot(np.cumsum(s**2) / np.sum(s**2), "o-")
axes[2].set_title("Explained variance")
# plt.show()

#Plot first two principal components
Phi = U[:, :2].T @ A_bar # take first two principal components, traspose U and project data
plt.close('all')
plt.scatter(Phi[0, :], Phi[1, :], marker='o', c=(grp == "Normal"), s=50)
# plt.show()

# Now 3 
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
Phi = U[:, :3].T @ A_bar
ax.scatter(Phi[0, :], Phi[1, :], Phi[2, :], marker="x", c=(grp == "Normal"), s=50)
ax.view_init(25, 20)
# plt.show()

# MORE COMPACT
import plotly.express as px
px.scatter_3d(x=Phi[0, :], y=Phi[1, :], z=Phi[2, :], color=grp)

# plot the protein components (3 principals)
# plt.plot(U[:, 0:3])
# plt.legend(["PC1", "PC2", "PC3"])
# plt.xlabel("Biomarker ID")
# plt.ylabel("Importance")