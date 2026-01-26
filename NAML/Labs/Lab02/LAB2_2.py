import matplotlib.pyplot as plt
import numpy as np

# Applicazione PCA a dataset

data = np.genfromtxt("./mnist_train_small.csv", delimiter=",")
data.shape

labels_full = data[:, 0]
A_full = data[:, 1:].transpose()
labels_full.shape, A_full.shape
# Strucure data: 6000 immagini di 28x28 pixel (784 features) con etichette da 0 a 9

# First 30 picture
fig, axs = plt.subplots(ncols=10, nrows=3, figsize=(20, 6))
axs = axs.reshape((-1,))
for i in range(30):
    image_i = A_full[:, i].reshape((28, 28))
    axs[i].imshow(image_i, cmap="gray")
    axs[i].set_title(f"{labels_full[i]:.0f}")
    axs[i].axis("off")
# plt.show()

A_filtered = A_full[:, labels_full == 9] # Filtraggio delle immagini con etichetta 9
fig, axs = plt.subplots(ncols=10, nrows=3, figsize=(20, 6))
axs = axs.reshape((-1,))
for i in range(30):
    image_i = A_filtered[:, i].reshape((28, 28))
    axs[i].imshow(image_i, cmap="gray")
    axs[i].axis("off")
# plt.show()

# Build reducted dataset with digit 0 and 9
digits = (0, 9)
mask = np.logical_or(labels_full == digits[0], labels_full == digits[1])
A = A_full[:, mask]
labels = labels_full[mask]

plt.close('all')
A_mean = A.mean(axis=1) # Avarage image of digits 0 and 9
plt.imshow(A_mean.reshape((28, 28)), cmap="gray")
plt.axis("off")
plt.title("Media delle immagini dei digit 0 e 9")
# plt.show()

A_bar = A - A_mean[:, None] # Subtraction of the mean image
U, s, VT = np.linalg.svd(A_bar, full_matrices=False)

plt.close('all')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

axes[0].semilogy(s, "o-")
axes[0].set_title("Singluar values")

axes[1].plot(np.cumsum(s) / np.sum(s), "o-")
axes[1].set_title("Cumulate fraction of singular values")

axes[2].plot(np.cumsum(s**2) / np.sum(s**2), "o-")
axes[2].set_title("Explained variance")
#plt.show()

plt.close('all')
fig, axs = plt.subplots(nrows=3, ncols=10, figsize=(20, 6))
axs = axs.reshape((-1,))
for i in range(len(axs)):
    image_i = U[:, i].reshape((28, 28))
    axs[i].imshow(image_i, cmap="gray")
    axs[i].axis("off")
    axs[i].set_title(f"$u_{{{i + 1}}}$")
# plt.show()

A_pc = np.matmul(U.T, A_bar)
print(f"1st principal component: {A_pc[0, 0]}")
print(f"2nd principal component: {A_pc[1, 0]}")

# firt two principal components scatter plot
plt.close('all')
for i in range(500):    
    # take the i-th sample and find the magnitude of its projection 
    # (that is, scalar prod) on the first and second principal axes
    x = np.inner(A_bar[:, i], U[:, 0]) # scalar product with first principal component
    y = np.inner(A_bar[:, i], U[:, 1]) # scalar product with second principal component
    col = "r" if labels[i] == digits[0] else "b"
    plt.scatter(x, y, marker="x", color=col, s=50)
plt.title("Proiezione sulle prime due componenti principali")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
# plt.show()

# plt.scatter(A_pc[0, :500], A_pc[1, :500], marker="x", c=labels[:500], s=50)


# Classifier based on PCA, find a threshold on the first principal component
threshold = 999 # initial value
plt.close('all')
plt.scatter(A_pc[0, :500], A_pc[1, :500], marker="x", c=labels[:500], s=50)
plt.axvline(threshold, color="k", linestyle="--")
# plt.show()

# Dataset
data_test = np.genfromtxt("./mnist_test.csv", delimiter=",")
labels_full_test = data_test[:, 0]
A_full_test = data_test[:, 1:].transpose()
labels_full_test.shape, A_full_test.shape

mask = np.logical_or(labels_full_test == digits[0], labels_full_test == digits[1])
A_test = A_full_test[:, mask]
labels_test = labels_full_test[mask]
labels_test.shape, A_test.shape

# A_mean is the one of the training dataset!
plt.close('all')
A_pc_test = U.T @ (A_test - A_mean[:, None])
plt.scatter(A_pc_test[0, :500], A_pc_test[1, :500], marker="x", c=labels_test[:500], s=50)
plt.axvline(threshold, color="k", linestyle="--")
# plt.show()

PC_1 = A_pc_test[0, :]

# SOLUTION-BEGIN
labels_predicted = np.where(PC_1 > threshold, digits[0], digits[1])

# the operator "&" in numpy is equivalent to "np.logical_and"
true_0 = np.sum((labels_test == digits[0]) & (labels_predicted == digits[0]))
false_0 = np.sum((labels_test == digits[1]) & (labels_predicted == digits[0]))
true_1 = np.sum((labels_test == digits[1]) & (labels_predicted == digits[1]))
false_1 = np.sum((labels_test == digits[0]) & (labels_predicted == digits[1]))

print(f"true  {digits[0]}: {true_0}")
print(f"false {digits[0]}: {false_0}")
print(f"true  {digits[1]}: {true_1}")
print(f"false {digits[1]}: {false_1}")
accuracy = (true_0 + true_1) / (true_0 + true_1 + false_0 + false_1)
print(f"accuracy = {accuracy * 100:.2f} %")