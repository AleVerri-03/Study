import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Generate a random boolean mask to simulate missing pixels in an image
def gen_mask(A, pct, random_state=0):
    shape = A.shape
    ndims = np.prod(shape, dtype=np.int32)
    mask = np.full(ndims, False)
    mask[: int(pct * ndims)] = True
    np.random.seed(random_state)
    np.random.shuffle(mask)
    mask = mask.reshape(shape)
    return mask

# Resize an image to a specified size while preserving aspect ratio
def resize_image(img, size):
    img = img.copy()
    img.thumbnail(size=(size, size))
    return img

fname = "./landscape.jpg"
raw_img = Image.open(fname) # Import
size = 400
img = resize_image(raw_img, size) # Resize
print(f"Raw image size: {raw_img.size}")
print(f"Resized image size: {img.size}")

img = np.array(img) / 255 # Normalization
img = img.mean(axis=-1) # Grayscale convertion
plt.imshow(img, cmap="gray")

# SVT alghoritm (image inpainting)
# 1) Generate mask to simulate missing pixels
pct = 0.5 # 50% missing pixels
mask = gen_mask(A=img, pct=pct, random_state=0)

n_iters = 700
tol = 0.01

delta = 1.2 # step-size SVT
tau = 5 * np.sqrt(img.shape[0]*img.shape[1]) # Theshold 
c0 = np.ceil(tau / (delta * np.linalg.norm(img, ord=2))) # Number iteration to converge

X = img.copy()
M = np.zeros_like(X)
Y = c0 * delta * mask * X # like "forzante" in inpainting
prev_err = 100000
err = prev_err

for i in range(n_iters):
    u, s, vh = np.linalg.svd(Y, full_matrices=False)
    shrink_s = np.maximum(s - tau, 0) # Delete over tau and shift
    M = (u * shrink_s) @ vh # New matrix

    # TODO: DA QUI NON PIU' CAPITO
    Y += delta * mask * (X - M)
    prev_err = err
    abs_err = np.linalg.norm(mask * (M - X), ord="fro")
    err = abs_err / np.linalg.norm(mask * X, ord="fro")
    print(f"Iteration: {i}; err: {err:.6f}")

    # save best
    if err < prev_err:
        prev_err = err
        best_X = np.clip(M, a_min=0, a_max=1)

    # visualize
    if i % 40 == 0:
        fig, axs = plt.subplots(2, 2, figsize=(15, 7))
        axs[0, 0].set_title("Original Image")
        axs[0, 0].imshow(X, cmap="gray")
        axs[0, 0].axis("off")
        axs[0, 1].set_title("Dirty Image")
        axs[0, 1].imshow(X * mask, cmap="gray")
        axs[0, 1].axis("off")
        axs[1, 0].set_title("Reconstructed Image")
        axs[1, 0].imshow(best_X, cmap="gray")
        axs[1, 0].axis("off")
        axs[1, 1].set_title("Error")
        axs[1, 1].imshow(np.abs(best_X - img), vmin=0, vmax=1, cmap="gray")
        axs[1, 1].axis("off")
        plt.show()

    if err <= tol:
        break