# !wget http://backgroundmodelschallenge.eu/data/real/Video_003.zip
# !unzip Video_003.zip

import moviepy.editor as mpe
import numpy as np
import skimage.transform
# %matplotlib inline
import matplotlib.pyplot as plt

video = mpe.VideoFileClip("Video_003/Video_003.avi")
print(f"Video duration: {video.duration} seconds")
print(f"Video size: {video.size}")
print(f"Video FPS: {video.fps}")

# ipython_display() works only in Jupyter Notebook
# video.subclip(0, 5).ipython_display(width=300)

# To preview: save a subclip as file
# video.subclip(0, 5).write_videofile("preview.mp4")

def create_data_matrix_from_video(clip, dims):
    number_of_frames = int(clip.fps * clip.duration)
    flatten_gray_frames = []
    for i in range(number_of_frames):
        # get_frame takes as input the time of the frame
        frame = clip.get_frame(i / clip.fps)
        # to gray scale
        gray_frame = np.mean(frame[..., :3], axis=-1).astype(int)
        # resize to reduce computational cost
        small_gray_frame = skimage.transform.resize(gray_frame, dims)
        # each frame becomes a column vector of A
        flatten_gray_frames.append(small_gray_frame.flatten())
    return np.vstack(flatten_gray_frames).T

scale = 0.50  # Adjust scale to change resolution of image
width, height = video.size
dims = (int(height * scale), int(width * scale))
A = create_data_matrix_from_video(video, dims)
print("frame size:", dims)
print("video matrix size:", A.shape)

plt.imshow(np.reshape(A[:, 140], dims), cmap="gray") #Visualize frame 140
plt.title("Frame 140")
# plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(A, cmap="gray", aspect="auto")
plt.title("Video matrix A")
plt.xlabel("Frames")
plt.ylabel("Pixels")
# plt.show()

# Use SVD
import time
def my_svd(XX, k):
    return np.linalg.svd(XX, full_matrices=False)

t0 = time.time()
U, s, VT = my_svd(A, 10)
print(f"SVD elapsed {time.time() - t0: .2f} [s]")

U.shape, s.shape, VT.shape

reconstructed_A = U @ np.diag(s) @ VT
np.allclose(A, reconstructed_A)

n_singular_values = 1
background = U[:, 0:n_singular_values] * s[0] * VT[0:n_singular_values, :]
plt.figure(figsize=(12, 6))
plt.imshow(background, cmap="gray", aspect="auto")

def plot_frames(A, background, time_ids):
    fig, axs = plt.subplots(len(time_ids), 3, figsize=(12, 4 * len(time_ids)))
    for i, t_id in enumerate(time_ids):
        axs[i, 0].imshow(np.reshape(A[:, t_id], dims), cmap="gray")
        axs[i, 1].imshow(np.reshape(background[:, t_id], dims), cmap="gray")
        axs[i, 2].imshow(
            np.reshape(A[:, t_id] - background[:, t_id], dims), cmap="gray"
        )

        axs[i, 0].set_ylabel(f"Frame {t_id}")

        if i == 0:
            axs[0, 0].set_title("Original video")
            axs[0, 1].set_title("Background")
            axs[0, 2].set_title("Foreground")


time_ids = [0, 150, 300, 450]
plot_frames(A, background, time_ids)
plt.show()