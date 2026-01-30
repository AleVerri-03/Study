import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from flax import linen as nn
from flax.training import train_state
import optax

# Data
data = np.genfromtxt("./mnist_train_small.csv", delimiter=",")
data.shape

labels = data[:, 0]
x_data = data[:, 1:].reshape((-1, 28, 28, 1)) / 255
labels.shape, x_data.shape

# First 30 pictures 
fig, axs = plt.subplots(ncols=10, nrows=3, figsize=(20, 6))
axs = axs.reshape((-1,))
for i in range(30):
    image_i = x_data[i]
    axs[i].imshow(image_i, cmap="gray")
    axs[i].set_title(int(labels[i]))
    axs[i].axis("off")

# One-hot -> convert category data (label data) in numeric data (vector)
# Each row corresponds to a class
labels_onehot = np.zeros((20000, 10))
for i in range(10):
    labels_onehot[labels == i, i] = 1

# Check that each row has exactly one "1"
row_sums = np.sum(labels_onehot, axis=1)
row_sums.min(), row_sums.max()

# --- Training

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        # The `softmax_cross_entropy` expects unnormalized logits.
        # There is also the `softmax_cross_entropy_with_integer_labels`
        # version that uses integers target labels.
        # If you apply a softmax first, you turn logits into probabilities, and the
        # loss might becomes numerically unstable and incorrect. Optax/JAX expects to
        # handle the softmax internally in a stable way (using logsumexp tricks).
        x = nn.Dense(features=10)(x)  # There are 10 classes in MNIST
        return x


cnn = CNN()

table = cnn.tabulate(
    jax.random.PRNGKey(0), jnp.zeros((1, 28, 28, 1)), console_kwargs={"width": 200}
)

print(table)

# --- Cross entropy loss and accuracy
def compute_metrics(logits, labels_onehot):
    loss = jnp.mean(optax.softmax_cross_entropy(logits, labels_onehot))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels_onehot, -1))
    return {"loss": loss, "accuracy": accuracy}

# --- Functions used for training
@jax.jit
def loss_fn(params, x, y):
    logits = cnn.apply({"params": params}, x)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))
    return loss, logits


@jax.jit
def train_step(state, x, y):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params, x, y)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, y)
    return state, metrics


def eval_model(state, dataset):
    logits = state.apply_fn({"params": state.params}, dataset["image"])
    metrics = compute_metrics(logits, dataset["label"])
    return metrics["loss"], metrics["accuracy"]

# --- Function for oen training epoch
def train_epoch(state, train_ds, batch_size, epoch, rng):
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch["image"], batch["label"])
        batch_metrics.append(metrics)
    training_epoch_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics[0]
    }
    print(
        f"{epoch:04}  | "
        f"{training_epoch_metrics['loss']:.4e} | "
        f"    {training_epoch_metrics['accuracy'] * 100:.2f} | ",
        end=""
    )
    return state, training_epoch_metrics

# --- Randomizing and slit
np.random.seed(0)

n_samples = x_data.shape[0]
perm = np.random.permutation(n_samples)
train_perc = 0.8
n_train_samples = int(train_perc * n_samples)
train_idxs = perm[:n_train_samples]
valid_idx = perm[n_train_samples:]

train_ds = {
    "image": jnp.array(x_data[train_idxs]),
    "label": jnp.array(labels_onehot[train_idxs], dtype=jnp.float32),
}
valid_ds = {
    "image": jnp.array(x_data[valid_idx]),
    "label": jnp.array(labels_onehot[valid_idx], dtype=jnp.float32),
}

# --- Run training
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

training_losses = []
training_accuracies = []
valid_losses = []
valid_accuracies = []

num_epochs = 10
batch_size = 64
learning_rate = 0.001

params = cnn.init(init_rng, jnp.ones([1, 28, 28, 1]))["params"]
tx = optax.adam(learning_rate=learning_rate)
state = train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)

print("epoch | train loss | train acc | valid loss | valid acc")
for epoch in range(1, num_epochs + 1):
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state, train_metrics = train_epoch(state, train_ds, batch_size, epoch, input_rng)
    # Evaluate on the test set after each training epoch
    valid_loss, valid_accuracy = eval_model(state, valid_ds)
    print(
        f"{valid_loss:.4e} | " f"{valid_accuracy * 100:.2f}",
    )
    # Store metrics for graph visualization
    training_losses.append(train_metrics["loss"])
    training_accuracies.append(train_metrics["accuracy"])
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

# --- Testing

data_test = np.genfromtxt("./mnist_test.csv", delimiter=",")
data_test.shape
labels_test = data_test[:, 0]
x_test = data_test[:, 1:] / 255

labels_onehot_test = np.zeros((x_test.shape[0], 10))
for i in range(10):
    labels_onehot_test[labels_test == i, i] = 1.0


test_ds = {
    "image": jnp.array(x_test.reshape((-1, 28, 28, 1))),
    "label": jnp.array(labels_onehot_test),
}

# --- Accuracy
test_loss, test_accuracy = eval_model(state, test_ds)
print(f"Loss: {test_loss:.2e}")
print(f"Accuracy: {test_accuracy * 100.:.2f}%")


# --- Prediction
offset = 0
n_images = 40

images_per_row = 10
y_predicted = state.apply_fn({"params": state.params}, test_ds["image"])


def draw_bars(ax, y_predicted, label):
    myplot = ax.bar(range(10), (y_predicted))
    ax.set_ylim([0, 1])
    ax.set_xticks(range(10))

    label_predicted = np.argmax(y_predicted)
    if label == label_predicted:
        color = "green"
    else:
        color = "red"
    myplot[label_predicted].set_color(color)


import math

n_rows = 2 * math.ceil(n_images / images_per_row)
_, axs = plt.subplots(n_rows, images_per_row, figsize=(3 * images_per_row, 3 * n_rows))
row = 0
col = 0
for i in range(n_images):
    axs[2 * row, col].imshow(x_test[offset + i].reshape((28, 28)), cmap="gray")
    axs[2 * row, col].set_title(int(labels_test[offset + i]))
    axs[2 * row, col].axis("off")

    draw_bars(
        axs[2 * row + 1, col], jax.nn.softmax(y_predicted[i]), labels_test[offset + i]
    )

    col += 1
    if col == images_per_row:
        col = 0
        row += 1


# --- Adversarial attacks
# Small modification, that confuse the alghotim

# 1)Compute the gradient of cross entrpy loss function with respect to the input
# 2)Superimpose a multiple of the gradient to the original image
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def fgsm_attack(params, image, label, epsilon):
    # 1)Compute gradient of loss w.r.t. the input image
    grad_fn = jax.grad(lambda img: loss_fn(params, img, label), has_aux=True)

    gradient, _ = grad_fn(image)

    # 2)Apply perturbation
    adv_image = image + epsilon * jnp.sign(gradient)

    adv_image = jnp.clip(adv_image, 0.0, 1.0)
    return adv_image


# By trial-and-error I give you the information that for the following images
# an `epsilon = 0.05` is large enough to fool the CNN
epsilon = 0.05
for idx in [11, 66, 115, 244]:
    x = test_ds["image"][idx : idx + 1]
    y = test_ds["label"][idx]

    # True prediction
    logits = cnn.apply({"params": state.params}, x)
    true_pred = jnp.argmax(logits, axis=-1)
    print("Original prediction:", true_pred)

    # Create adversarial example
    x_adv = fgsm_attack(state.params, x, y, epsilon)

    # Prediction on adversarial image
    logits_adv = cnn.apply({"params": state.params}, x_adv)
    adv_pred = jnp.argmax(logits_adv, axis=-1)
    print("Adversarial prediction:", adv_pred)

    # Plot
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title(f"Original ({true_pred})")
    plt.imshow(x[0, :, :, 0], cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title(f"Adversarial ({adv_pred})")
    plt.imshow(x_adv[0, :, :, 0], cmap="gray")

    plt.show()

def get_first_layer_output(cnn_module, params, input_data):
    """
    Manually applies the first convolutional layer and ReLU.
    """
    # 1. Access the first Conv layer from the parameters
    # The name of the first Conv layer is typically 'Conv_0' by default in Flax
    # if it's the first nn.Conv without an explicit name.
    
    # Instantiate the first Conv layer with the correct features and kernel size
    conv_layer = nn.Conv(features=32, kernel_size=(3, 3), name='Conv_0')

    # Apply the convolution using the parameters (weights and biases)
    # This requires using the `apply` method of the layer and passing the specific subset of parameters
    conv_output = conv_layer.apply({'params': params['Conv_0']}, input_data)
    
    # 2. Apply ReLU
    final_output = nn.relu(conv_output)
    
    return final_output

idx = 0
first_layer_output = get_first_layer_output(cnn, state.params, test_ds["image"][idx : idx + 1])
first_layer_output.shape

for i in range(32):
    plt.figure()
    plt.imshow(first_layer_output[0, :, :, i])