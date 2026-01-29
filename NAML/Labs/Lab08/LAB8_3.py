import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import jax.numpy as jnp
import jax

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]
data = pd.read_csv(
    url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True
)
data

# Check
print(data.isna().sum())

# Delete missing
data = data.dropna()
print(data.isna().sum())

# Some info
data.head()
data.info()
data.describe()

# Predict MPG, plot distribution
sns.displot(data["MPG"], kde=True)
sns.heatmap(data.corr(), annot=True, cmap="vlag_r", vmin=-1, vmax=1)
sns.pairplot(data, diag_kind="kde")

# Normalize in order to better training
data_mean = data.mean()
data_std = data.std()
data_normalized = (data - data_mean) / data_std
_, ax = plt.subplots(figsize=(16, 6))
sns.violinplot(data=data_normalized, ax=ax)

data_normalized_np = data_normalized.to_numpy()
np.random.seed(0)
np.random.shuffle(data_normalized_np)

fraction_validation = 0.2
num_train = int(data_normalized_np.shape[0] * (1 - fraction_validation))
x_train = data_normalized_np[:num_train, 1:]
y_train = data_normalized_np[:num_train, :1]
x_valid = data_normalized_np[num_train:, 1:]
y_valid = data_normalized_np[num_train:, :1]

print("train set size     : %d" % x_train.shape[0])
print("validation set size: %d" % x_valid.shape[0])

# ANN

def initialize_params(layers_size):
    np.random.seed(0)
    params = list()
    for i in range(len(layers_size) - 1):
        W = np.random.randn(layers_size[i + 1], layers_size[i]) * np.sqrt(
            2 / (layers_size[i + 1] + layers_size[i])
        )
        b = np.zeros((layers_size[i + 1], 1))
        params.append(W)
        params.append(b)
    return params

activation = jax.nn.relu

# Activation and ANN definition
def ANN(x, params):
    layer = x.T
    num_layers = int(len(params) / 2 + 1)
    weights = params[0::2]
    biases = params[1::2]
    for i in range(num_layers - 1):
        layer = weights[i] @ layer - biases[i]
        if i < num_layers - 2:
            layer = activation(layer)
    return layer.T


params = initialize_params([7, 10, 1])
ANN(x_train[:10, :], params)

# MSE
def MSE(x, y, params):
    error = ANN(x, params) - y
    return jnp.mean(error * error)


params = initialize_params([7, 10, 1])
print(MSE(x_train, y_train, params))

# Avarage Weight^2
def MSW(params):
    weights = params[::2]
    partial_sum = 0.0
    n_weights = 0
    for W in weights:
        partial_sum += jnp.sum(W * W)
        n_weights += W.shape[0] * W.shape[1]
    return partial_sum / n_weights # Help to understand the weight of the net


def loss(x, y, params, penalization):
    return MSE(x, y, params) + penalization * MSW(params) # Usefull to prevent "overfitting"

print(MSW(params))
print(loss(x_train, y_train, params, 1.0))

# Monitor training
from IPython import display


class Callback:
    def __init__(self, refresh_rate=250):
        self.refresh_rate = refresh_rate
        self.fig, self.axs = plt.subplots(1, figsize=(16, 8))
        self.epoch = 0
        self.__call__(-1)

    def __call__(self, epoch):
        self.epoch = epoch
        if (epoch + 1) % self.refresh_rate == 0:
            self.draw()
            display.clear_output(wait=True)
            display.display(plt.gcf())
            time.sleep(1e-16)

    def draw(self):
        if self.epoch > 0:
            self.axs.clear()
            self.axs.loglog(history_loss_train, "b-", label="loss train")
            self.axs.loglog(history_loss_valid, "r-", label="loss validation")
            self.axs.loglog(history_MSE_train, "b--", label="RMSE train")
            self.axs.loglog(history_MSE_valid, "r--", label="RMSE validation")
            self.axs.legend()
            self.axs.set_title("epoch %d" % (self.epoch + 1))

# ---
# Training

# Hyperparameters
layers_size = [7, 20, 20, 1]
penalization = 2.0
# Training options
num_epochs = 5000
learning_rate_max = 1e-1
learning_rate_min = 5e-3
learning_rate_decay = 1000
batch_size = 100
alpha = 0.9

params = initialize_params(layers_size)

grad = jax.grad(loss, argnums=2)
MSE_jit = jax.jit(MSE)
loss_jit = jax.jit(loss)
grad_jit = jax.jit(grad)

n_samples = x_train.shape[0]

history_loss_train = list()
history_loss_valid = list()
history_MSE_train = list()
history_MSE_valid = list()


def dump():
    history_loss_train.append(loss_jit(x_train, y_train, params, penalization))
    history_loss_valid.append(loss_jit(x_valid, y_valid, params, penalization))
    history_MSE_train.append(MSE_jit(x_train, y_train, params))
    history_MSE_valid.append(MSE_jit(x_valid, y_valid, params))


dump()
cb = Callback(refresh_rate=500)

velocity = [0.0 for i in range(len(params))]
for epoch in range(num_epochs):
    learning_rate = max(
        learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay)
    )
    idxs = np.random.choice(n_samples, batch_size)
    grads = grad_jit(x_train[idxs, :], y_train[idxs, :], params, penalization)

    for i in range(len(params)):
        velocity[i] = alpha * velocity[i] - learning_rate * grads[i]
        params[i] += velocity[i]

    dump()
    cb(epoch)
cb.draw()

print("loss (train     ): %1.3e" % history_loss_train[-1])
print("loss (validation): %1.3e" % history_loss_valid[-1])
print("MSE  (train     ): %1.3e" % history_MSE_train[-1])
print("MSE  (validation): %1.3e" % history_MSE_valid[-1])

# ---


# Effect of the penalization parameter
# Hyperparameters
layers_size = [7, 20, 20, 1]
# Training options
num_epochs = 5000
learning_rate_max = 1e-1
learning_rate_min = 5e-3
learning_rate_decay = 1000
batch_size = 100
alpha = 0.9


def train(penalization):
    params = initialize_params(layers_size)
    n_samples = x_train.shape[0]

    velocity = [0.0 for i in range(len(params))]
    for epoch in range(num_epochs):
        learning_rate = max(
            learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay)
        )
        idxs = np.random.choice(n_samples, batch_size)
        grads = grad_jit(x_train[idxs, :], y_train[idxs, :], params, penalization)
        for i in range(len(params)):
            velocity[i] = alpha * velocity[i] - learning_rate * grads[i]
            params[i] += velocity[i]
    return {
        "MSE_train": MSE(x_train, y_train, params),
        "MSE_valid": MSE(x_valid, y_valid, params),
        "MSW": MSW(params),
    }

# Make all

results = []

pen_values = np.arange(0, 2.1, 0.25)
for beta in pen_values:
    print("training for beta = %f..." % beta)
    results.append({"beta": beta, **train(beta)}) 

hyper_tuning_df = pd.DataFrame(results)
hyper_tuning_df

# Plot the trend

_, axs = plt.subplots(1, 3, figsize=(12, 6))

axs[0].plot(pen_values, hyper_tuning_df["MSE_train"], "o-")
axs[1].plot(pen_values, hyper_tuning_df["MSE_valid"], "o-")
axs[2].plot(pen_values, hyper_tuning_df["MSW"], "o-")

axs[0].set_title("MSE_train")
axs[1].set_title("MSE_valid")
axs[2].set_title("MSW")
axs[0].set_xlabel(r"$\beta$")
axs[1].set_xlabel(r"$\beta$")
axs[2].set_xlabel(r"$\beta$")

# Plot Tikhonov L-curve
plt.plot(hyper_tuning_df["MSW"], hyper_tuning_df["MSE_train"], "o-")
plt.xlabel("MSW")
plt.ylabel("MSE_train")