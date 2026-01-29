# Aritificial ANN (Artificial Neural Network) to predict avarage prices of California Houses

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import time  
import jax.numpy as jnp
import jax


data = pd.read_csv("./california_housing_train.csv")

data.head() # database structure
data.info()
data.describe() # statistics

# Distribuzione del valore mediano delle case (median_house_value)
sns.histplot(data["median_house_value"], kde=True) # Istograph media house value

data = data[data["median_house_value"] < 500001]
sns.histplot(data["median_house_value"], kde=True)

sns.scatterplot(data=data, x="longitude", y="latitude", hue="median_house_value") # Position and value of the house

# Matrix of linear correlation (between feetures)
data.corr()
# HeatMap
sns.heatmap(data.corr(), annot=True, cmap="vlag_r", vmin=-1, vmax=1)
# Scatter plot tra latitudine e valore della casa per vedere correlazione
sns.scatterplot(data=data, x="latitude", y="median_house_value")

# Normalizzazione delle features: ogni feature avrà media 0 e deviazione standard 1
data_mean = data.mean()  # Media di ogni colonna
data_std = data.std()   # Deviazione standard di ogni colonna
data_normalized = (data - data_mean) / data_std  # Normalizzazione Z-score

data_normalized.describe()
# Violin plot per vedere la distribuzione delle features normalizzate
_, ax = plt.subplots(figsize=(16, 6))
sns.violinplot(data=data_normalized, ax=ax)

# Mischiamo i dati per evitare bias nell'addestramento
np.random.seed(0)
data_normalized_np = data_normalized.to_numpy()  # Conversione a numpy array
np.random.shuffle(data_normalized_np)

# Suddivisione in training e validation set
fraction_validation = 0.2
num_train = int(data_normalized_np.shape[0] * (1 - fraction_validation))  
x_train = data_normalized_np[:num_train, :-1]  
y_train = data_normalized_np[:num_train, -1:] 
x_valid = data_normalized_np[num_train:, :-1] 
y_valid = data_normalized_np[num_train:, -1:] 

# --- ANN

# layers_size: lista delle dimensioni dei layer, es. [8, 5, 5, 1] significa input 8, hidden 5, hidden 5, output 1
def initialize_params(layers_size):
    np.random.seed(0)
    params = list()
    for i in range(len(layers_size) - 1):  # Per ogni connessione tra layer
        # Inizializzazione dei pesi con distribuzione normale (Xavier/Glorot initialization)
        W = np.random.randn(layers_size[i + 1], layers_size[i]) * np.sqrt(
            2 / (layers_size[i + 1] + layers_size[i])  # Fattore di scala (GAussiana)
        )
        b = np.zeros((layers_size[i + 1], 1))
        params.append((W, b))  # Aggiungi alla lista
    return params

params = initialize_params([8, 5, 5, 1])

# Definizione della funzione di attivazione: prima tanh, poi sovrascritta con ReLU
activation = jnp.tanh
activation = lambda x: jnp.maximum(0.0, x)  # ReLU: max(0, x), per evitare vanishing gradient

# Funzione che implementa la forward pass della rete neurale
# x: input (campioni x features), params: lista di (W, b)
def ANN(x, params):
    layer = x.T  # Trasponi per avere features x campioni
    for i, (W, b) in enumerate(params):  # Per ogni layer
        layer = W @ layer - b  # Prodotto matriciale + bias (nota: -b perché è sottrazione)
        if i < len(params) - 1:  # Se non è l'ultimo layer (output)
            layer = activation(layer)  # Applica funzione di attivazione
    return layer.T  # Trasponi di nuovo per avere campioni x output

# Test
ANN(x_train, params)

# Funzione di perdita: Mean Squared Error (MSE) quadratica
def loss(x, y, params):
    error = ANN(x, params) - y
    return jnp.mean(error * error)

params = initialize_params([8, 5, 5, 1])
loss(x_train, y_train, params)

# ---
# METODO DI GRADIENT DESCENT (GD) - Addestramento batch

layers_size = [8, 20, 20, 1] 
num_epochs = 2000
lr = 1e-1

params = initialize_params(layers_size)

grad = jax.jit(jax.grad(loss, argnums=2))  # JIT per accelerazione
loss_jit = jax.jit(loss)  # JIT per la loss
grad_jit = jax.jit(grad)  # JIT per il gradiente

n_samples = x_train.shape[0]  # Numero di campioni di training

history_train = list()
history_valid = list()
history_train.append(loss_jit(x_train, y_train, params))
history_valid.append(loss_jit(x_valid, y_valid, params))

t0 = time.time()
for epoch in range(num_epochs):
    grads = grad_jit(x_train, y_train, params)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

    history_train.append(loss_jit(x_train, y_train, params))
    history_valid.append(loss_jit(x_valid, y_valid, params))

print("elapsed time: %f s" % (time.time() - t0))
print("loss train     : %1.3e" % history_train[-1])
print("loss validation: %1.3e" % history_valid[-1])

fig, axs = plt.subplots(1, figsize=(16, 8))
axs.loglog(history_train, label="train")
axs.loglog(history_valid, label="validation")
plt.legend()

# -- 
# STOCHASTIC GRADIENT DESCENT (SGD) - Addestramento a mini-batch

layers_size = [8, 20, 20, 1]
learning_rate_max = 1e-1
learning_rate_min = 1e-1
learning_rate_decay = num_epochs
batch_size = 1000

params = initialize_params(layers_size)

grad = jax.jit(jax.grad(loss, argnums=2))
loss_jit = jax.jit(loss)
grad_jit = jax.jit(grad)

n_samples = x_train.shape[0]

history_train = list()
history_valid = list()
history_train.append(loss_jit(x_train, y_train, params))
history_valid.append(loss_jit(x_valid, y_valid, params))

t0 = time.time() 
for epoch in range(num_epochs):
    lr = max(
        learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay)
    )
    
    perm = np.random.permutation(n_samples)
    for i in range(0, n_samples, batch_size):
        batch_idx = perm[i : i + batch_size]  
        x_batch = x_train[batch_idx] 
        y_batch = y_train[batch_idx]  
        grads = grad_jit(x_batch, y_batch, params)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

    history_train.append(loss_jit(x_train, y_train, params))


print("elapsed time: %f s" % (time.time() - t0))
print("loss train     : %1.3e" % history_train[-1])
print("loss validation: %1.3e" % history_valid[-1])

fig, axs = plt.subplots(1, figsize=(16, 8))
axs.loglog(history_train, label="train")
axs.loglog(history_valid, label="validation")
plt.legend()

# --- TESTING 

data_test = pd.read_csv("./california_housing_test.csv")
data_test = data_test[data_test["median_house_value"] < 500001]
data_test_normalized = (data_test - data.mean()) / data.std()
x_test = data_test_normalized.drop("median_house_value", axis=1).to_numpy() 
Y_test = data_test["median_house_value"].to_numpy()[:, None]  

y_predicted = ANN(x_test, params)
Y_predicted = (y_predicted * data["median_house_value"].std()) + data["median_house_value"].mean()

test = pd.DataFrame({"predicted": Y_predicted[:, 0], "actual": Y_test[:, 0]})
fig = sns.jointplot(data=test, x="actual", y="predicted")
fig.ax_joint.plot([0, 500000], [0, 500000.0], "r")

# Calcolo dell'errore: Root Mean Squared Error (RMSE)
error = Y_test - Y_predicted 
RMSE = jnp.sqrt(jnp.mean(error * error))  
print("RMSE: %0.2f k$" % (RMSE * 1e-3))  