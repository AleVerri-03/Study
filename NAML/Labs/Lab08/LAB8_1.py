# Questo script implementa vari algoritmi di ottimizzazione di primo ordine per addestrare una rete neurale
# a interpolare una funzione matematica rumorosa. Confronta GD, SGD, Momentum, RMSProp e Adam.

import numpy as np 
import time
import jax.numpy as jnp
import jax 

# OTTIMIZZAZIONE DI PRIMO ORDINE (basata su gradienti)

f = lambda x: np.sin(x) * np.exp(-0.1 * x) + 0.1 * np.cos(np.pi * x)
a, b = 0, 10 

def get_training_data(N, noise):
    np.random.seed(0)
    x = np.linspace(a, b, N)[:, None]
    y = f(x) + noise * np.random.randn(N, 1)
    return x, y

# Plot della funzione vera (senza rumore) per riferimento
x_fine = np.linspace(a, b, 1000)[:, None]
plt.plot(x_fine, f(x_fine))

# Generazione dati di training con 100 punti e rumore 0.05
xx, yy = get_training_data(100, 0.05)
plt.plot(xx, yy, "o")

def initialize_params(layers_size):
    np.random.seed(0)
    params = list()
    for i in range(len(layers_size) - 1):  # Per ogni layer
        W = np.random.randn(layers_size[i + 1], layers_size[i])  # Pesi casuali
        b = np.zeros((layers_size[i + 1], 1))  # Bias a zero
        params.append(W) 
        params.append(b) 
    return params

# Funzione della rete neurale: input normalizzato, due hidden layer con tanh, output lineare
def ANN(x, params):
    layer = (2 * x.T - (a + b)) / (b - a) # Normalizzazione dell'input nell'intervallo [-1, 1] per stabilità
    num_layers = int(len(params) / 2 + 1)  # Numero di layer (input + hidden + output)
    weights = params[0::2]  # Pesi (elementi pari)
    biases = params[1::2]   # Bias (elementi dispari)
    for i in range(num_layers - 1):  # Per ogni layer
        layer = jnp.dot(weights[i], layer) - biases[i]  # Prodotto matriciale + bias
        if i < num_layers - 2:  # Se non è l'ultimo layer
            layer = jnp.tanh(layer)  # Attivazione tanh
    return layer.T  # Trasponi per output (campioni x output)

# Mean Squared Error (MSE)
def loss(x, y, params):
    error = ANN(x, params) - y
    return jnp.mean(error * error)

# Test 
params = initialize_params([1, 5, 5, 1])
loss(xx, yy, params)  # Perdita iniziale (dovrebbe essere alta)

from IPython import display

# Classe per callback durante l'addestramento: aggiorna i plot ogni tot epoche
class Callback:
    def __init__(self, refresh_rate=250):
        self.refresh_rate = refresh_rate  # Ogni quante epoche aggiornare il plot
        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 8))  # Due subplot: perdita e previsione
        self.x_fine = np.linspace(a, b, 200)[:, None]  # Punti per plot liscio
        self.epoch = 0
        self.__call__(-1)  # Chiamata iniziale per setup

    def __call__(self, epoch):
        self.epoch = epoch
        if (epoch + 1) % self.refresh_rate == 0:  # Se è tempo di aggiornare
            self.draw()  # Disegna
            display.clear_output(wait=True)  # Cancella output precedente
            display.display(plt.gcf())  # Mostra figura
            time.sleep(1e-16)  # Piccola pausa

    def draw(self):
        if self.epoch > 0:  # Se non è l'inizio
            self.axs[0].clear()  # Pulisci subplot sinistro
            self.axs[0].loglog(history)  # Plot perdita in scala log
            self.axs[0].set_title("epoch %d" % (self.epoch + 1))  # Titolo con epoca

        self.axs[1].clear()  # Pulisci subplot destro
        self.axs[1].plot(self.x_fine, f(self.x_fine))  # Funzione vera
        self.axs[1].plot(self.x_fine, ANN(self.x_fine, params))  # Previsione rete
        self.axs[1].plot(xx, yy, "o")  # Punti di training

# --- PRIMA SEZIONE: GRADIENT DESCENT (GD) - Addestramento batch completo

# Dati
n_training_points = 100
noise = 0.05
layers_size = [1, 5, 5, 1]  # Architettura
# Opzioni training
num_epochs = 2000 
learning_rate = 1e-1

# Rigenera dati (stesso seed)
xx, yy = get_training_data(n_training_points, noise)
params = initialize_params(layers_size)

# Ottimizzazione JIT
loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.grad(loss, argnums=2))

history = list()
history.append(loss_jit(xx, yy, params))  # Perdita iniziale

# Callback per visualizzazione
cb = Callback(refresh_rate=200)  # Aggiorna ogni 200 epoche

for epoch in range(num_epochs):
    grads = grad_jit(xx, yy, params) # Calcola gradiente su tutto il dataset
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)     # Aggiorna parametri: p = p - lr * g (discesa del gradiente)
    history.append(loss_jit(xx, yy, params)) # Registra perdita dopo aggiornamento
    cb(epoch)  # Chiama callback per plot

cb.draw()  # Disegno finale
print("loss: %1.3e" % history[-1])  # Perdita finale

# --- SECONDA SEZIONE: STOCHASTIC GRADIENT DESCENT (SGD) - Addestramento a mini-batch con decadimento learning rate

n_training_points = 100
noise = 0.05
layers_size = [1, 5, 5, 1]
num_epochs = 20000 
learning_rate_max = 1e-1 
learning_rate_min = 2e-2
learning_rate_decay = 10000 
batch_size = 10 

xx, yy = get_training_data(n_training_points, noise) # Rigenera dati
params = initialize_params(layers_size) # Re-inizializza parametri

loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.grad(loss, argnums=2))

history = list()
history.append(loss_jit(xx, yy, params))

cb = Callback(refresh_rate=250)  # Aggiorna ogni 250 epoche

for epoch in range(num_epochs):
    lr = max(learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay))     # Decadimento lineare del learning rate
    perm = np.random.permutation(n_training_points)  # Permutazione per shuffle dei dati ogni epoca
    for i in range(0, n_training_points, batch_size): # Ciclo sui mini-batch
        batch_idx = perm[i : i + batch_size]  # Indici batch
        x_batch = xx[batch_idx]  # Features batch
        y_batch = yy[batch_idx]  # Target batch
        grads = grad_jit(x_batch, y_batch, params) # Gradiente sul batch
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads) # Aggiornamento parametri
    
    history.append(loss_jit(xx, yy, params))     # Dopo ogni epoca, registra perdita su tutto il dataset
    cb(epoch)

cb.draw()
print("loss: %1.3e" % history[-1])

# --- TERZA SEZIONE: MOMENTUM - SGD con momentum per accelerare convergenza

n_training_points = 100
noise = 0.05
layers_size = [1, 5, 5, 1]
num_epochs = 20000
learning_rate_max = 1e-1
learning_rate_min = 1e-2
learning_rate_decay = 10000
batch_size = 10
alpha = 0.9 

xx, yy = get_training_data(n_training_points, noise)
params = initialize_params(layers_size)

loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.grad(loss, argnums=2))

history = list()
history.append(loss_jit(xx, yy, params))

cb = Callback(refresh_rate=250)

velocity = [0.0 for _ in range(len(params))] # Inizializza velocità (velocity) a zero per ogni parametro
for epoch in range(num_epochs):
    lr = max(learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay)) # Decadimento LR
    
    perm = np.random.permutation(n_training_points)     # Shuffle
    for i in range(0, n_training_points, batch_size):
        batch_idx = perm[i : i + batch_size]
        x_batch = xx[batch_idx]
        y_batch = yy[batch_idx]
        grads = grad_jit(x_batch, y_batch, params)
        # Aggiornamento con momentum: v = alpha * v - lr * g, p = p + v
        # (Nota: segno negativo perché discesa)
        velocity = jax.tree_util.tree_map(lambda v, g: alpha * v - lr * g, velocity, grads)
        params = jax.tree_util.tree_map(lambda p, v: p + v, params, velocity)
    
    history.append(loss_jit(xx, yy, params))
    cb(epoch)

cb.draw()
print("loss: %1.3e" % history[-1])

# --- QUARTA SEZIONE: RMSProp - Adattivo, divide per radice quadrata media gradiente

n_training_points = 100
noise = 0.05
layers_size = [1, 5, 5, 1]
num_epochs = 20000
batch_size = 10
learning_rate = 1e-1 
delta = 1e-7

xx, yy = get_training_data(n_training_points, noise)
params = initialize_params(layers_size)

loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.grad(loss, argnums=2))

history = list()
history.append(loss_jit(xx, yy, params))

cb = Callback(refresh_rate=250)

cumulated_square_grad = [0.0 for i in range(len(params))] # Inizializza accumulatore quadrato gradiente a zero
for epoch in range(num_epochs):
    perm = np.random.permutation(n_training_points) # Shuffle
    for i in range(0, n_training_points, batch_size):
        batch_idx = perm[i : i + batch_size]
        x_batch = xx[batch_idx]
        y_batch = yy[batch_idx]
        grads = grad_jit(x_batch, y_batch, params)
        # Aggiorna accumulatore: E[g^2] = decay_rate * E[g^2] + (1-decay_rate) * g^2
        # Qui decay_rate = 0.9 implicito? No, nel codice è fisso, ma tipicamente 0.9
        # Nel codice originale non c'è decay_rate, forse errore, ma procedo con accumulo semplice
        cumulated_square_grad = jax.tree_util.tree_map(lambda e, g: e + g**2, cumulated_square_grad, grads)
        # Aggiornamento: p = p - lr * g / sqrt(E[g^2] + delta)
        params = jax.tree_util.tree_map(lambda p, g, e: p - learning_rate * g / (jnp.sqrt(e) + delta), params, grads, cumulated_square_grad)
    
    history.append(loss_jit(xx, yy, params))
    cb(epoch)

cb.draw()
print("loss: %1.3e" % history[-1])

cb.draw()
print("loss: %1.3e" % history[-1])

# --- QUINTA SEZIONE: ADAM - Combina momentum e adattività (RMSProp + momentum)

n_training_points = 100
noise = 0.05
layers_size = [1, 5, 5, 1]
num_epochs = 20000
batch_size = 50  # Batch più grande
learning_rate = 1e-3  # LR più piccolo per Adam
decay_rate = 0.9  # Tasso di decadimento per medie mobili
delta = 1e-7  # Epsilon per stabilità

xx, yy = get_training_data(n_training_points, noise)
params = initialize_params(layers_size)

loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.grad(loss, argnums=2))

history = list()
history.append(loss_jit(xx, yy, params))

cb = Callback(refresh_rate=250)

# Inizializza accumulatori per Adam: m (primo momento), v (secondo momento)
cumulated_square_grad = [0.0 for i in range(len(params))]  # v
velocity = [0.0 for _ in range(len(params))]  # m
for epoch in range(num_epochs):
    perm = np.random.permutation(n_training_points) # Shuffle
    for i in range(0, n_training_points, batch_size):
        batch_idx = perm[i : i + batch_size]
        x_batch = xx[batch_idx]
        y_batch = yy[batch_idx]
        grads = grad_jit(x_batch, y_batch, params)
        # Aggiorna momenti: m = beta1 * m + (1-beta1) * g, v = beta2 * v + (1-beta2) * g^2
        # Qui beta1=0.9, beta2=decay_rate=0.9 (tipicamente beta2=0.999, ma qui semplificato)
        velocity = jax.tree_util.tree_map(lambda m, g: decay_rate * m + (1 - decay_rate) * g, velocity, grads)
        cumulated_square_grad = jax.tree_util.tree_map(lambda v, g: decay_rate * v + (1 - decay_rate) * g**2, cumulated_square_grad, grads)
        # Aggiornamento: p = p - lr * m / (sqrt(v) + delta)
        params = jax.tree_util.tree_map(lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + delta), params, velocity, cumulated_square_grad)
    
    history.append(loss_jit(xx, yy, params))
    cb(epoch)

cb.draw()
print("loss: %1.3e" % history[-1])