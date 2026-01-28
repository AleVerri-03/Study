import jax
import jax.numpy as jnp
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# SVR CLASS
class SVR:
    def __init__(self, epsilon=0.1, lmbda=1.0):
        self.epsilon = epsilon # "Margine insensibilità"
        self.lmbda = lmbda # "Peso reolarizzazione L2"
        self.w = None # "Pesi"

    def loss(self, params, X, y):
        predictions = X.reshape((-1, self.n_features)) @ params[:-1] + params[-1]
        epsilon_loss = jnp.maximum(0, jnp.abs(predictions - y) - self.epsilon)
        reg_term = self.lmbda * jnp.sum(params**2)
        return reg_term + jnp.mean(epsilon_loss)

    def train(self, X, y, lr=1e-2, max_iter=1000):
        _, self.n_features = X.shape
        self.w = jnp.zeros(self.n_features + 1)
        grad_fn = jax.grad(self.loss, argnums=0)
        @jax.jit
        def step(w):
            return w - lr * grad_fn(w, X, y)
        for _ in range(max_iter):
            self.w = step(self.w)

    def predict(self, X):
        return X.reshape((-1, self.n_features)) @ self.w[:-1] + self.w[-1]
    
# 1)Generate synthesis data

np.random.seed(0)  # For reproducibility
m = 2.5  
c = 1.0 
n_samples = 100

X = np.random.uniform(0, 10, size=(n_samples, 1))

noise = np.random.normal(0, 1, size=(n_samples, 1))
y = m * X + c + noise
y = y.flatten() # make mono-dimensional

# 2)Slit the data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

# 3)Create SVR Instance and train. Then make prediction

svr = SVR(epsilon=1.0, lmbda=0.1)
svr.train(X_train, y_train)

y_pred_train = svr.predict(X_train)
y_pred_test = svr.predict(X_test)

# 4)Evaluation using MSE
mse_train = jnp.mean((y_train - y_pred_train) ** 2)
mse_test = jnp.mean((y_test - y_pred_test) ** 2)
print(f"Train MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# 5)Plot the data
plt.figure(figsize=(10, 6))

plt.scatter(X_train, y_train, color="blue", label="Training data")
plt.scatter(X_test, y_test, color="green", marker="x", label="Test data")

x_range = jnp.linspace(0, 10, 100)
y_pred_line = svr.predict(x_range)
plt.plot(x_range, y_pred_line, color="red", label="SVR prediction")
plt.fill_between(
    x_range,
    y_pred_line - svr.epsilon,
    y_pred_line + svr.epsilon,
    label="SVR tube",
    color="r",
    alpha=0.1,
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("SVR on Synthetic Data with Gaussian Noise")
plt.legend()
plt.show()


# --- 

# SVM Class
class SVM:
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
        self.w = None

    def loss(self, params, X, y):
        print(X.shape, params.shape, y.shape)
        decision = X @ params[:-1] + params[-1]
        loss_val = jnp.maximum(0, 1 - y * decision)
        reg_term = self.lmbda * jnp.sum(params**2)
        return reg_term + jnp.mean(loss_val)

    def train(self, X, y, lr=1e-2, max_iter=1000):
        _, self.n_features = X.shape
        self.w = jnp.zeros(self.n_features + 1)
        grad_fn = jax.grad(self.loss, argnums=0)
        @jax.jit
        def step(w):
            return w - lr * grad_fn(w, X, y)
        for _ in range(max_iter):
            self.w = step(self.w)

    def predict(self, X):
        decision = X @ self.w[:-1] + self.w[-1]
        return jnp.sign(decision)
    
# 1) Generate data

np.random.seed(42)
n_samples = 100

X = np.random.uniform(0, 10, size=(n_samples, 2))

y = np.where(X.sum(axis=1) > 10, 1, -1) #y_i = -1 or 1
y = y.flatten()

# 2) Split the data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

# 3) Train and prediction

svm = SVM(lmbda=0.001)
svm.train(X_train, y_train, lr=1e-1, max_iter=5000)
print("Loss: ", svm.loss(svm.w, X_train, y_train))
print(svm.w)

y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)

# 4) Accuracy 

accuracy_train = jnp.mean(y_pred_train == y_train)
accuracy_test = jnp.mean(y_pred_test == y_test)
print(f"Train Accuracy: {accuracy_train:.4f}")
print(f"Test Accuracy: {accuracy_test:.4f}")

# 5) Visualize

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label="Training data", marker="o")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, label="Test data", marker="x")

t = jnp.linspace(0, 10, 1000)
xx1, xx2 = jnp.meshgrid(t, t)
xx = jnp.stack([xx1.flatten(), xx2.flatten()], axis=1)
yy = svm.predict(xx)
plt.contourf(xx1, xx2, yy.reshape(xx1.shape), alpha=0.1)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("SVM Decision Boundary on Synthetic Data")
plt.legend(loc="upper right")
plt.show()