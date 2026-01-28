import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Set random seed for reproducibility
key = jax.random.PRNGKey(0)

# Plotting helper function
#   Visualize optimization trajectories on 2D contour plot
#   1) Optimization path od GD
#   2) Optimization path with "BackTricaking Line Search"
def plot_optimization_2d(func, gd_path, gd_backtrack_path, title):
    x_vals = jnp.linspace(-5, 5, 50)
    y_vals = jnp.linspace(-5, 5, 50)
    X, Y = jnp.meshgrid(x_vals, y_vals)
    Z = jnp.array([[func(jnp.array([x, y])) for x in x_vals] for y in y_vals])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    cs = axs[0].contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(cs)
    axs[0].contour(X, Y, Z, colors="white")

    gd_path = jnp.array(gd_path)

    axs[0].plot(gd_path[:, 0], gd_path[:, 1], "r.-", label="GD")

    if gd_backtrack_path != []:
        gd_backtrack_path = jnp.array(gd_backtrack_path)
        axs[0].plot(
            gd_backtrack_path[:, 0],
            gd_backtrack_path[:, 1],
            ".-",
            color="orange",
            label="GD + backtracking",
        )
    axs[0].set_title(title)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_ylim([-5, 5])
    axs[0].set_xlim([-5, 5])
    axs[0].legend()

    axs[1].semilogy([func(x) for x in gd_path], "ro-", label="GD")
    axs[1].semilogy(
        [func(x) for x in gd_backtrack_path],
        "o-",
        color="orange",
        label="GD + backtracking",
    )
    axs[1].legend()
    plt.tight_layout()

    # Define the benchmark functions


# 1. Rastrigin function
@jax.jit
def rastrigin(x):
    return 10 * x.size + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x)) + 1e-10


# 2. Ackley function
@jax.jit
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * jnp.pi
    sum1 = jnp.sum(x**2)
    sum2 = jnp.sum(jnp.cos(c * x))
    return (
        -a * jnp.exp(-b * jnp.sqrt(sum1 / x.size))
        - jnp.exp(sum2 / x.size)
        + a
        + jnp.exp(1)
    )


# 3. Quadratic function
quadratic_A = jnp.array([[3.0, 0.5], [0.5, 1.0]])
quadratic_b = jnp.array([-1.0, 2.0])
quadratic_c = jnp.dot(quadratic_b, jnp.linalg.solve(quadratic_A, quadratic_b)) / 2


@jax.jit
def quadratic(x):
    return (
        0.5 * jnp.dot(x.T, jnp.dot(quadratic_A, x))
        + jnp.dot(quadratic_b, x)
        + quadratic_c
    )

# Find the minimum of a function
def gradient_descent(grad_func, x0, lr=0.01, tol=1e-6, max_iter=1000):
    x = x0
    path = [x]
    for _ in range(max_iter):
        grad_val = grad_func(x) # calc gradient
        x = x - lr * grad_val # update x going opposite to the gradient
        path.append(x) # save new x to the path
        if jnp.linalg.norm(grad_val) < tol:
            break
    return x, path


def gradient_descent_backtracking(
    func, grad_func, x0, alpha=0.3, beta=0.8, tol=1e-6, max_iter=100
):
    x = x0
    path = [x]
    for _ in range(max_iter):
        grad_val = grad_func(x)
        t = 1.0 # initial step 
        while func(x - t * grad_val) > func(x) - alpha * t * jnp.dot( # Armijo condition
            grad_val, grad_val # We need that the new value is enough low   
        ):
            t *= beta
        x = x - t * grad_val # now we have a valid t, update x
        path.append(x)
        if jnp.linalg.norm(grad_val) < tol:
            break
    return x, path

# Specific for quadratic
def exact_line_search_quadratic(A, b, x0, tol=1e-6, max_iter=100):
    x = x0
    path = [x]
    for _ in range(max_iter):
        grad_val = jnp.dot(A, x) + b
        t = jnp.dot(grad_val, grad_val) / jnp.dot(grad_val, jnp.dot(A, grad_val)) # Calculate a specific t
        x -= t * grad_val
        path.append(x)
        if jnp.linalg.norm(grad_val) < tol:
            break
    return x, path


# --- Testing functions
x0 = jnp.array([4.0, 4.0])

test_functions = [
    (rastrigin, "Rastrigin"),
    (ackley, "Ackley"),
    (quadratic, "Quadratic"),
]

for func, name in test_functions:
    grad_func = jax.jit(jax.grad(func))
    print(f"Testing on {name} function")

    x_gd, path_gd = gradient_descent(grad_func, x0)
    x_gd_bt, path_gd_bt = gradient_descent_backtracking(func, grad_func, x0)

    plot_optimization_2d(func, path_gd, path_gd_bt, title=f"{name}")

x_exact, path_exact = exact_line_search_quadratic(quadratic_A, quadratic_b, x0)
plot_optimization_2d(
    quadratic, path_exact, [], title="Quadratic Function - Exact Line Search"
)

# --- Test linear regression with SVG

import numpy as np
np.random.seed(0)

N = 200
x_data = np.random.uniform(size=(N,)) * 10 # array 200 val -> between 0 - 10
y_data = 1.5 * x_data + 3 + np.random.normal(size=(N,)) #x * 1.5 + 3 + Noise

plt.scatter(x_data, y_data) # Preview

from sklearn.model_selection import train_test_split

# Training test function
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

# Define the linear regression model (Linear Function)
@jax.jit
def model(theta, x):
    return theta[0] + theta[1] * x

# Mean Squared Error loss function
@jax.jit
def mse_loss(theta, x, y):
    y_pred = model(theta, x)
    return jnp.mean((y_pred - y) ** 2)

grad_mse_loss = jax.jit(jax.grad(mse_loss))

# Update SGD on small part of dataset
@jax.jit
def sgd_update(theta, x_batch, y_batch, learning_rate):
    grads = grad_mse_loss(theta, x_batch, y_batch)
    return theta - learning_rate * grads

# Stochastic Gradient Descent with mini-batches
def stochastic_gradient_descent(
    theta,
    training_input,
    training_labels,
    validation_input,
    validation_labels,
    learning_rate=0.01,
    epochs=100,
    batch_size=10,
    state=0,
):
    key = jax.random.PRNGKey(state)
    # Iterate over the number of epochs
    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        # Shuffle data indices using JAX's random key
        perm = jax.random.permutation(subkey, len(training_input))
        # Process data in mini-batches
        for i in range(0, len(training_input), batch_size):
            batch_idx = perm[i : i + batch_size]
            x_batch = training_input[batch_idx]
            y_batch = training_labels[batch_idx]
            # Perform SGD update
            theta = sgd_update(theta, x_batch, y_batch, learning_rate)
        # Print the loss every 10 epochs
        if epoch % 10 == 0:
            loss = mse_loss(theta, validation_input, validation_labels)
            print(f"Epoch {epoch}, Loss: {loss:.4f} (test)")
    return theta

# Run the Stochastic Gradient Descent
# Initial guess for theta_0 (intercept) and theta_1 (slope)
theta = jnp.array([0.0, 0.0])
theta_opt = stochastic_gradient_descent(
    theta, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=100, batch_size=10
)
print(
    f"Optimized parameters: theta = [{theta_opt[0]:.2f}, {theta_opt[1]:.2f}]"
)

# After the training loop, use the optimized parameters to plot the regression line
# Generate predictions using the optimized parameters
y_pred = model(theta_opt, x_data)

# Plot the original data points
plt.scatter(X_train, y_train, label="Training points")
plt.scatter(X_test, y_test, label="Test points")

# Plot the regression line
plt.plot(x_data, y_pred, label="Fitted line", color="black", linewidth=2)

# Add labels and a legend
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Stochastic Gradient Descent (SDG)")
plt.legend()
plt.show()