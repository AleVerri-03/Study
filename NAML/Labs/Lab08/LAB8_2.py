import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

# Enable double precision in JAX for better numerical accuracy
jax.config.update("jax_enable_x64", True)

# Set up the problem: solve a linear system Ax = b using optimization
# Generate a random n x n matrix A, exact solution x_ex, and compute b = A @ x_ex
n = 100
np.random.seed(0)
A = np.random.randn(n, n)
x_ex = np.random.randn(n)
b = A @ x_ex

# Define the loss function as the sum of squared residuals: ||Ax - b||^2
def loss(x):
    return jnp.sum(jnp.square(A @ x - b))

# Print the loss at the exact solution (should be 0)
print(loss(x_ex))

# Compute the gradient and Hessian of the loss function using JAX automatic differentiation
grad = jax.grad(loss) # Gradient Function
hess = jax.jacfwd(jax.jacrev(loss)) # Hessian Function

# JIT compile the functions for faster execution
loss_jit = jax.jit(loss)
grad_jit = jax.jit(grad)
hess_jit = jax.jit(hess)

# Result Check
# Verify that the computed gradient and Hessian match the analytical expressions
np.random.seed(0)
x_guess = np.random.randn(n)
G_ad = grad_jit(x_guess)  # Automatic differentiation gradient
G_ex = 2 * A.T @ (A @ x_guess - b)  # Analytical gradient for least squares
print(np.linalg.norm(G_ad - G_ex))  # Should be close to 0
H_ad = hess_jit(x_guess)  # Automatic differentiation Hessian
H_ex = 2 * A.T @ A  # Analytical Hessian for least squares
print(np.linalg.norm(H_ad - H_ex))  # Should be close to 0

# Define Hessian-vector product (HVP) for matrix-free methods
# gvp computes the gradient-vector product: grad(x)^T * v
gvp = lambda x, v: jnp.dot(grad(x), v)
# hvp computes the Hessian-vector product using automatic differentiation
hvp = lambda x, v: jax.grad(gvp, argnums=0)(x, v)
hvp_basic = hvp  # Define hvp_basic as the same as hvp (likely for timing comparison)
hvp_basic_jit = jax.jit(hvp_basic)  # JIT compiled basic HVP
hvp_jit = jax.jit(hvp)  # JIT compiled HVP
# Define a random vector v for testing HVP
v = np.random.randn(n)
Hv_ad = hvp_jit(x_guess, v)  # HVP computed via AD
Hv_ex = H_ex @ v  # HVP computed analytically
print(np.linalg.norm(Hv_ad - Hv_ex))  # Should be close to 0

# Time comparison between basic and JIT-compiled HVP
%timeit hvp_basic_jit(x_guess, v)
%timeit hvp_jit(x_guess, v)

# ---
# Newton method to minimize loss function
# Newton's method uses second-order information (Hessian) for faster convergence
x = x_guess.copy()  # Start from initial guess
num_epochs = 100
eps = 1e-8  # Tolerance for convergence

for epoch in range(num_epochs):
    H = hess_jit(x)  # Compute Hessian at current x
    G = grad_jit(x)  # Compute gradient at current x
    incr = np.linalg.solve(H, -G)  # Solve H * incr = -G for the increment
    x += incr  # Update x

    print("============ epoch %d" % epoch)
    print("loss: %1.3e" % loss_jit(x))
    print("grad: %1.3e" % np.linalg.norm(G))  # Norm of gradient
    print("incr: %1.3e" % np.linalg.norm(incr))  # Norm of increment

    if np.linalg.norm(incr) < eps:  # Check convergence
        break

# Compute relative error compared to exact solution
rel_err = np.linalg.norm(x - x_ex) / np.linalg.norm(x_ex)
print(f"Relative error: {rel_err:1.3e}")

# Solve the system (Hessian Matrix-free version)
# Use conjugate gradient (CG) to solve H * incr = -G without explicitly forming H
# This is memory-efficient for large problems
x = x_guess.copy()
num_epochs = 100
eps = 1e-8

for epoch in range(num_epochs):
    G = grad_jit(x)
    # Use CG to solve lambda y: hvp_jit(x, y) * incr = -G
    incr, info = jax.scipy.sparse.linalg.cg(lambda y: hvp_jit(x, y), -G, tol=eps)
    x += incr

    print("============ epoch %d" % epoch)
    print("loss: %1.3e" % loss_jit(x))
    print("grad: %1.3e" % np.linalg.norm(G))
    print("incr: %1.3e" % np.linalg.norm(incr))

    if np.linalg.norm(incr) < eps:
        break

rel_err = np.linalg.norm(x - x_ex) / np.linalg.norm(x_ex)
print(f"Relative error: {rel_err:1.3e}")

# Now optimization loop for loss function
# Change the loss to sum of fourth powers for a different problem (non-quadratic)
def loss(x):
    return jnp.sum((A @ x - b) ** 4)  # Fourth power loss, more challenging to optimize

# Redefine gradient and Hessian for the new loss
grad = jax.grad(loss)
hess = jax.jacfwd(jax.jacrev(loss))

# JIT compile again
loss_jit = jax.jit(loss)
grad_jit = jax.jit(grad)
hess_jit = jax.jit(hess)

# Run Newton method on the new loss
x = x_guess.copy()
num_epochs = 100
eps = 1e-8

hist = [loss_jit(x)]  # History of loss values
for epoch in range(num_epochs):
    hist.append(loss_jit(x))  # Note: this appends before update, might be misplaced

    H = hess_jit(x)
    G = grad_jit(x)
    incr = np.linalg.solve(H, -G)
    x += incr
    
    if np.linalg.norm(incr) < eps:
        print("convergence reached!")
        break

# Plot the loss history on a semilog scale
plt.semilogy(hist, "o-")
print("epochs: %d" % epoch)
print("relative error: %1.3e" % (np.linalg.norm(x - x_ex) / np.linalg.norm(x_ex)))

# --- 
# Quasi-Newton Method
# Minimize objective function with BFGS update
# BFGS approximates the inverse Hessian iteratively, avoiding full Hessian computation
max_epochs = 1000
tol = 1e-8  # Tolerance for gradient norm

import scipy as sp  # For line search

np.random.seed(0)

epoch = 0
x = x_guess.copy()
I = np.eye(x.size)  # Identity matrix
Binv = I  # Initial approximation of inverse Hessian (identity)
grad = grad_jit(x_guess)
history = [loss_jit(x_guess)]  # Loss history

while np.linalg.norm(grad) > tol and epoch < max_epochs:
    epoch += 1
    p = -Binv @ grad  # Search direction: -Binv * grad

    # Line search to find step size alpha
    alpha = sp.optimize.line_search(loss_jit, grad_jit, x, p)[0]
    alpha = 1e-8 if alpha is None else alpha  # Fallback if line search fails
    x_new = x + alpha * p  # Update x

    # Compute differences for BFGS update
    s = x_new - x  # Step vector
    x = x_new
    grad_new = grad_jit(x_new)
    y = grad_new - grad  # Gradient difference
    grad = grad_new

    # Sherman-Morrison update for inverse Hessian approximation
    rho = 1.0 / (np.dot(y, s))
    E = I - rho * np.outer(y, s)
    Binv = E.T @ Binv @ E + rho * np.outer(s, s)

    history.append(loss_jit(x))

# Plot the loss history
plt.semilogy(history, "o-")