# Numerical Analysis for Machine Learning

---

## Table of Contents

1. [Linear Algebra Fundamentals](#lesson-23-09)
2. [Eigenvalue Decomposition and QR](#lesson-26-09)
3. [Singular Value Decomposition](#lesson-29-09)
4. [Noise, Thresholding, and Cumulative Energy](#lesson-30-09)
5. [Eckart-Young-Mirsky and PCA](#lesson-07-10)
6. [Least Squares and Kernel Methods](#lesson-13-10)
7. [PageRank and Matrix Completion](#lesson-20-10)
8. [Automatic Differentiation](#lesson-21-10)
9. [Neural Networks and Backpropagation](#lesson-27-10)
10. [Cross-Entropy, Regularization, and Weight Initialization](#lesson-03-11)
11. [Gradient Descent Convergence](#lesson-04-11)
12. [First-Order Optimization Methods](#lesson-10-11)
13. [Second-Order and Quasi-Newton Methods](#lesson-17-11)
14. [L-BFGS and Levenberg-Marquardt](#lesson-25-11)
15. [Functional Analysis Foundations](#lesson-02-12)
16. [Universal Approximation Theorem](#lesson-09-12)
17. [Physics-Informed Neural Networks](#lesson-16-12)

---

# Lesson 23-09

## Linear Algebra Fundamentals

### Column Space and Rank

The **column space** $C(A)$ of a matrix $A$ is the span of its columns. The **rank** of a matrix is the dimension of its column space:

$$\text{rank}(A) = \dim(C(A))$$

### CR Factorization

Any matrix $A$ of rank $r$ can be factored as:

$$A = CR$$

where:
- $C$ contains $r$ linearly independent columns of $A$
- $R$ contains the coefficients to reconstruct all columns

### Orthogonal Matrices

A matrix $Q$ is **orthogonal** if:

$$Q^T Q = QQ^T = I$$

**Properties:**
- Columns form an orthonormal basis
- $\|Qx\|_2 = \|x\|_2$ (length preserving)
- Eigenvalues have magnitude 1

---

# Lesson 26-09

## Eigenvalue Decomposition

For a square matrix $A$, if $Av = \lambda v$, then $\lambda$ is an **eigenvalue** and $v$ is an **eigenvector**.

### Spectral Theorem

For symmetric matrices $A = A^T$:

$$A = Q\Lambda Q^T$$

where $Q$ is orthogonal and $\Lambda$ is diagonal with real eigenvalues.

### QR Decomposition

Any matrix $A$ can be factored as:

$$A = QR$$

where $Q$ is orthogonal and $R$ is upper triangular.

### Gram-Schmidt Process

Given vectors $\{a_1, \ldots, a_n\}$, produce orthonormal vectors $\{q_1, \ldots, q_n\}$:

1. $q_1 = \frac{a_1}{\|a_1\|}$
2. For $k = 2, \ldots, n$:
   - $\tilde{q}_k = a_k - \sum_{j=1}^{k-1} (q_j^T a_k) q_j$
   - $q_k = \frac{\tilde{q}_k}{\|\tilde{q}_k\|}$

### Symmetric Positive Definite (SPD) Matrices

A symmetric matrix $A$ is **positive definite** if:

$$x^T A x > 0 \quad \forall x \neq 0$$

Equivalently, all eigenvalues are positive.

---

# Lesson 29-09

## Singular Value Decomposition (SVD)

### Fundamental Theorem

Every matrix $A \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$A = U\Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$ is orthogonal (left singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$ is orthogonal (right singular vectors)

### Types of SVD

| Type | Form | Size |
|------|------|------|
| Full SVD | $A = U\Sigma V^T$ | $U: m \times m$, $V: n \times n$ |
| Reduced SVD | $A = U_r \Sigma_r V_r^T$ | Truncated to rank $r$ |
| Compact SVD | Non-zero singular values only | |

### Relationship to Eigenvalues

- $A^T A = V \Sigma^2 V^T$ (eigenvalue decomposition)
- $AA^T = U \Sigma^2 U^T$
- Singular values: $\sigma_i = \sqrt{\lambda_i(A^T A)}$

---

# Lesson 30-09

## Noise and Thresholding

### Marchenko-Pastur Law

For random matrices with i.i.d. entries, singular values follow a specific distribution with bounds:

$$\sigma_{\pm} = \sqrt{n}\left(1 \pm \sqrt{\frac{m}{n}}\right)$$

### Cumulative Energy

The cumulative energy captured by the first $k$ singular values:

$$E_k = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{n} \sigma_i^2}$$

Use this to choose the number of components to retain.

### Matrix Norms

**Frobenius norm:**
$$\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\sum_{i} \sigma_i^2}$$

**Spectral norm (operator norm):**
$$\|A\|_2 = \sigma_1 = \max_{\|x\|=1} \|Ax\|$$

---

# Lesson 07-10

## Eckart-Young-Mirsky Theorem

The best rank-$k$ approximation to $A$ in both Frobenius and spectral norms is given by truncated SVD:

$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

**Error bounds:**
$$\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$
$$\|A - A_k\|_2 = \sigma_{k+1}$$

## Principal Component Analysis (PCA)

### Algorithm

1. Center the data: $\tilde{X} = X - \bar{X}$
2. Compute covariance matrix: $C = \frac{1}{n-1}\tilde{X}^T\tilde{X}$
3. Compute eigendecomposition: $C = V\Lambda V^T$
4. Project onto top $k$ eigenvectors: $Z = \tilde{X}V_k$

### Connection to SVD

For centered data matrix $\tilde{X}$:
$$\tilde{X} = U\Sigma V^T$$

The principal components are the columns of $V$.

### Python Implementation

```python
import numpy as np

def pca(X, k):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project onto top k components
    return X_centered @ eigenvectors[:, :k]
```

---

# Lesson 13-10

## Least Squares

### Problem Formulation

Given $Ax = b$ with no exact solution, find:

$$x^* = \arg\min_x \|Ax - b\|_2^2$$

### Normal Equations

The solution satisfies:

$$A^T A x = A^T b$$

If $A$ has full column rank:
$$x^* = (A^T A)^{-1} A^T b = A^+ b$$

where $A^+ = (A^T A)^{-1} A^T$ is the **pseudoinverse**.

### Ridge Regression (Tikhonov Regularization)

Add regularization to prevent overfitting:

$$x^* = \arg\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2$$

**Solution:**
$$x^* = (A^T A + \lambda I)^{-1} A^T b$$

## Kernel Methods

### The Kernel Trick

Replace inner products with kernel functions:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

Common kernels:
- **Linear:** $K(x, y) = x^T y$
- **Polynomial:** $K(x, y) = (x^T y + c)^d$
- **RBF (Gaussian):** $K(x, y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$

### Representer Theorem

The solution to regularized empirical risk minimization has the form:

$$f^*(x) = \sum_{i=1}^{n} \alpha_i K(x_i, x)$$

---

# Lesson 20-10

## PageRank Algorithm

### Problem Setup

Given a web graph with transition matrix $P$, find the stationary distribution $\pi$:

$$\pi^T P = \pi^T$$

### Power Method

Starting from initial vector $\pi^{(0)}$:

$$\pi^{(k+1)} = P^T \pi^{(k)}$$

Converges to principal eigenvector (with eigenvalue 1).

### Google Matrix

To handle dangling nodes and ensure convergence:

$$G = \alpha P + (1-\alpha)\frac{1}{n}\mathbf{1}\mathbf{1}^T$$

where $\alpha \approx 0.85$ is the damping factor.

## Matrix Completion

### Netflix Problem

Recover a low-rank matrix $M$ from partial observations.

### Nuclear Norm Minimization

$$\min_X \|X\|_* \quad \text{subject to} \quad X_{ij} = M_{ij} \text{ for observed } (i,j)$$

where $\|X\|_* = \sum_i \sigma_i(X)$ is the nuclear norm.

---

# Lesson 21-10

## Automatic Differentiation

### Four Methods of Differentiation

| Method | Pros | Cons |
|--------|------|------|
| Manual | Exact | Error-prone, tedious |
| Symbolic | Exact, closed-form | Expression swell |
| Numerical (finite diff) | Easy to implement | Approximation, numerical errors |
| Automatic | Exact, efficient | Requires framework |

### Forward Mode

Propagate derivatives forward through computation graph using **dual numbers**:

$$a + b\epsilon \quad \text{where } \epsilon^2 = 0$$

**Computational complexity:** $O(n)$ function evaluations for $n$ inputs.

### Reverse Mode (Backpropagation)

Propagate derivatives backward from output to inputs.

**Computational complexity:** $O(m)$ function evaluations for $m$ outputs.

For neural networks with many inputs, few outputs: **reverse mode is more efficient**.

### Wengert List (Tape)

Store intermediate values during forward pass for use in backward pass:

```
v_{-1} = x_1
v_0 = x_2
v_1 = v_{-1} * v_0
v_2 = sin(v_1)
...
```

---

# Lesson 27-10

## Neural Networks

### Perceptron

Single neuron with threshold activation:

$$y = \sigma(w^T x + b)$$

### Feedforward Network

Layer-by-layer computation:

$$a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$$

### Matrix Notation

For a batch of inputs $X$:

$$A^{(l)} = \sigma(W^{(l)} A^{(l-1)} + B^{(l)})$$

## Backpropagation

### The Four Fundamental Equations

**BP1 - Error at output layer:**
$$\delta^{(L)} = \nabla_a C \odot \sigma'(z^{(L)})$$

**BP2 - Error propagation:**
$$\delta^{(l)} = ((W^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})$$

**BP3 - Gradient w.r.t. biases:**
$$\frac{\partial C}{\partial b^{(l)}} = \delta^{(l)}$$

**BP4 - Gradient w.r.t. weights:**
$$\frac{\partial C}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

### Activation Functions

| Function | Formula | Derivative |
|----------|---------|------------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ |
| Tanh | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ |
| ReLU | $\max(0, z)$ | $\mathbf{1}_{z > 0}$ |
| Mish | $z \cdot \tanh(\ln(1 + e^z))$ | Complex |

---

# Lesson 03-11

## Cross-Entropy Cost

For classification problems:

$$C = -\frac{1}{n}\sum_x [y \ln a + (1-y) \ln(1-a)]$$

**Advantages over quadratic cost:**
- Faster learning when far from correct answer
- No vanishing gradient issue with sigmoid

## Regularization

### L2 Regularization (Weight Decay)

$$\tilde{C} = C + \frac{\lambda}{2n}\sum_w w^2$$

Effect: Weights decay by factor $(1 - \frac{\eta\lambda}{n})$ each step.

### L1 Regularization

$$\tilde{C} = C + \frac{\lambda}{n}\sum_w |w|$$

Effect: Produces sparse weights (many exactly zero).

### Dropout

During training, randomly set fraction $p$ of neurons to zero.

At test time, scale weights by $(1-p)$.

## Weight Initialization

### Problem with Standard Initialization

With $N(0, 1)$ weights and many inputs, $z = \sum w_i x_i$ has large variance, saturating activation functions.

### Solution

Initialize weights with:

$$w \sim N\left(0, \frac{1}{n_{in}}\right)$$

or Xavier initialization:

$$w \sim N\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

---

# Lesson 04-11

## Gradient Descent Convergence

### L-Smoothness

A function $f$ is **L-smooth** if:

$$f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x}) + \frac{L}{2} \|\mathbf{y} - \mathbf{x}\|^2$$

### μ-Strong Convexity

A function $f$ is **μ-strongly convex** if:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x}) + \frac{\mu}{2} \|\mathbf{y} - \mathbf{x}\|^2$$

### Convergence Rates

| Function Class | Error Rate | Iterations for error ε |
|----------------|------------|------------------------|
| Lipschitz Convex | $O(1/\sqrt{T})$ | $O(1/\epsilon^2)$ |
| Smooth Convex | $O(1/T)$ | $O(1/\epsilon)$ |
| Smooth & Strongly Convex | $O((1-\mu/L)^T)$ | $O(\kappa \log(1/\epsilon))$ |

where $\kappa = L/\mu$ is the **condition number**.

### Line Search Methods

**Exact Line Search:**
$$\gamma_k = \arg\min_{\gamma>0} f(\mathbf{x}_k - \gamma \nabla f(\mathbf{x}_k))$$

**Backtracking Line Search (Armijo Condition):**
While $f(\mathbf{x}_k + \gamma \mathbf{d}_k) > f(\mathbf{x}_k) + c \gamma \nabla f(\mathbf{x}_k)^T \mathbf{d}_k$:
- $\gamma \leftarrow \tau \gamma$

## Projected Gradient Method

For constrained optimization $\min_{x \in C} f(x)$:

$$\mathbf{x}_{k+1} = \mathcal{P}_C(\mathbf{x}_k - \gamma_k \nabla f(\mathbf{x}_k))$$

where $\mathcal{P}_C$ is the projection onto set $C$.

---

# Lesson 10-11

## First-Order Optimization Methods

### Momentum

$$\mathbf{v}_t = \mu \mathbf{v}_{t-1} + \gamma \nabla J(\mathbf{w})$$
$$\mathbf{w} = \mathbf{w} - \mathbf{v}_t$$

### Nesterov Accelerated Gradient (NAG)

$$\mathbf{v}_t = \mu \mathbf{v}_{t-1} + \gamma \nabla J(\mathbf{w} - \mu \mathbf{v}_{t-1})$$
$$\mathbf{w} = \mathbf{w} - \mathbf{v}_t$$

### Adagrad

Adapts learning rate per-parameter:

$$w_{t+1, i} = w_{t, i} - \frac{\gamma}{\sqrt{G_{t,ii} + \epsilon}} \cdot g_{t,i}$$

where $G_{t,ii} = \sum_{\tau=1}^t g_{\tau,i}^2$.

**Problem:** Learning rate decays to zero over time.

### RMSprop

Uses exponentially decaying average:

$$E[g^2]_t = \mu E[g^2]_{t-1} + (1-\mu)g_t^2$$
$$w_{t+1} = w_t - \frac{\gamma}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

### Adam

Combines momentum and RMSprop:

**Update biased moments:**
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$$

**Bias correction:**
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$
$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

**Update:**
$$w_{t+1} = w_t - \frac{\gamma}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

Default values: $\gamma = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$.

---

# Lesson 17-11

## Second-Order Methods

### Newton's Method

For optimization:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - [D^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

**Pros:** Quadratic convergence near minimum
**Cons:** $O(n^3)$ per iteration, requires Hessian

## Quasi-Newton Methods

### Key Idea

Approximate the Hessian $B_k \approx D^2 f(\mathbf{x}_k)$ using gradient information only.

### Secant Condition

The approximation must satisfy:

$$B_{k+1} \boldsymbol{\delta}_k = \boldsymbol{\gamma}_k$$

where:
- $\boldsymbol{\delta}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$
- $\boldsymbol{\gamma}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$

### SR1 (Symmetric Rank-1) Update

$$B_{k+1} = B_k + \frac{(\boldsymbol{\gamma}_k - B_k \boldsymbol{\delta}_k)(\boldsymbol{\gamma}_k - B_k \boldsymbol{\delta}_k)^T}{(\boldsymbol{\gamma}_k - B_k \boldsymbol{\delta}_k)^T \boldsymbol{\delta}_k}$$

**Problem:** Does not guarantee positive definiteness.

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

Symmetric rank-2 update that maintains positive definiteness.

**Properties:**
- $O(n^2)$ per iteration
- Superlinear convergence
- Guarantees descent direction

---

# Lesson 25-11

## L-BFGS (Limited-Memory BFGS)

### Key Idea

Store only the last $m$ curvature pairs $\{(\boldsymbol{\delta}_i, \boldsymbol{\gamma}_i)\}$ instead of full $n \times n$ matrix.

### Comparison

| Feature | BFGS | L-BFGS |
|---------|------|--------|
| Memory | $O(n^2)$ | $O(mn)$ |
| CPU per iteration | $O(n^2)$ | $O(mn)$ |
| Convergence | Superlinear | Linear (but fast) |
| Use case | Small/medium $n$ | Large scale $n > 1000$ |

## Levenberg-Marquardt Algorithm

For nonlinear least squares:

$$\min_{\mathbf{w}} \sum_{i=1}^m [r_i(\mathbf{w})]^2$$

### Update Rule

Solve:
$$(\mathbf{J}^T \mathbf{J} + \lambda \mathbf{I}) \boldsymbol{\delta}_k = -\mathbf{J}^T \mathbf{r}$$

- **Small λ:** Gauss-Newton (fast, less stable)
- **Large λ:** Gradient descent (slow, stable)

### Adaptive Strategy

- If step reduces error: Accept step, decrease λ
- If step increases error: Reject step, increase λ

## Convolution

### Convolution Theorem

Convolution in time/spatial domain equals element-wise multiplication in frequency domain:

$$\mathbf{c} * \mathbf{d} = \text{IDFT}(\text{DFT}(\mathbf{c}) \odot \text{DFT}(\mathbf{d}))$$

Complexity: $O(n \log n)$ using FFT (vs. $O(n^2)$ direct).

---

# Lesson 02-12

## Functional Analysis Foundations

### Vector Spaces

A **vector space** $V$ is a set with addition and scalar multiplication satisfying:
- Commutativity, Associativity
- Distributivity
- Zero element exists

### Inner Products

An **inner product** $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ satisfies:
1. $\langle \mathbf{u}, \mathbf{u} \rangle \geq 0$ (equality iff $\mathbf{u} = \mathbf{0}$)
2. $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$ (symmetry)
3. Linearity in first argument

**Example in $C([0,1])$:**
$$\langle f, g \rangle = \int_0^1 f(x)g(x) \, dx$$

### Norms

A **norm** $\|\cdot\| : V \to \mathbb{R}$ satisfies:
1. Triangle inequality: $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$
2. Homogeneity: $\|\lambda \mathbf{u}\| = |\lambda| \|\mathbf{u}\|$
3. Positivity: $\|\mathbf{u}\| \geq 0$, equality iff $\mathbf{u} = \mathbf{0}$

### Cauchy-Schwarz Inequality

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$$

### Banach and Hilbert Spaces

- **Banach Space:** Complete normed vector space
- **Hilbert Space:** Complete inner product space

Every Hilbert space is a Banach space (but not vice versa).

### Lebesgue Integration

Lebesgue integration partitions the **range** instead of the domain:

$$\int f \, d\mu \approx \sum_i \alpha_i \mu(S_i)$$

where $S_i = \{x : f(x) \in [\alpha_i, \alpha_{i+1}]\}$.

**Key fact:** Countable sets have measure zero.

### $L^p$ Spaces

$$L^p(\Omega) = \left\{ f : \|f\|_{L^p} < \infty \right\}$$

$$\|f\|_{L^p} = \left( \int_\Omega |f(x)|^p \, dx \right)^{1/p}$$

$L^2$ is a Hilbert space with inner product:
$$\langle f, g \rangle_{L^2} = \int_\Omega f(x)g(x) \, dx$$

### Weak Derivatives

A function $g$ is the **weak derivative** of $u$ if:

$$\int_\Omega g(x)\phi(x) \, dx = -\int_\Omega u(x)\phi'(x) \, dx$$

for all test functions $\phi \in \mathcal{D}(\Omega)$.

### Sobolev Spaces

$$W^{k,p}(\Omega) = \{u \in L^p(\Omega) : D^\alpha u \in L^p(\Omega) \text{ for } |\alpha| \leq k\}$$

$H^1(\Omega) = W^{1,2}(\Omega)$ is a Hilbert space.

---

# Lesson 09-12

## Universal Approximation Theorem

### Statement

A neural network with a single hidden layer can approximate any continuous function to any desired precision.

### Discriminatory Functions

An activation function $\sigma$ is **discriminatory** if the only measure $\mu$ satisfying:

$$\int \sigma(\mathbf{w}^T \mathbf{x} + \theta) \, d\mu(\mathbf{x}) = 0 \quad \forall \mathbf{w}, \theta$$

is $\mu = 0$.

### Key Result

Any continuous sigmoidal function is discriminatory, and so is ReLU.

### Proof Sketch (by contradiction)

1. Assume network outputs are not dense in $C(I_n)$
2. By Hahn-Banach, exists non-zero functional $L$ that vanishes on all network outputs
3. By Riesz representation, $L$ corresponds to non-zero measure $\mu$
4. This contradicts discriminatory property of activation function

### Constructive Proof with ReLUs

1. ReLUs can create step functions (large weights)
2. Combine ReLUs to create "hat" (triangular) functions
3. Sum hat functions to approximate any continuous function
4. This is analogous to Finite Element Method with P1 basis functions

## Depth vs. Width

### Curse of Dimensionality

For shallow networks with input dimension $d$ and smoothness $r$:

$$N_s \approx \epsilon^{-d/r}$$

Exponential in dimension!

### Advantage of Depth

For compositional functions with local dimension $\tilde{d}$:

$$N_d \approx d \cdot \epsilon^{-\tilde{d}/r}$$

Linear in dimension!

### Example

With $d = 1000$, $\tilde{d} = 2$, $r = 10$, $\epsilon = 10^{-2}$:
- Shallow: $N_s \approx 10^{200}$ (infeasible)
- Deep: $N_d \approx 2500$ (practical)

---

# Lesson 16-12

## Physics-Informed Neural Networks (PINNs)

### Core Idea

Embed physical laws (PDEs) directly into the neural network loss function.

### PDE Formulation

$$f(\mathbf{x}, u, \nabla u, \nabla^2 u, \ldots; \lambda) = 0, \quad \mathbf{x} \in \Omega$$

with boundary conditions:
$$\mathcal{B}(u, \mathbf{x}) = 0, \quad \mathbf{x} \in \partial\Omega$$

### Loss Function

$$L(\theta) = w_f L_f(\theta) + w_b L_b(\theta) + w_u L_u(\theta)$$

where:
- $L_f$: PDE residual loss (physics)
- $L_b$: Boundary condition loss
- $L_u$: Data loss (if available)

### PDE Residual Loss

$$L_f(\theta) = \frac{1}{|\mathcal{T}_f|} \sum_{\mathbf{x} \in \mathcal{T}_f} \left\| f(\mathbf{x}, \hat{u}, \nabla \hat{u}, \ldots) \right\|^2_2$$

Derivatives computed via **Automatic Differentiation**.

### Advantages

- **Mesh-free:** No need for domain discretization
- **Unified framework:** Same approach for forward and inverse problems
- **Data-physics balance:** Works with sparse data

### Error Decomposition

$$E = \underbrace{\mathcal{E}_{app}}_{\text{Approximation}} + \underbrace{\mathcal{E}_{gen}}_{\text{Generalization}} + \underbrace{\mathcal{E}_{opt}}_{\text{Optimization}}$$

### PINN vs FEM

| Aspect | PINN | FEM |
|--------|------|-----|
| Basis function | Neural network (nonlinear) | Polynomial (linear) |
| Training points | Scattered (mesh-free) | Mesh points |
| PDE embedding | Loss function | Algebraic system |
| Error bounds | Not available | Partially available |

### Boundary Conditions

**Soft constraints:** Add boundary loss term (flexible, general)

**Hard constraints:** Construct ansatz that automatically satisfies BCs:
$$\hat{u}(x) = x(x-1) N(x; \theta)$$

### Extensions

- **Integro-Differential Equations:** Use numerical quadrature for integral terms
- **Residual-Based Adaptive Refinement (RAR):** Add points where residual is high

---

## Summary

This document covers the mathematical foundations of machine learning, from linear algebra basics through advanced optimization methods to neural network theory. Key topics include:

1. **Matrix decompositions:** SVD, eigenvalue decomposition, QR factorization
2. **Dimensionality reduction:** PCA and its connection to SVD
3. **Optimization:** From gradient descent convergence theory to adaptive methods (Adam) and second-order methods (BFGS, L-BFGS)
4. **Neural networks:** Backpropagation, activation functions, regularization
5. **Theoretical foundations:** Functional analysis, Universal Approximation Theorem
6. **Scientific ML:** Physics-Informed Neural Networks

The depth of treatment progresses from practical algorithms to theoretical guarantees, providing a comprehensive foundation for understanding modern machine learning methods.
