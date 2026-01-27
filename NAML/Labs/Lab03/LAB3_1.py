import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def my_pinv_fullSVD(A):
    U, s, VT = np.linalg.svd(A)
    s[s > 0] = 1 / s[s > 0]
    Pinv = VT.transpose() @ la.diagsvd(s, A.shape[1], A.shape[0]) @ U.transpose()
    return Pinv

def my_pinv_thinSVD(A):
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    s[s > 0] = 1 / s[s > 0]
    Pinv =  (VT.transpose() * s) @ U.transpose()
    return Pinv

A = np.random.randn(5, 4)
Apinv_numpy = np.linalg.pinv(A)
Apinv_fullSVD = my_pinv_fullSVD(A)
Apinv_thinSVD = my_pinv_thinSVD(A)
print(np.linalg.norm(Apinv_numpy - Apinv_fullSVD) / np.linalg.norm(Apinv_numpy))
print(np.linalg.norm(Apinv_numpy - Apinv_thinSVD) / np.linalg.norm(Apinv_numpy))

# Linear model y = mx + q

m =2.0
q = 3.0
N = 100 # Numer of elements
noise = 2.0

np.random.seed(0)
X = np.random.randn(N) # N random points
Y = m * X + q + noise * np.random.randn(N) # Linear model + noise
plt.scatter(X, Y)
plt.plot(X, m*X+q, color="red")
plt.show()

# Build the phi
Phi = np.column_stack([X[:, np.newaxis], np.ones((N, 1))])
# X[:, np.newaxis] Trasformo in una colonna Nx1
# np.ones((N, 1)) Colonna di 1 lunga N
#np.column_stack([...]) Mette una colonna accanto all'altra Nx2
z = my_pinv_thinSVD(Phi) @ Y
m_hat = z[0]
q_hat = z[1]
print(f"m_hat = {m_hat:.3f}")
print(f"q_hat = {q_hat:.3f}")

plt.scatter(X, Y, alpha=0.5)
plt.plot(X, m*X+q, color="red")
plt.plot(X, m_hat*X+q_hat, color="k")
plt.show()

# EXACT SOLUTION
# z2 = np.linalg.solve(Phi.transpose() @ Phi, Phi.transpose() @ Y)
# np.linalg.norm(z - z2)


# f(x) = tanh(2x-1)
np.random.seed(0)

N = 100
noise = 0.1
y_ex = lambda x: np.tanh(2*(x-1)) # lambda function

X = np.random.randn(N, 1)
Y = y_ex(X) + noise * np.random.randn(N, 1)

N_test = 1000
X_test = np.linspace(-3, 3, N_test).reshape((-1, 1)) #linspaxe X on col
Y_test = y_ex(X_test)

plt.scatter(X, Y, marker="+", color="black", label="data")
plt.plot(X_test, Y_test, color="black", label="$y_{ex}(x)$")
plt.legend()
plt.show() # plot noised data

Phi = np.block([X, np.ones((N,1))])
z = my_pinv_thinSVD(Phi) @ Y
m_hat = z[0,0]
q_hat = z[1,0]
print(f"m_hat = {m_hat:.3f}")
print(f"q_hat = {q_hat:.3f}")
Y_test_LS = m_hat*X_test + q_hat

plt.scatter(X, Y, marker="+", color="black")
plt.plot(X_test, Y_test, color="black", label="$y_{ex}(x)$")
plt.plot(X_test, Y_test_LS, color="green", label="LS regression") # Best linear approximation
plt.legend()
# plt.show()

# RIDGE REGRESSION
lam = 1.0
PhiPhiT = Phi @Phi.transpose()
alpha = np.linalg.solve(PhiPhiT + lam * np.eye(N), Y) # ((Phi.T * Phi) + lam * I)^-1 * y
w = Phi.transpose() @ alpha
w2 = np.linalg.solve(Phi.T @ Phi + lam * np.eye(2), Phi.T @ Y)
Phi_test = np.block([X_test, np.ones((N_test, 1))])
Y_test_RR = Phi_test @ w

plt.scatter(X, Y, marker="+", color="black", label="data")
plt.plot(X_test, Y_test, color="black", label="$y_{ex}(x)$")
plt.plot(X_test, Y_test_LS, color="green", label="LS regression")
plt.plot(X_test, Y_test_RR, color="red", linestyle="--", label="Ridge regression")
plt.legend()

#KERNEL REGRESSION
lam = 1.0
q = 4.0
sigma = 1.0

def product_kernel(x1, x2):
    return x1 * x2 + 1

def high_order_kernel(x1, x2):
    return (x1 * x2 + 1) ** q

def gaussian_kernel(x1, x2):
    return np.exp(-(((x1 - x2) / sigma) ** 2) / 2)

def kernel_regression(kernel):
    N = X.shape[0] # row dimension of X
    K = np.empty((N, N)) # empy matrix
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(X[i], X[j]) # Compose the elemets of the matrix usign the operation I want
    alpha = np.linalg.solve(K + lam * np.eye(N), Y) # Solve (K + lam * I) alpha = y
    K_test = np.empty((X_test.shape[0], N))
    for i in range(K_test.shape[0]):
        for j in range(K_test.shape[1]):
            K_test[i, j] = kernel(X_test[i], X[j]) # K_test = (x_i_test, x_j)
    Y_test_KR = K_test @ alpha # Y = K_test * alpha
    return Y_test_KR

plt.scatter(X, Y, marker="+", color="black", label="data")
plt.plot(X_test, Y_test, color="black", label="$y_{ex}(x)$")
plt.plot(
    X_test,
    kernel_regression(gaussian_kernel),
    linestyle="--",
    label=f"Gaussian KR $\sigma = {sigma}$",
)
plt.plot(
    X_test,
    kernel_regression(product_kernel),
    linestyle=":",
    label=f"Scalar KR $q = 1$",
)
plt.plot(
    X_test,
    kernel_regression(high_order_kernel),
    linestyle=":",
    label=f"Scalar KR $q = {q}$",
)
plt.legend()
plt.ylim([-2, 2])