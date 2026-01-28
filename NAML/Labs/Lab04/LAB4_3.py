import jax
import jax.numpy as jnp
import numpy as np

from jax import lax
import matplotlib.pyplot as plt

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)

# JAX arrays are immutable!
size = 10
index = 0
value = 23

# In NumPy arrays are mutable
x = np.arange(size)
print(x)
x[index] = value
print(x)

x = jnp.arange(size)
print(x)
# x[index] = value  # ERROR!

# If this seems wasteful to you, it is indeed true (in normal settings).
# The thing is, if the input value x of x.at[idx].set(y) is not reused,
# you can tell JAX to optimize the array update to occur in-place.
# We will see the details after

jax_array = jnp.zeros((3, 3), dtype=jnp.float32)
updated_array = jax_array.at[1, :].set(1.0)

print("original array unchanged:\n", jax_array)
print("updated array:\n", updated_array)

# The expresiveness of NumPy is still there!

print("new array post-addition:")
new_jax_array = jax_array.at[::2, 1:].add(7.0) # Raplace 7.0 in add positions and from 1 to the last
print(new_jax_array)

# Fact 3: JAX handles random numbers differently (for the same reason arrays are immutable)
seed = 0
key = jax.random.PRNGKey(seed) # key to random created

x = jax.random.normal(key, (10,))  # you need to explicitly pass the key i.e. PRNG state
print(type(x), x)  # notice the DeviceArray type - that leads us to the next cell!

# Fact 4: JAX is AI accelerator agnostic. Same code runs everywhere!

size = 3000

# Data is automatically pushed to the AI accelerator (GPU, TPU)
x_jnp = jax.random.normal(key, (size, size), dtype=jnp.float32)
x_np = np.random.normal(size=(size, size)).astype(np.float32)  # some diff in API exists!

print("[1] GPU")
# %timeit jnp.dot(x_jnp, x_jnp.T).block_until_ready()  # 1) on GPU - fast
print("[2] Pure numpy (CPU)")
# %timeit np.dot(x_np, x_np.T)  # 2) on CPU - slow (NumPy only works with CPUs)
print("[3] GPU + data transfer")
# %timeit jnp.dot(x_np, x_np.T).block_until_ready()  # 3) on GPU with transfer overhead

x_np_device = jax.device_put(x_np)  # push NumPy explicitly to GPU
print("[4] GPU + explicit pre-data transfer (like [1] but explicit)")
# %timeit jnp.dot(x_np_device, x_np_device.T).block_until_ready()  # same as 1)


# TRANSFORM FUNCTION
def visualize_fn(fn, l=-10, r=10, n=1000):
  x = np.linspace(l, r, num=n)
  y = fn(x)
  plt.plot(x, y); plt.show()

# Define a function
def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
# x > 0 -> LMBDA * x
# x < 0 -> lmdba * (alpha * exp(x) - alpha)

selu_jit = jax.jit(selu)  # let's jit it to BEST PERFORMANCE

# Visualize SELU (just for your understanding, it's always a good idea to visualize stuff)
visualize_fn(selu)

# Benchmark non-jit vs jit version
data = jax.random.normal(key, (1000000,))

print('non-jit version:')
#%timeit selu(data).block_until_ready()
print('jit version:')
#%timeit selu_jit(data).block_until_ready()



x = 1.0

f = lambda x: x**2 + x + 4 # callable object
visualize_fn(f, l=-1, r=2, n=100)

dfdx = jax.grad(f)  # 2*x + 1
d2fdx = jax.grad(dfdx)  # 2
d3fdx = jax.grad(d2fdx)  # 0

print(f(x), dfdx(x), d2fdx(x), d3fdx(x))


f = lambda x, y: x**2 + y**2 

def eval_hessian(f): # Calc Hessian matrix
    return jax.jit(jax.jacfwd(jax.jacrev(f, argnums=(0, 1)), argnums=(0, 1)))
#jacrev -> gradient respect x and y
#jacfwd -> second derivative of gradient
#jit -> best performance
#argmus -> declare what variables as to be derivated

jacobian = jax.jacrev(f, argnums=(0, 1))(1.0, 1.0) # evaluated in x=1 y=1
hessian = eval_hessian(f)(1.0, 1.0)
print(f"Jacobian = {np.asarray(jacobian)}")
print(f"Hessian = {np.asarray(hessian)}")


f = lambda x: x[0] * x[0] + x[1] * x[1] # Array input


def eval_hessian(f):
    return jax.jit(jax.jacfwd(jax.jacrev(f)))

x0 = jnp.array([2.0, 1.0])
jacobian = jax.jacrev(f)(x0)
hessian = eval_hessian(f)(x0)
print(f"Jacobian = {np.asarray(jacobian)}")
print(f"Hessian = {np.asarray(hessian)}")

f = lambda x: abs(x)
visualize_fn(f)

dfdx = jax.grad(f)
print(f"dfdx(0.0)   = {dfdx(0.0)  :.17e}")
print(f"dfdx(+1e-5) = {dfdx(+1e-5):.17e}")
print(f"dfdx(-1e-5) = {dfdx(-1e-5):.17e}")

# --- dot product between vectors

def custom_dot(x, y):
    return jnp.dot(x, y) ** 2

def naive_custom_dot(x_batched, y_batched):
    return jnp.stack([
        custom_dot(v1, v2)
        for v1, v2 in zip(x_batched, y_batched)
    ])

@jax.jit
def jit_naive_custom_dot(x_batched, y_batched):
    return jnp.stack([
        custom_dot(v1, v2)
        for v1, v2 in zip(x_batched, y_batched)
    ])

batched_custom_dot = jax.vmap(custom_dot, in_axes=[0, 0])
jit_batched_custom_dot = jax.jit(jax.vmap(custom_dot, in_axes=[0, 0]))

x = jnp.asarray(np.random.rand(1000, 50)) # 1000 array, dim = 50
y = jnp.asarray(np.random.rand(1000, 50))

print("Naive")
# %timeit naive_custom_dot(x, y)
print("Vectorized")
# %timeit batched_custom_dot(x, y)
print("JIT")
# %timeit jit_naive_custom_dot(x, y)
print("Vectorized + JIT")
# %timeit jit_batched_custom_dot(x, y)

# --- 

# Example 1: lax is stricter
# jax.numpy (jnp) si comporta come numpy: se sommi un intero e un float, promuove automaticamente il tipo a float
print(jnp.add(1, 1.0)) 
# jax.lax è più "basso livello" e più rigoroso: accetta solo tipi uguali
print(lax.add(1.0, 1.0))
print(lax.add(1, 1.0))  # ERRORE! jax.lax richiede tipi uguali (qui int e float)

# Example 2: lax è più potente (ma meno user-friendly)
# Esempio di convoluzione: vogliamo convolvere due vettori
x = jnp.array([1, 2, 1])
y = jnp.ones(10)

# Con jax.numpy (come numpy):
result1 = jnp.convolve(x, y)  # semplice, sintassi familiare

# Con jax.lax: si usa una funzione più generale e "basso livello"
# Bisogna esplicitamente:
# - fare il reshape degli array per aggiungere dimensioni batch/canale
# - convertire i tipi se necessario
# - specificare stride e padding manualmente
result2 = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # reshape a (batch=1, canale=1, lunghezza=3) e cast a float
    y.reshape(1, 1, 10),               # reshape a (batch=1, canale=1, lunghezza=10)
    window_strides=(1,),               # stride di 1
    padding=[(len(y) - 1, len(y) - 1)],# padding "full" come in numpy
)  # output shape: (1, 1, 12)

print(result1)       
print(result2[0][0]) 
assert np.allclose(result1, result2[0][0], atol=1e-6)  # verifica che i risultati coincidano

# --- 

# Example: JAX JIT limitations with dynamic slicing
def get_negatives(x):
    return x[x < 0]  # Restituisce solo i valori negativi dell'array x



# Genera 10 numeri casuali (normali) con JAX
x = jax.random.normal(key, (10,), dtype=jnp.float32)
print(get_negatives(x))  # Funziona normalmente
# print(jax.jit(get_negatives)(x))  # ERRORE! JIT non supporta slicing booleano dinamico

# Example: What happens when you use print inside a JIT-compiled function?
# The print statements are executed only during compilation (tracing), not during execution.
@jax.jit
def f(x, y):
    print("Running f():")  # Questo print viene eseguito solo durante la compilazione, non ad ogni chiamata!
    print(f"  x = {x}")
    print(f"  y = {y}")
    result = jnp.dot(x + 1, y + 1)
    print(f"  result = {result}")
    return result


# Prima chiamata: la funzione viene compilata e i print vengono eseguiti
x = np.random.randn(3, 4)
y = np.random.randn(4)
print(f(x, y))  # La prima chiamata compila la funzione e mostra i print

# Seconda chiamata: la funzione è già compilata, i print NON vengono più eseguiti!
x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
print("Second call:")
print(f(x2, y2))  # I print non appaiono più

# Same function as before but without print (more "JAX-friendly")
def f(x, y):
    return jnp.dot(x + 1, y + 1)

# Visualize the JAXPR representation of the function (trace of operations)
print(jax.make_jaxpr(f)(x, y))

# 2nd example of a failure: conditional control flow depending on runtime values
@jax.jit
def f(x, neg):  # result depends on the value - remember tracer cares about shapes and types!
    return -x if neg else x

@jax.jit
def f2(x, neg):
    return x * (1.0 - 2.0 * neg)

f2(x, False)
# f(1, True) # ERROR!

# Soluzione: dichiarare argomenti "statici" (noti a compile-time)
from functools import partial

@partial(jax.jit, static_argnums=(1,))  # "neg" è statico: ogni valore diverso ricompila la funzione
def f(x, neg):
    print(x)
    return -x if neg else x

print(f(1, True))   # Compila per (True)
print(f(2, True))   # Usa la compilazione precedente
print(f(2, False))  # Compila per (False)
print(f(23, False)) # Usa la compilazione precedente
print(f(44, 1))     # Compila per (1)


# Example 3: Impure functions using global variables

g = 0.0


def impure_uses_globals(x):
    return x + g  # Violating both #1 and #2


# JAX captures the value of the global during the first run
print("First call: ", jax.jit(impure_uses_globals)(4.0))

# Let's update the global!
g = 10.0

# Subsequent runs may silently use the cached value of the globals
print("Second call: ", jax.jit(impure_uses_globals)(5.0))

# JAX re-runs the Python function when the type or shape of the argument changes
# This will end up reading the latest value of the global
print("Third call, different type: ", jax.jit(impure_uses_globals)(jnp.array([4.0])))

# JAX behavior with out-of-bounds access:
# 1) updates at out-of-bounds indices are skipped
# 2) retrievals result in index being clamped
# In general, there are currently some bugs so consider the behavior undefined!

print(jnp.arange(10).at[11].add(23))  # example of 1)
print(jnp.arange(10)[11])  # example of 2)


# ===================== Differenze PRNG: NumPy vs JAX =====================
# NumPy: PRNG (random) è "stateful" (lo stato cambia ad ogni estrazione)
seed = 0
np.random.seed(seed)
rng_state = np.random.get_state()
print("numpy random state", rng_state[2:])
print("A random number", np.random.random())
rng_state = np.random.get_state()
print("numpy random state",rng_state[2:])
print("A random number", np.random.random())
rng_state = np.random.get_state()
print("numpy random state",rng_state[2:])

# JAX: PRNG è "stateless"! Devi gestire tu la chiave (key) manualmente
key = jax.random.PRNGKey(seed)
print("JAX rng state", key)  # key = stato (2 uint32)
print("A random number", jax.random.normal(key, shape=(1,)))
print("JAX rng state",key)  # la key NON cambia!
print("A random number", jax.random.normal(key, shape=(1,)))  # stesso risultato di prima!
print("JAX rng state",key)
# Soluzione: ogni volta che vuoi un nuovo random, "splitta" la key
print("old key", key)
key, subkey = jax.random.split(key, 2)
normal_pseudorandom = jax.random.normal(subkey, shape=(1,))
print("    \\---SPLIT --> new key   ", key)
print("             \\--> new subkey", subkey, "--> normal", normal_pseudorandom)


# ===================== Esempio di non-determinismo con NumPy e thread =====================
import threading
import time

# Simula training su CPU/GPU #1
def worker1(result_container):
    for _ in range(10):
        result_container["worker1"].append(np.random.uniform())
        time.sleep(0.001)

# Simula training su CPU/GPU #2
def worker2(result_container):
    for _ in range(10):
        result_container["worker2"].append(np.random.uniform())
        time.sleep(0.002)

# Funzione che lancia entrambi i worker in parallelo e somma i risultati
def do_parallel_work():
    result = {"worker1": [], "worker2": []}
    t1 = threading.Thread(target=worker1, args=(result,))
    t2 = threading.Thread(target=worker2, args=(result,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return (np.array(result["worker1"]) + 2 * np.array(result["worker2"]).sum())

# Se eseguiamo più volte, otteniamo risultati diversi (non determinismo!)
for i in range(5):
    np.random.seed(42)
    print(f"Run {i+1}: {do_parallel_work()}")

# Python control flow + grad() -> everything is ok
def f(x):
    if x < 3:
        return 3.0 * x**2
    else:
        return -4 * x


x = np.linspace(-10, 10, 1000)
y = [f(el) for el in x]

print(jax.grad(f)(2.0))  # ok!
print(jax.grad(f)(4.0))  # ok!

# Python control flow + jit() -> problems in paradise.

# "The tradeoff is that with higher levels of abstraction we gain a more general view
# of the Python code (and thus save on re-compilations),
# but we require more constraints on the Python code to complete the trace."

# Solution (recall: we already have seen this)
f_jit = jax.jit(f, static_argnums=(0,))
print(f_jit(x))

# Esempio: uso di variabili globali (NON raccomandato con JAX)
# WARNING: still for each x we have to jit a new function, this might be expensive!
def impure_uses_globals(x):
    return x + g  # JAX "cattura" il valore della globale solo alla prima compilazione

f_jit_error = jax.jit(f)
f_jit_error(2.0)

# native jax functions give you powerfull alternatives

def f(x):
    return jnp.where(x < 3.0, 3.0 * x**2, -4 * x)

f_jit = jax.jit(f)
fgrad_jit = jax.jit(jax.grad(f))

print(f_jit(2.0))
print(f_jit(4.0))
print(fgrad_jit(2.0))
print(fgrad_jit(4.0))

# Example 2: range depends on value again


def f(x, n):
    y = 0.0
    for i in range(n):
        y = y + x[i]
    return y


f_jit = jax.jit(f, static_argnums=(1,))
x = (jnp.array([2.0, 3.0, 4.0]), 3)

print(jax.make_jaxpr(f_jit, static_argnums=(1,))(*x))  # notice how for loop gets unrolled
print(f_jit(*x))

# Note: there is a catch - static args should not change a lot!

# Even "better" solution is to use low level API
def f_fori(x, n):
    body_fun = lambda i, val: val + x[i]
    return lax.fori_loop(0, n, body_fun, 0.0)


f_fori_jit = jax.jit(f_fori)

print(jax.make_jaxpr(f_fori_jit)(*x))
print(f_fori_jit(*x))

# If you want to debug where the NaNs are coming from, there are multiple ways
# to do that, here is one:
jax.config.update("jax_debug_nans", False)  # Change this flag and re-run the cell
jnp.divide(0.0, 0.0)

# JAX enforces single precision! There are simple ways around it though.
jax.config.update("jax_enable_x64", True)
x = jax.random.uniform(key, (1000,), dtype=jnp.float64)
print(x.dtype)