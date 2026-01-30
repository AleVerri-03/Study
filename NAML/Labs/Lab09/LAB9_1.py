import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg
import time

v = np.zeros(100)
v[50:75] = 1

plt.plot(v)

### first kernel
k = np.ones(10) / 10 # Filtro o maschera per elaborare i dati

### second kernel
# k = signal.gaussian(20, std=3)
# k = k / np.sum(k)

### third kernel
# k = np.array([-1,2,-1])

plt.plot(k)

# --- 4 Convolutions (between v and k)

# --- Toeplitz matrix k*v=Kv
t0 = time.time()

# SOLUTION-BEGIN
k_padded = np.zeros(len(v) + len(k) - 1)
k_padded[: len(k)] = k
K = linalg.toeplitz(k_padded, np.zeros(len(v)))
v_conv1 = np.matmul(K, v)
# SOLUTION-END

print("Execution time: %1.2e s" % (time.time() - t0))

plt.plot(v_conv1)

# --- Direct definition
# Kernel decresce, scorro kernel al contrario, evito bordi

t0 = time.time()

l_out = len(v) - len(k) + 1
v_conv2 = np.zeros(l_out)
for i in range(l_out):
    # for j in range(len(k)):
    #    v_conv2[i] += k[-j] * v[i + j]
    v_conv2[i] = np.sum(np.flip(k) * v[i : i + len(k)])

print("Execution time: %1.2e s" % (time.time() - t0))

plt.plot(v_conv2)

# --- Convolution through DFT
# When signal are same size, we use FFT
# Convoluzione time domain == moltiplicazione frequency domain

from numpy.fft import ifft, fft, fftshift, fftfreq

t0 = time.time()

v_fft = fft(v) # From time to frequency
k_fft = fft(k, len(v))

vk_fft = v_fft * k_fft # Hadamard product

v_conv3 = np.real(ifft(vk_fft)) # Frequency to time

print("Execution time: %1.2e s" % (time.time() - t0))

freq = fftfreq(len(v))

fig, axs = plt.subplots(2, 2, figsize=(22, 8))
axs[0, 0].plot(fftshift(freq), fftshift(np.absolute(v_fft)))
axs[0, 0].set_title("FFT of v")
axs[0, 1].plot(fftshift(freq), fftshift(np.absolute(k_fft)))
axs[0, 1].set_title("FFT of k")
axs[1, 0].plot(fftshift(freq), fftshift(np.absolute(vk_fft)))
axs[1, 0].set_title("FFT of v * k")
axs[1, 1].plot(v_conv3)
axs[1, 1].set_title("v * k")

# --- Command

t0 = time.time()

v_conv4 = signal.convolve(v, k, mode="valid")

print("Execution time: %1.2e s" % (time.time() - t0))

plt.plot(v_conv4)