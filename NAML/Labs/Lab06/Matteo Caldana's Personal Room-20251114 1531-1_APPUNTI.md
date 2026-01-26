# 📝 Appunti: Matteo Caldana's Personal Room-20251114 1531-1
**Modello:** gemini-3-flash-preview | **Data:** 23/01/2026

---

## ⏱️ Min 0-30

# Analisi Numerica per il Machine Learning: Implementazione di Funzioni di Loss e Reti Neurali

Questi appunti coprono l'implementazione pratica della funzione di costo (Loss Function) per le Support Vector Machines (SVM) e la costruzione da zero di una rete neurale artificiale (ANN) per risolvere il problema logico dello XOR.

---

## 1. Funzione di Loss per SVM (Hinge Loss)

L'obiettivo iniziale è implementare correttamente la funzione di costo per una SVM con regolarizzazione.

### Formulazione Matematica
La funzione di costo $L(w)$ è definita come la **Hinge Loss** più un termine di regolarizzazione:
$$L(w) = \lambda \|w\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w^T x_i + b))$$

Dove:
- $w$: parametri del modello (pesi).
- $b$: termine di bias.
- $\lambda$: parametro di regolarizzazione.
- $y_i$: etichetta reale del campione $i$.
- $x_i$: vettore delle feature del campione $i$.

### Errore Comune: Il Bug del Broadcasting in NumPy
In fase di implementazione tramite `numpy`, un errore frequente riguarda la gestione delle dimensioni dei vettori e l'operazione di **broadcasting**.

1.  **Vettori 1D vs 2D**: In Python, un array di forma `(n,)` è un vettore riga/colonna generico (vettore 1D). Un array `(n, 1)` è esplicitamente un vettore colonna (matrice 2D).
2.  **Il Problema**: Se si calcola il termine di decisione $z_i = y_i(w^T x_i + b)$ e si forza una forma tramite `.reshape(-1, 1)`, si ottiene un vettore colonna.
3.  **Broadcasting Implicito**: Quando si moltiplica un vettore colonna `(10, 1)` per un vettore 1D `(10,)`, NumPy non esegue una moltiplicazione elemento per elemento (element-wise), ma applica il broadcasting creando una matrice di **prodotto esterno (outer product)** di dimensione `(10, 10)`.
4.  **Conseguenza**: Applicando successivamente la funzione `.mean()`, si calcola la media di 100 elementi invece di 10, portando a un valore di loss errato ma apparentemente plausibile, rendendo difficile l'individuazione del bug.

---

## 2. Implementazione di una Rete Neurale Artificiale (ANN)

Il laboratorio prosegue con l'implementazione "from scratch" di una rete neurale per apprendere la funzione **XOR (Exclusive OR)**.

### Il Problema dello XOR
Lo XOR è un classico problema di classificazione non lineare. Un semplice modello lineare (come un percettrone a singolo strato) non può separare le classi $[(0,0), (1,1)]$ (output 0) e $[(0,1), (1,0)]$ (output 1). È necessaria una rete neurale con almeno uno strato nascosto.

### Architettura della Rete
La rete considerata nel laboratorio ha la seguente struttura:
- **Input**: 2 nodi (coordinate $x_1, x_2$).
- **Hidden Layer 1**: 4 nodi con attivazione $tanh(\cdot)$.
- **Hidden Layer 2**: 3 nodi con attivazione $tanh(\cdot)$.
- **Output Layer**: 1 nodo con attivazione Sigmoide (per interpretare l'output come probabilità/likelihood tra 0 e 1).

### Iperparametri e Parametri
- **Iperparametri**: Definiscono l'architettura (numero di strati, numero di neuroni per strato, es: `n1=2, n2=4, n3=3, n4=1`).
- **Parametri ($\theta$)**: Sono i pesi ($W$) e i bias ($b$) che la rete deve apprendere tramite Gradient Descent.

---

## 3. Implementazione Pratica (Python/JAX)

### Inizializzazione dei Parametri
I pesi devono essere inizializzati correttamente (es. distribuzione normale standard) per evitare la saturazione delle funzioni di attivazione, mentre i bias possono essere inizializzati a zero.

```python
import numpy as np

# Inizializzazione pesi (W) e bias (b) per 3 strati
W1 = np.random.randn(n2, n1) # Da input a hidden 1
b1 = np.zeros(n2)

W2 = np.random.randn(n3, n2) # Da hidden 1 a hidden 2
b2 = np.zeros(n3)

W3 = np.random.randn(n4, n3) # Da hidden 2 a output
b3 = np.zeros(n4)
```

### Logica del Feed-forward
Ogni strato esegue una trasformazione affine seguita da una non linearità:
$$a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$$

**Dettagli sulle funzioni di attivazione:**
- **Tangente Iperbolica (`tanh`)**: Centrata in zero, range $[-1, 1]$. In JAX: `jnp.tanh`.
- **Sigmoide**: Utilizzata nell'ultimo strato per mappare l'output in $[0, 1]$. Può essere derivata dalla `tanh`:
  $$Sigmoid(x) = \frac{1 + \tanh(x/2)}{2}$$

### Ottimizzazione
L'obiettivo è minimizzare la funzione di loss calcolando il gradiente rispetto ai parametri $\theta$ attraverso la **Backpropagation** (automatizzata in librerie come JAX tramite `grad`). Il metodo utilizzato è il **Gradient Descent** iterativo:
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$
dove $\eta$ è il learning rate.

---

## ⏱️ Min 30-60

Ecco dei dettagliati appunti universitari basati sul segmento video riguardante l'implementazione di una Rete Neurale Artificiale (ANN) per risolvere il problema logico XOR utilizzando Python e la libreria JAX.

---

# Appunti: Implementazione di una ANN per la Funzione XOR

## 1. Obiettivo della Lezione
L'obiettivo è implementare una rete neurale artificiale (ANN) "fully connected" per apprendere la funzione logica **XOR (Exclusive OR)**. Verranno analizzati i passaggi per definire l'architettura, gestire le dimensioni delle matrici e implementare le funzioni di perdita (loss functions).

## 2. Struttura dei Dati e Convenzioni
- **Input ($X$):** Una matrice di tipo `numpy.array` con dimensioni `num_samples x 2`. Ogni riga rappresenta un campione (esempio: `[0, 0]`, `[0, 1]`, ecc.).
- **Parametri (`params`):** Una lista contenente pesi ($W$) e bias ($b$) per ogni layer.
- **Convenzione Matematiche:** In ambito accademico, spesso si vede $W \cdot x$. Se $X$ ha i campioni sulle righe, è necessario trasporre ($X^T$) per eseguire il prodotto matriciale correttamente come $W \cdot X^T$.

## 3. Definizione della Funzione ANN (Forward Pass)
La funzione `ANN(x, params)` calcola l'output della rete dato un input e un set di parametri.

### A. Unpacking dei Parametri
I parametri vengono estratti dalla lista per facilitarne l'uso:
```python
W1, b1, W2, b2, W3, b3 = params
```

### B. Calcolo dei Layer
La rete presentata nel video ha più layer con funzioni di attivazione **tangente iperbolica (`tanh`)**.

1.  **Layer 1:** 
    *   Operazione: $z_1 = W_1 \cdot X^T + b_1$
    *   Attivazione: $a_1 = \tanh(z_1)$
2.  **Layer successivi:** Il processo si ripete utilizzando l'output del layer precedente come input per quello successivo.

### C. Gestione del Bias e Broadcasting
Un punto critico evidenziato è l'aggiunta del bias ($b$). Per assicurarsi che il **broadcasting** di JAX funzioni correttamente aggiungendo il bias a ogni colonna (campione), i bias devono essere definiti come vettori colonna 2D (es. dimensioni `(n, 1)` invece di un array 1D `(n,)`).

### D. Layer di Output e Sigmoide
Per problemi di classificazione binaria come XOR, l'output finale deve essere una probabilità tra 0 e 1. Se l'attivazione dell'ultimo layer è una `tanh` (che spazia tra -1 e 1), è possibile convertirla in una funzione **Sigmoide** con la formula:
$$\sigma(z) = \frac{\tanh(z) + 1}{2}$$

### E. Output Finale e Trasposizione
L'output finale della funzione viene trasposto (`.T`) per ritornare alla convenzione originale (campioni sulle righe), rendendo l'output coerente con la struttura dei target $Y$.

```python
def ANN(x, params):
    W1, b1, W2, b2, W3, b3 = params
    layer2 = jnp.tanh(W1 @ x.T + b1)
    layer3 = jnp.tanh(W2 @ layer2 + b2)
    layer4 = jnp.tanh(W3 @ layer3 + b3)
    # Trasformazione in sigmoide per probabilità [0, 1]
    output = (layer4 + 1) / 2.0
    return output.T
```

## 4. Funzioni di Perdita (Loss Functions)

Una volta ottenute le predizioni, è necessario definire una funzione di errore per guidare l'addestramento tramite **Gradient Descent**.

### A. Quadratic Loss (Mean Squared Error - MSE)
Calcola la media del quadrato delle differenze tra predizione e target.
```python
def loss_quadratic(x, y, params):
    pred = ANN(x, params)
    return jnp.mean((pred - y)**2)
```

### B. Cross-Entropy Loss
Utilizzata comunemente nei compiti di classificazione perché penalizza maggiormente le predizioni errate fatte con "alta confidenza". La formula implementata è:
$$\mathcal{L} = -\sum [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

*   **Vantaggio:** Se il target è 1 e la rete predice un valore vicino a 0 (es. 0.01), il termine $\log(0.01)$ genera un valore negativo molto grande che, con il segno meno davanti, produce una perdita elevata, forzando un aggiornamento dei pesi più deciso.

```python
def loss_crossentropy(x, y, params):
    pred = ANN(x, params)
    # Implementazione della formula matematica della cross-entropy
    return -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))
```

## 5. Osservazioni sull'Inizializzazione
Nel video viene mostrato che testando la rete con parametri iniziali casuali, la tabella di verità prodotta non corrisponde a quella dello XOR (es. predice valori vicini a 0.5 per tutti gli input). Ciò dimostra la necessità della fase di **training** per ottimizzare i parametri $W$ e $b$.

---
*Termini Tecnici Chiave:*
- **Broadcasting:** Capacità delle librerie numeriche di eseguire operazioni tra array di dimensioni diverse.
- **Forward Pass:** Il percorso dei dati dall'input all'output attraverso i layer della rete.
- **Hyperbolic Tangent (tanh):** Funzione di attivazione non lineare che mappa i valori nell'intervallo (-1, 1).
- **Sigmoid:** Funzione di attivazione che mappa i valori nell'intervallo (0, 1).

---

## ⏱️ Min 60-90

Ecco degli appunti universitari dettagliati basati sul segmento video riguardante l'implementazione e l'addestramento di una rete neurale artificiale (ANN) utilizzando la libreria **JAX** per risolvere il problema logico dell'**XOR**.

---

# Appunti di Machine Learning: Implementazione di Reti Neurali con JAX

## 1. Implementazione della Funzione di Loss: Cross-Entropy
La Cross-Entropy (Entropia Incrociata) è la funzione di costo standard per problemi di classificazione binaria. Misura la discrepanza tra la distribuzione di probabilità predetta dal modello e quella reale.

### Formula Matematica
$$L = -\frac{1}{N} \sum_{i} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$
Dove:
*   $y_i$ è il valore target (0 o 1).
*   $\hat{y}_i$ è il valore predetto dal modello (output della sigmoide).

### Implementazione in JAX
Nel codice, viene utilizzata la funzione `jnp.mean` (invece di `jnp.sum`) per rendere la loss indipendente dalla dimensione del batch.
```python
def loss_crossentropy(x, y, params):
    pred = ANN(x, params)
    # Calcolo della cross-entropy binaria
    return -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))
```
*Nota: Viene inserito il segno meno poiché la loss deve misurare la discrepanza; minimizzare la loss equivale a massimizzare la somiglianza tra le distribuzioni.*

---

## 2. Compilazione JIT e Differenziazione Automatica
Per ottimizzare le prestazioni, JAX permette di compilare le funzioni tramite **JIT (Just-In-Time)** e di calcolare i gradienti automaticamente con `jax.grad`.

### Calcolo dei Gradienti
La funzione `jax.grad` calcola la derivata della loss rispetto ai parametri del modello.
*   **Dettaglio Tecnico (`argnums`)**: È fondamentale specificare l'indice dell'argomento rispetto al quale calcolare il gradiente. Se la firma della funzione è `loss(x, y, params)`, allora `argnums=2` indica che vogliamo il gradiente rispetto a `params`.

```python
# Compilazione JIT delle funzioni di loss
loss_mse_jit = jax.jit(loss_quadratic)
loss_xent_jit = jax.jit(loss_crossentropy)

# Calcolo dei gradienti compilati
grad_mse_jit = jax.jit(jax.grad(loss_quadratic, argnums=2))
grad_xent_jit = jax.jit(jax.grad(loss_crossentropy, argnums=2))
```

---

## 3. Algoritmo di Gradient Descent (GD)
Viene implementato un ciclo di addestramento basato sul discesa del gradiente "full batch" (l'intero dataset viene processato ad ogni epoca).

### Parametri di Addestramento
*   **Learning Rate ($\alpha$):** Impostato a $0.1$ ($10^{-1}$).
*   **Epoche:** $2000$ iterazioni per garantire la convergenza.

### Ciclo di Update
Ad ogni epoca, i parametri vengono aggiornati muovendosi nella direzione opposta al gradiente per minimizzare la loss:
$$\theta_{new} = \theta_{old} - \alpha \cdot \nabla_{\theta} L$$

```python
learning_rate = 0.1
epochs = 2000

for epoch in range(epochs):
    # 1. Calcolo del gradiente rispetto ai parametri correnti
    grads = grad_fn(inputs, outputs, params)
    
    # 2. Update dei parametri (Pesi e Bias)
    for i in range(len(params)):
        params[i] -= grads[i] * learning_rate
    
    # 3. Registrazione della storia della loss per monitoraggio
    history_mse.append(loss_mse_jit(inputs, outputs, params))
    history_xent.append(loss_xent_jit(inputs, outputs, params))
```

---

## 4. Analisi dei Risultati e Confronto delle Loss
Uno dei punti focali della lezione è il confronto tra l'utilizzo della **Mean Squared Error (MSE)** e della **Cross-Entropy (XEnt)** in un contesto di classificazione.

### Visualizzazione (Loss Curves)
Utilizzando una scala logaritmica sull'asse Y (`plt.yscale('log')`), si osserva l'andamento della convergenza:
*   **Cross-Entropy:** Tende a convergere molto più velocemente e a raggiungere valori di errore decisamente più bassi (fino a $10^{-6}$ per l'MSE riflesso) rispetto all'uso della sola funzione quadratica.
*   **Perché la Cross-Entropy è migliore?** Matematicamente, penalizza molto più severamente le predizioni "sicure ma errate" (quando il modello predice quasi 0 ma il target è 1, o viceversa). Questo porta a gradienti più forti che accelerano l'apprendimento nelle fasi critiche.

### Test Finale (Tabella di Verità XOR)
Dopo 2000 epoche con Cross-Entropy, il modello predice correttamente le coppie dell'XOR:
*   `0 XOR 0` $\rightarrow$ predizione $\approx 0.01$
*   `0 XOR 1` $\rightarrow$ predizione $\approx 0.98$
*   `1 XOR 0` $\rightarrow$ predizione $\approx 0.97$
*   `1 XOR 1` $\rightarrow$ predizione $\approx 0.02$

Questi risultati confermano che la rete neurale ha appreso con successo la natura non lineare della funzione XOR.

---

## Glossario Tecnico
*   **JIT (Just-In-Time):** Tecnica di compilazione che trasforma il codice Python/JAX in codice macchina ottimizzato per CPU/GPU al momento dell'esecuzione.
*   **Differenziazione Automatica:** Capacità di un framework di calcolare i derivati di funzioni complesse senza richiedere il calcolo manuale delle formule.
*   **Full Batch:** Metodo di addestramento che utilizza tutti i dati disponibili ad ogni iterazione del gradient descent.
*   **Iperparametro:** Parametro dell'algoritmo di apprendimento (es. learning rate, numero di epoche) che non viene appreso dai dati ma deve essere impostato manualmente.

---

## ⏱️ Min 90-120

Ecco degli appunti universitari dettagliati basati sul segmento video fornito, incentrati sulla valutazione delle Reti Neurali Artificiali (ANN) e sulla loro generalizzazione per problemi di classificazione complessi.

---

# Valutazione e Generalizzazione delle Reti Neurali Artificiali (ANN)

## 1. Valutazione del Modello XOR
Dopo l'addestramento di una rete neurale per risolvere il problema XOR, è fondamentale valutarne le prestazioni attraverso l'analisi delle perdite e il calcolo dell'accuratezza.

### 1.1 Analisi delle Perdite (Loss Analysis)
Il monitoraggio delle funzioni di perdita durante le epoche di addestramento rivela la convergenza del modello:
*   **MSE (Mean Squared Error):** Tende a diminuire costantemente.
*   **Xent (Cross-Entropy):** Spesso preferita per problemi di classificazione; in questo caso, entrambe mostrano una chiara tendenza alla convergenza verso valori prossimi a $10^{-6}$ dopo 2000 epoche.
*   **Visualizzazione:** L'uso di scale logaritmiche sull'asse delle ordinate ($y$) aiuta a visualizzare meglio la discesa del gradiente quando i valori diventano molto piccoli.

### 1.2 Calcolo dell'Accuratezza
Per convertire gli output probabilistici della rete in classi binarie (0 o 1), si applica una soglia (threshold):
*   **Logica:** Se la probabilità è $> 0.5$, la classe è `True` (1), altrimenti è `False` (0).
*   **Implementazione (JAX/Python):**
    ```python
    pred = ANN(inputs, params) > 0.5
    accuracy = jnp.sum(pred == y) / y.size
    ```
    *Un'accuratezza di 1.0 indica che tutte le predizioni del modello corrispondono ai target reali.*

### 1.3 Matrice di Confusione (Confusion Matrix)
Uno strumento diagnostico essenziale per la classificazione che mostra i conteggi di:
*   **Veri Positivi (TP) / Veri Negativi (TN)**
*   **Falsi Positivi (FP) / Falsi Negativi (FN)**

Utilizzando `sklearn.metrics.confusion_matrix`, una matrice perfetta per un problema bilanciato (come lo XOR con 4 campioni) presenterà valori solo sulla diagonale principale:
$$
\begin{bmatrix}
2 & 0 \\
0 & 2
\end{bmatrix}
$$
(Zero elementi sulla anti-diagonale indicano zero errori).

---

## 2. Generalizzazione del Processo: Il Dataset "Circles"
Per rendere il modello più realistico e complesso, si passa dal problema XOR a un dataset sintetico 2D composto da due cerchi concentrici.

### 2.1 Descrizione del Dataset
*   **Struttura:** Un cerchio interno più piccolo circondato da un anello (cerchio più grande) esterno.
*   **Scopo:** Classificare se un punto $(x, y)$ appartiene al cerchio interno (classe 1) o all'anello esterno (classe 0).
*   **Preparazione:**
    *   Generazione tramite `sklearn.datasets.make_circles`.
    *   **Train/Test Split:** Fondamentale per la validazione. Tipicamente si usa l'80% per il training e il 20% per il testing (`test_size=0.2`).
    *   **Visualizzazione:** I punti di training sono spesso rappresentati con cerchi, mentre i punti di test con "x" per distinguerli visivamente.

---

## 3. Implementazione Modulare di un MLP (Multi-Layer Perceptron)
Per gestire architetture flessibili con un numero variabile di strati e neuroni, è necessario un approccio funzionale e generalizzato.

### 3.1 Funzioni Core ("Ingredienti")
L'implementazione richiede diversi componenti chiave:

1.  **`init_layer_params(key, in_dim, out_dim)`**:
    Inizializza pesi ($W$) e bias ($b$) per un singolo strato utilizzando una distribuzione normale casuale (tramite JAX).
2.  **`init_mlp_params(key, layer_sizes)`**:
    Prende una lista di dimensioni (es. `[2, 16, 16, 1]`) e genera automaticamente una lista di parametri per l'intera rete neurale profonda.
3.  **Funzioni di Attivazione**:
    *   **Tangente Iperbolica (tanh):** Utilizzata negli strati nascosti.
    *   **Sigmoide:** Utilizzata nell'ultimo strato per mappare l'output tra 0 e 1: $\sigma(x) = \frac{1}{1 + e^{-x}}$.
4.  **`forward(params, x)`**:
    Esegue la propagazione in avanti (forward pass) utilizzando un ciclo `for` per iterare attraverso tutti gli strati, rendendo la rete estensibile a qualsiasi profondità.
5.  **`binary_cross_entropy(params, x, y)`**:
    Calcola la perdita di entropia incrociata per la classificazione binaria.
6.  **`update(params, x, y, lr)`**:
    Implementa lo **Stochastic Gradient Descent (SGD)** su mini-batch. Invece di usare l'intero dataset ad ogni iterazione, calcola il gradiente su un sottoinsieme (`batch_size`), migliorando l'efficienza computazionale.

### 3.2 Esempio di Configurazione
Un'architettura tipica per il problema dei cerchi potrebbe essere:
*   **Input:** 2 (coordinate x, y)
*   **Hidden Layers:** 16, 16
*   **Output:** 1 (probabilità della classe)
*   **Iperparametri:** Epoche (es. 5000), Batch size (es. 64), Learning rate (es. 0.01).

---

## ⏱️ Min 120-150

Ecco degli appunti universitari dettagliati basati sul contenuto del video, riguardanti l'implementazione di una Rete Neurale Artificiale (ANN) utilizzando la libreria **JAX**.

---

# Appunti di Machine Learning: Implementazione di una MLP con JAX

## 1. Convenzioni Dimensionali e Dati
Il docente introduce una variazione nella convenzione dei dati rispetto alle lezioni precedenti per favorire la flessibilità del programmatore:
*   **Convenzione Precedente:** Campioni organizzati per **colonne**.
*   **Convenzione Attuale:** Campioni organizzati per **righe** (standard in molti framework moderni).
    *   Dato un input $X$ e una matrice di pesi $W$: l'operazione diventa $X \cdot W + b$.
    *   Il numero di colonne di $X$ deve corrispondere al numero di righe di $W$.

## 2. Inizializzazione della Rete

### 2.1 Inizializzazione di un singolo strato (`init_layer_params`)
La funzione genera pesi casuali e bias inizializzati a zero.
*   **Pesi ($W$):** Utilizza `jax.random.normal`. La forma è `(in_dim, out_dim)`.
*   **Bias ($b$):** Utilizza `jnp.zeros`. La forma è `(out_dim,)`.
*   **Nota sul Broadcasting:** Con i campioni sulle righe, il broadcasting del bias avviene ora in modo coerente sulle righe dopo la moltiplicazione matriciale.

### 2.2 Inizializzazione Multi-strato (`init_mlp_params`)
Per inizializzare l'intera rete (Multi-Layer Perceptron):
1.  **Gestione Chiavi JAX:** JAX richiede una gestione esplicita dello stato casuale. Si usa `jax.random.split(key, num_layers)` per ottenere chiavi distinte per ogni strato.
2.  **Iterazione:** Si cicla sulle dimensioni fornite nella lista `layer_sizes`.
3.  **Struttura dati:** I parametri vengono salvati come una lista di tuple `[(W1, b1), (W2, b2), ...]`.

```python
def init_mlp_params(key, layer_sizes):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for i in range(len(layer_sizes) - 1):
        params.append(init_layer_params(keys[i], layer_sizes[i], layer_sizes[i+1]))
    return params
```

## 3. Funzioni di Attivazione e Forward Pass

### 3.1 Sigmoide
Viene definita per lo strato di output (classificazione binaria).
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
Il docente mostra che può essere implementata anche tramite la tangente iperbolica: `return (1 + jnp.tanh(x / 2)) / 2`.

### 3.2 Forward Pass (`forward`)
La funzione calcola l'output della rete:
*   **Strati Nascosti:** Utilizzano la funzione di attivazione `tanh` (tangente iperbolica).
*   **Strato Finale:** Utilizza la `sigmoide` per mappare l'output in un range $[0, 1]$, interpretandolo come probabilità.
*   **Logica:** Si usa un ciclo `for` per processare tutti gli strati tranne l'ultimo, che viene gestito separatamente per applicare l'attivazione corretta.

## 4. Funzione di Perdita e Ottimizzazione

### 4.1 Binary Cross-Entropy
Utilizzata per compiti di classificazione binaria:
$$\mathcal{L} = -\frac{1}{N} \sum [y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]$$

### 4.2 Aggiornamento dei Parametri con `tree_map`
Il docente introduce un concetto fondamentale di JAX: i **Pytrees**.
Invece di usare loop annidati complessi per aggiornare pesi e bias in strutture dati nidificate, si usa `jax.tree_util.tree_map`.

*   **Vantaggio:** Permette di applicare una funzione (es. l'aggiornamento del gradiente) a ogni foglia della struttura dati (lista di tuple), indipendentemente dalla complessità della rete.
*   **Codice di aggiornamento (Stochastic Gradient Descent - SGD):**
    ```python
    @jax.jit
    def update(params, x, y, lr):
        grads = jax.grad(binary_cross_entropy)(params, x, y)
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    ```
*   **`@jax.jit`:** Il decoratore Just-In-Time compilation accelera drasticamente l'esecuzione compilando la funzione di aggiornamento.

## 5. Ciclo di Addestramento (Training Loop)
Il processo segue questi step per ogni epoca:
1.  **Permutazione:** Shuffle casuale dei dati per l'addestramento stocastico (`jax.random.permutation`).
2.  **Mini-batch:** Suddivisione dei dati in blocchi (es. `batch_size = 64`).
3.  **Update:** Aggiornamento dei parametri per ogni mini-batch.
4.  **Monitoraggio:** Stampa della *loss* su un set di test ogni 100 epoche per verificare la convergenza.

## 6. Valutazione Finale
Al termine delle epoche (es. 5000), si valuta la bontà del modello:
*   **Accuratezza (Accuracy):** Calcolata confrontando le predizioni (con soglia a 0.5) con le etichette reali.
*   **Matrice di Confusione:** Importata da `sklearn.metrics` per analizzare i falsi positivi e falsi negativi.

---
**Termini Tecnici Chiave:**
*   **PRNG Key:** Stato del generatore di numeri casuali in JAX (stateless).
*   **Broadcasting:** Meccanismo con cui JAX/NumPy gestisce operazioni tra array di forme diverse.
*   **JIT (Just-In-Time):** Compilazione a runtime per ottimizzare le performance su CPU/GPU/TPU.
*   **Pytree:** Struttura ad albero di contenitori (liste, tuple, dict) che JAX può processare ricorsivamente.

---

## ⏱️ Min 150-180

Ecco degli appunti universitari dettagliati basati sul segmento video della lezione:

---

# Appunti del Laboratorio: Reti Neurali Artificiali (ANN)
**Argomento:** Valutazione delle Prestazioni e Visualizzazione dei Confini di Decisione (Decision Boundaries)

## 1. Valutazione delle Prestazioni del Modello
Dopo l'addestramento di una rete neurale, è fondamentale valutarne l'efficacia su un set di dati mai visto prima (test set).

### 1.1 Calcolo dell'Accuracy
L'accuratezza rappresenta la proporzione di predizioni corrette rispetto al totale delle osservazioni.
*   **Codice utilizzato:** 
    ```python
    y_pred = forward(params, X_test) > 0.5
    accuracy = jnp.mean(y_pred == y_test)
    ```
*   **Risultato:** In questo esempio, il modello ha ottenuto un'accuratezza di circa **0.9833** (98.3%).

### 1.2 Matrice di Confusione
La matrice di confusione fornisce una visione granulare degli errori di classificazione, mostrando come le classi reali si confrontano con quelle predette.
*   **Esempio dal video:**
    ```
    [[34, 0],
     [ 1, 25]]
    ```
*   **Interpretazione:**
    *   **Classe 1 (Reale):** 34 campioni sono stati correttamente classificati come Classe 1 (**Veri Positivi**). Zero errori per questa classe.
    *   **Classe 2 (Reale):** 25 campioni sono stati correttamente classificati come Classe 2 (**Veri Negativi**).
    *   **Errore:** C'è un solo elemento (**Falso Negativo**) situato sulla anti-diagonale. Un punto appartenente alla Classe 2 è stato erroneamente predetto come Classe 1.
    *   Il set di dati appare bilanciato e la rete neurale mostra un'ottima capacità di generalizzazione.

---

## 2. Visualizzazione dei Risultati
La visualizzazione spaziale aiuta a comprendere come la rete neurale "percepisce" e divide lo spazio delle caratteristiche.

### 2.1 Tecnica del Meshgrid
Per mappare il **Decision Boundary** (confine di decisione), viene creata una griglia cartesiana molto fitta (meshgrid) sopra lo spazio degli input.
1.  **Creazione della griglia:** Si definiscono i limiti minimi e massimi degli assi (x, y).
2.  **Predizione:** La rete neurale valuta ogni singolo punto della griglia.
3.  **Colorazione:** Viene assegnato un colore diverso a seconda della classe predetta dalla rete per quel punto specifico.

### 2.2 Analisi del Grafico
Il grafico finale mostra chiaramente la natura non lineare del problema:
*   **Punti (Scatter Plot):**
    *   **Cerchi:** Rappresentano i dati di addestramento (*training data*).
    *   **Croci (X):** Rappresentano i dati di test (*test data*).
*   **Aree Colorate:**
    *   **Area Rosa (Interna):** Corrisponde alla zona in cui la rete classifica i punti come "Classe 1". La forma è circolare/ellittica.
    *   **Area Blu Chiaro (Esterna):** Corrisponde alla zona classificata come "Classe 2".
*   **Decision Boundary:** È la linea di demarcazione tra il rosa e il blu. La rete ha imparato che tutto ciò che si trova all'interno della "patch" rosa appartiene alla prima categoria, mentre l'esterno appartiene alla seconda.

---

## 3. Note sul Codice e Implementazione
*   **Librerie:** Viene utilizzato `sklearn.metrics` per il calcolo delle metriche e `matplotlib.pyplot` (`plt`) per la parte grafica.
*   **Preprocessing:** Viene menzionata l'importanza dello **shuffling** (rimescolamento) dei dati e della gestione dei **mini-batch** durante la fase di aggiornamento dei parametri (`update`), sebbene vista solo brevemente nello scorrimento del codice.
*   **Log della Loss:** Durante l'addestramento, la *Binary Cross-Entropy Loss* viene monitorata ogni 100 epoche per assicurarsi che il modello stia convergendo correttamente.

---
*Fine della sessione di laboratorio.*

---

