# 📝 Appunti: Matteo Caldana's Personal Room-20251107 1514-1
**Modello:** gemini-3-flash-preview | **Data:** 23/01/2026

---

## ⏱️ Min 0-30

Ecco degli appunti universitari dettagliati basati sul segmento video della sessione di laboratorio sull'ottimizzazione.

---

# Laboratorio di Ottimizzazione Numerica: Gradient Descent e JAX

## 1. Introduzione e Obiettivi della Sessione
La sessione si focalizza sulla comprensione teorica e l'implementazione pratica di algoritmi di ottimizzazione del primo ordine. L'obiettivo è esplorare come diverse varianti del **Gradient Descent (GD)** si comportano su funzioni con diverse proprietà topologiche (convesse e non convesse).

### Algoritmi Trattati:
1.  **Vanilla Gradient Descent (GD):** Versione base con learning rate fisso.
2.  **Gradient Descent con Backtracking Line Search:** Algoritmo con learning rate adattivo basato sulla condizione di Armijo.
3.  **Exact Line Search per Funzioni Quadratiche:** Ottimizzazione analitica del passo per forme quadratiche.

---

## 2. Funzioni di Benchmark
Per testare la robustezza degli ottimizzatori, vengono utilizzate funzioni standard nella letteratura dell'ottimizzazione, caratterizzate da molti minimi locali che possono "intrappolare" gli algoritmi.

*   **Rastrigin Function:** Funzione non convessa e multi-modale. Utilizzata per testare la capacità di un algoritmo di sfuggire a minimi locali.
*   **Ackley Function:** Un'altra funzione non convessa ampiamente utilizzata per il benchmarking.
*   **Funzione Quadratica:** Definita come $f(x) = \frac{1}{2}x^T Ax + b^T x + c$. È una funzione convessa (se $A$ è definita positiva) e rappresenta il caso ideale per testare la convergenza teorica e l'esatta ricerca del passo.

---

## 3. Strumenti e Visualizzazione
L'implementazione utilizza **JAX**, una libreria per il calcolo numerico ad alte prestazioni che permette la differenziazione automatica (necessaria per calcolare i gradienti senza derivazione manuale).

### La tecnica del `meshgrid` per funzioni 2D
Per visualizzare le traiettorie di ottimizzazione su una superficie 2D, è necessario creare una griglia di punti:
1.  **Discretizzazione:** Si definiscono vettori lineari per gli assi X e Y usando `jnp.linspace(-5, 5, 50)`.
2.  **Griglia Cartesiana:** Si utilizza `jnp.meshgrid(x_vals, y_vals)` per generare due matrici (X, Y) che rappresentano tutte le coordinate della griglia.
3.  **Valutazione:** La funzione obiettivo viene valutata su ogni punto della griglia per ottenere la matrice Z.
4.  **Plotting:** Si usa `plt.contourf(X, Y, Z)` per creare un grafico a curve di livello (contour plot), dove il colore indica il valore della funzione (il blu solitamente indica i minimi, il giallo i massimi).

---

## 4. Algoritmo 1: Vanilla Gradient Descent
È un metodo iterativo definito dalla regola di aggiornamento:
$$x_{k+1} = x_k - \eta \nabla f(x_k)$$
dove $\eta$ (learning rate o LR) è una costante fissata.

### Dettagli di Implementazione:
*   **Input:** Gradiente della funzione (`grad_func`), punto iniziale (`x0`), learning rate (`lr`), tolleranza (`tol`), massimo numero di iterazioni (`max_iter`).
*   **Copia dei dati:** È fondamentale usare `x = x0.copy()` per evitare di modificare accidentalmente l'input originale (passaggio per riferimento).
*   **Loop di Ottimizzazione:**
    1.  Calcolo del valore del gradiente nel punto attuale: `grad_val = grad_func(x)`.
    2.  Aggiornamento della posizione: `x = x - lr * grad_val`.
    3.  Salvataggio del percorso (`path`) per la visualizzazione successiva.
*   **Criterio di Arresto:** Se la norma del gradiente ($\|\nabla f(x)\|$) è inferiore alla tolleranza (`tol`), l'algoritmo ha raggiunto un punto stazionario (minimo) e può terminare (`break`).

---

## 5. Algoritmo 2: GD con Backtracking Line Search
Il limite del GD classico è la scelta di $\eta$. Se troppo grande, l'algoritmo diverge; se troppo piccolo, la convergenza è lenta. Il **Backtracking** modifica dinamicamente $\eta$ ad ogni iterazione per garantire una "diminuzione sufficiente" della funzione.

### Condizione di Armijo:
L'algoritmo cerca un passo $t$ tale che:
$$f(x - t \nabla f(x)) \leq f(x) - \alpha t \|\nabla f(x)\|^2$$
dove $\alpha \in (0, 0.5)$ è un parametro di controllo.

### Logica del Backtracking:
1.  Si inizia con un passo ideale $t = 1$.
2.  **Ciclo While:** Finché la condizione di Armijo non è soddisfatta, si riduce il passo: $t = \beta t$ (dove $\beta \approx 0.8$).
3.  Una volta trovato il $t$ adatto, si procede con l'aggiornamento della posizione.

---

## 6. Algoritmo 3: Exact Line Search (Caso Quadratico)
Per le funzioni quadratiche del tipo $f(x) = \frac{1}{2}x^T Ax + b^T x + c$, è possibile calcolare analiticamente il passo ottimale $t^*$ che minimizza la funzione lungo la direzione del gradiente in un singolo step di ricerca locale.

### Formula del passo ottimale:
$$t^* = \frac{\nabla f(x)^T \nabla f(x)}{\nabla f(x)^T A \nabla f(x)}$$

In questo scenario, invece di usare un LR fisso o un backtracking iterativo, si calcola direttamente $t^*$ ad ogni iterazione, portando a una convergenza molto più efficiente verso il minimo della parabola $n$-dimensionale.

---

## Note Tecniche Aggiuntive
*   **Convenzione `_` (Underscore):** In Python, quando una variabile restituita da un iteratore (come l'indice in un `for _ in range(...)`) non viene utilizzata nel corpo del loop, si usa per convenzione il carattere underscore.
*   **Colorbar:** È utile aggiungere `plt.colorbar()` ai grafici di contour per mappare visivamente i colori ai valori numerici della funzione obiettivo (es. per verificare che il massimo sia effettivamente 1 in funzioni seno/coseno combinate).

---

## ⏱️ Min 30-60

Certamente! Ecco degli appunti universitari dettagliati basati sul segmento video fornito, strutturati per una comprensione accademica approfondita.

---

# Appunti di Ottimizzazione: Gradient Descent e Varianti

## 1. Introduzione all'Ottimizzazione Iterativa
L'obiettivo principale dell'ottimizzazione numerica è trovare il minimo di una funzione obiettivo $f(x)$. L'algoritmo più fondamentale per questo scopo è il **Gradient Descent (Discesa del Gradiente)**, definito dalla regola di aggiornamento:
$$x_{k+1} = x_k - \eta \nabla f(x_k)$$
dove $\eta$ rappresenta il **learning rate** (o dimensione del passo).

## 2. Gradient Descent con Backtracking Line Search
La scelta di un learning rate statico $\eta$ può portare a convergenza lenta (se troppo piccolo) o a divergenza/overshooting (se troppo grande). Il **Backtracking Line Search** modifica dinamicamente $\eta$ in ogni iterazione per garantire una "diminuzione sufficiente" della funzione obiettivo.

### Meccanismo del Backtracking
1.  **Inizializzazione:** Si parte con un passo generoso, tipicamente $\eta = 1.0$.
2.  **Condizione di Armijo:** Finché la condizione seguente è vera, il passo viene ridotto:
    $$f(x - \eta \nabla f(x)) > f(x) - \alpha \eta \|\nabla f(x)\|^2$$
    *   $\alpha \in (0, 0.5)$ è un parametro che controlla la diminuzione richiesta.
3.  **Aggiornamento del passo:** $\eta = \eta \cdot \beta$, dove $\beta \in (0, 1)$ (comunemente $\approx 0.8$).
4.  **Risultato:** Fornisce un compromesso tra costo computazionale e qualità della convergenza, evitando di rimanere bloccati prematuramente in minimi locali piatti o di "saltare" oltre il minimo.

## 3. Exact Line Search per Funzioni Quadratiche
Per le funzioni quadratiche della forma $f(x) = \frac{1}{2}x^T A x + b^T x + c$, è possibile calcolare analiticamente il passo ottimale $t^*$ in ogni iterazione.

### Derivazione Matematica
Definendo $g(t) = f(x^k + t s^k)$ dove $s^k = -\nabla f(x^k)$, il minimo si trova ponendo $g'(t^*) = 0$.
La formula risultante per il passo ottimale è:
$$t^* = \frac{\nabla f(x^k)^T \nabla f(x^k)}{\nabla f(x^k)^T A \nabla f(x^k)}$$
Questo metodo garantisce la massima discesa possibile lungo la direzione del gradiente in un singolo step.

## 4. Analisi Comparativa su Funzioni Benchmark
Nel video vengono testati gli algoritmi su diverse funzioni per osservarne il comportamento:

*   **Funzione di Rastrigin:** Funzione non convessa e multi-modale con molti minimi locali.
    *   *GD Vanilla:* Molto sensibile al learning rate; rischia di oscillare violentemente o bloccarsi.
    *   *GD con Backtracking:* Converge molto più velocemente verso il minimo globale (0,0) adattando il passo.
*   **Funzione di Ackley:** Un'altra funzione complessa con una superficie molto irregolare. Si osserva come il backtracking permetta di "navigare" meglio tra i minimi locali rispetto a un GD con passo fisso troppo grande (overshooting) o troppo piccolo (lentezza estrema).
*   **Funzione Quadratica:** L'Exact Line Search mostra la convergenza più rapida, raggiungendo precisioni di $10^{-6}$ in pochissime iterazioni.

## 5. Regressione Lineare con Stochastic Gradient Descent (SGD)
Il video introduce l'applicazione pratica dell'ottimizzazione nella regressione lineare.

### Setup del Problema
*   **Modello:** $y = \theta_0 + \theta_1 x$
*   **Parametri da apprendere:** $\theta = [\theta_0, \theta_1]$ (intercetta e pendenza).
*   **Funzione di Perdita (Loss Function):** Mean Squared Error (MSE).
    $$MSE(\theta) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

### Implementazione con JAX
Viene utilizzato il framework **JAX** per:
*   **Automatic Differentiation:** `jax.grad` calcola automaticamente il gradiente della funzione MSE rispetto ai parametri $\theta$.
*   **JIT Compilation:** `jax.jit` compila le funzioni per un'esecuzione più rapida.
*   **SGD Update:** Invece di usare l'intero dataset, l'SGD aggiorna i parametri usando piccoli "mini-batch" di dati, permettendo una convergenza più rapida e scalabilità su grandi set di dati.

---

### Terminologia Tecnica Chiave
*   **Convevessità:** Proprietà di una funzione dove ogni minimo locale è anche un minimo globale.
*   **Overshooting:** Fenomeno in cui il passo di apprendimento è così grande da superare il minimo, causando divergenza.
*   **Mini-batch:** Sottoinsieme casuale del dataset utilizzato per un singolo aggiornamento dei parametri nell'SGD.
*   **Epsilon di macchina ($\epsilon$):** La più piccola differenza distinguibile tra due numeri rappresentati nel sistema; nel video citato come $\approx 10^{-8}$ per la singola precisione.

---

## ⏱️ Min 60-90

Certamente. Ecco degli appunti universitari dettagliati basati sul segmento video riguardante l'implementazione dello **Stochastic Gradient Descent (SGD) con Mini-batch**.

---

# Appunti di Machine Learning: Stochastic Gradient Descent (SGD) con Mini-batch

## 1. Introduzione e Motivazione Teorica
Il Gradient Descent standard (Batch Gradient Descent) calcola il gradiente della funzione di perdita considerando l'intero dataset ad ogni iterazione. Sebbene questo garantisca la direzione esatta di massima decrescita, presenta due limiti principali:
1.  **Costo Computazionale:** Con dataset di grandi dimensioni ($N$ molto elevato), calcolare il gradiente su tutti i campioni è estremamente oneroso in termini di memoria e tempo di calcolo.
2.  **Efficienza:** Spesso è possibile ottenere una buona approssimazione della direzione ottimale analizzando solo una frazione dei dati, permettendo di effettuare più aggiornamenti dei parametri nello stesso intervallo di tempo.

Lo **Stochastic Gradient Descent (SGD)** con **Mini-batch** rappresenta il compromesso ideale: si calcola il gradiente su un piccolo sottoinsieme di dati (batch) scelto casualmente.

### Il Trade-off
*   **Gradiente Approssimato:** La direzione non è quella "vera" calcolata sull'intero dataset, ma una stima rumorosa.
*   **Velocità:** Si effettuano molti più aggiornamenti dei parametri per ogni "epoca" (passaggio completo sui dati), portando spesso a una convergenza più rapida.

---

## 2. Componenti dell'Implementazione (Libreria JAX)
L'implementazione mostrata utilizza **JAX**, una libreria per il calcolo numerico ad alte prestazioni che supporta la differenziazione automatica e la compilazione JIT (Just-In-Time).

### 2.1 Il Modello (Regressione Lineare)
Il modello è definito come una funzione lineare semplice:
$$y = \theta_0 + \theta_1 \cdot x$$
In Python:
```python
@jax.jit
def model(theta, x):
    return theta[0] + theta[1] * x
```
*Nota: `theta` è un array contenente l'intercetta e la pendenza.*

### 2.2 Funzione di Perdita: Mean Squared Error (MSE)
Si utilizza l'errore quadratico medio per misurare la discrepanza tra le predizioni ($\hat{y}$) e i valori reali ($y$):
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$
```python
@jax.jit
def mse_loss(theta, x, y):
    y_pred = model(theta, x)
    return jnp.mean((y_pred - y)**2)
```

### 2.3 Differenziazione Automatica
Invece di derivare manualmente il gradiente, JAX permette di calcolarlo automaticamente:
```python
grad_mse = jax.jit(jax.grad(mse_loss))
```
La funzione `jax.grad` restituisce una nuova funzione che computa il gradiente della loss rispetto al primo argomento (`theta`).

---

## 3. L'Algoritmo SGD con Mini-batch

### 3.1 Lo Step di Aggiornamento (`sgd_update`)
Per ogni mini-batch, i parametri vengono aggiornati sottraendo il gradiente pesato dal **learning rate** ($\alpha$):
$$\theta_{new} = \theta_{old} - \alpha \cdot \nabla MSE(\theta, X_{batch}, Y_{batch})$$
```python
@jax.jit
def sgd_update(theta, x_batch, y_batch, learning_rate):
    grads = grad_mse(theta, x_batch, y_batch)
    theta = theta - learning_rate * grads
    return theta
```

### 3.2 Il Ciclo di Ottimizzazione Principale
La funzione `stochastic_gradient_descent` coordina l'intero processo:

1.  **Epoche:** Si itera per un numero prestabilito di epoche (es. 100).
2.  **Rimescolamento (Permutation):** All'inizio di ogni epoca, i dati vengono mescolati casualmente. Questo è fondamentale per garantire che i batch siano diversi ad ogni ciclo e per evitare bias nell'apprendimento.
3.  **Iterazione sui Mini-batch:** Il dataset viene suddiviso in blocchi di dimensione `batch_size` (es. 10).
4.  **Aggiornamento:** Si chiama `sgd_update` per ogni blocco.

```python
def stochastic_gradient_descent(theta, training_input, training_labels, ...):
    for epoch in range(epochs):
        # 1. Generazione permutazione casuale
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(training_input))
        
        # 2. Loop sui mini-batch
        for i in range(0, len(training_input), batch_size):
            batch_idx = perm[i : i + batch_size]
            x_batch = training_input[batch_idx]
            y_batch = training_labels[batch_idx]
            
            # 3. Aggiornamento parametri
            theta = sgd_update(theta, x_batch, y_batch, learning_rate)
    return theta
```

---

## 4. Analisi dei Risultati
Nel video viene generato un dataset sintetico basato sulla retta $y = 1.5x + 3$ con rumore gaussiano.
*   **Parametri Iniziali:** $\theta = [0.0, 0.0]$
*   **Parametri Ottimizzati:** Dopo 100 epoche con batch size 10, l'algoritmo restituisce valori molto vicini ai target (es. $\theta_0 \approx 3.15$, $\theta_1 \approx 1.46$).

### Parametri di Controllo (Hyperparameters)
*   **Batch Size:** Se piccolo, il gradiente è più rumoroso ma il calcolo è velocissimo. Se grande, la stima è più precisa ma più lenta.
*   **Learning Rate:** Definisce la grandezza del "passo" verso il minimo. Se troppo alto, l'algoritmo può divergere; se troppo basso, la convergenza è eccessivamente lenta.

---

## Glossario Tecnico
*   **Epoch (Epoca):** Un passaggio completo su tutto il dataset di addestramento.
*   **Mini-batch:** Un sottoinsieme del dataset utilizzato per un singolo calcolo del gradiente.
*   **Learning Rate ($\alpha$):** Scalare che determina l'entità dell'aggiornamento dei pesi.
*   **JIT (Just-In-Time):** Compilazione del codice durante l'esecuzione per ottimizzarne la velocità su CPU/GPU/TPU.
*   **Permutation:** Riordinamento casuale degli indici dei dati per garantire l'imparzialità dei mini-batch.

---

## ⏱️ Min 90-120

Ecco degli appunti universitari dettagliati basati sul segmento video relativo alla lezione di Analisi Numerica per il Machine Learning, focalizzata sulle **Support Vector Machines (SVM)** per regressione e classificazione, implementate con la libreria **JAX**.

---

# Appunti di Analisi Numerica per il Machine Learning

## 1. Visualizzazione e Valutazione della Regressione Lineare
Prima di passare a modelli più complessi, è fondamentale saper visualizzare i risultati di un modello di regressione lineare semplice.

*   **Scatter Plot**: Utilizzo di `plt.scatter` per distinguere visivamente il *training set* (punti blu) dal *test set* (punti arancioni).
*   **Retta di Regressione**: Viene generata una sequenza di punti con `jnp.linspace` e calcolato il valore predetto $y$ utilizzando i parametri ottimizzati ($\theta_{opt}$).
*   **Metrica di Errore (MSE)**: La bontà del modello viene valutata calcolando il **Mean Squared Error (MSE)** esclusivamente sul **test set**. Questo garantisce che la metrica rifletta la capacità di generalizzazione del modello su dati non visti.
    *   Esempio nel video: $MSE \approx 0.75$.

## 2. Support Vector Regression (SVR)
La SVR è una variante delle SVM applicata alla regressione. A differenza della regressione lineare classica che minimizza la somma dei quadrati degli scarti, la SVR cerca di adattare l'errore all'interno di una soglia specifica.

### 2.1 La Funzione di Perdita (Loss Function)
La funzione di perdita per la SVR è definita come **$\epsilon$-insensitive loss** (o perdita tubolare), spesso combinata con una regolarizzazione $L_2$:

$$L(w) = \lambda \|w\|^2 + \frac{1}{m} \sum_{i=1}^{n} \max(0, |f(x_i) - y_i| - \epsilon)$$

**Componenti della formula:**
*   $\|w\|^2$: Termine di **regolarizzazione** (norma quadra dei pesi). Serve a controllare la complessità del modello e prevenire l'overfitting.
*   $\lambda$ (lambda): Parametro di regolarizzazione che determina il peso del termine di penalità.
*   $\epsilon$ (epsilon): Definisce una "zona morta" (tubo) attorno alla funzione predetta. Se l'errore di predizione è inferiore a $\epsilon$, la perdita è nulla.
*   $f(x_i) - y_i$: Residuo tra valore predetto e valore reale.

### 2.2 Implementazione della Classe SVR
Il modello viene incapsulato in una classe Python per una gestione più ordinata.

1.  **`__init__(self, epsilon, lambda)`**: Inizializza i parametri di tolleranza ($\epsilon$) e regolarizzazione ($\lambda$).
2.  **`loss(self, params, X, y)`**:
    *   Calcola le predizioni: $pred = X \cdot w + b$ (dove $b$ è il bias o intercetta).
    *   Implementa la parte $\epsilon$-insensitive usando `jnp.maximum(0, jnp.abs(residuo) - epsilon)`.
    *   Aggiunge il termine di regolarizzazione $\lambda \cdot \text{sum}(params^2)$.
3.  **`train(self, X, y)`**:
    *   Inizializza i pesi ($w$) a zero. La dimensione sarà `n_features + 1` (per includere il bias).
    *   Definisce la funzione gradiente tramite `jax.grad`.
    *   Utilizza una funzione di `step` ottimizzata con `@jax.jit` per eseguire la discesa del gradiente stocastica (SGD).
4.  **`predict(self, X)`**: Applica i pesi appresi ai nuovi dati $X$ per ottenere le stime di $y$.

### 2.3 Visualizzazione del Risultato
Il grafico risultante mostra la retta di regressione circondata da una fascia colorata (il "tubo" di raggio $\epsilon$). I punti che cadono all'interno di questa fascia non contribuiscono alla perdita, rendendo il modello robusto rispetto a piccoli rumori nei dati.

---

## 3. Support Vector Classification (SVC)
Il passaggio successivo riguarda la classificazione binaria, dove l'obiettivo non è più prevedere un valore continuo, ma assegnare un punto a una categoria (es. Classe 0 o Classe 1).

### 3.1 Hinge Loss
Per la classificazione, si utilizza la **Hinge Loss**:

$$L(w) = \lambda \|w\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w^T x_i + b))$$

*   Qui $y_i$ assume solitamente valori in $\{-1, 1\}$.
*   La funzione cerca di massimizzare il margine tra le due classi.

### 3.2 Differenze strutturali rispetto alla regressione
*   **Feature Space**: Mentre l'esempio di regressione era unidimensionale (1D), la classificazione viene presentata in uno spazio bidimensionale (2D) con feature $x_1$ e $x_2$.
*   **Decision Boundary**: Il modello cerca di trovare un **iperpiano di separazione** (in 2D è una retta) che divida i cluster di punti.
*   **Metrica di Valutazione**: Invece dell'MSE, si utilizza l'**Accuracy** (rapporto tra predizioni corrette e totale delle osservazioni). Un modello SVC ben addestrato dovrebbe superare il 90% di accuratezza sui dati sintetici proposti.

### 3.3 Gestione delle Dimensioni in JAX
Un aspetto tecnico cruciale evidenziato è l'uso di `reshape(-1, 1)`. JAX richiede precisione millimetrica nelle operazioni tra matrici (dot product). Anche se un set di dati sembra un vettore (1D), deve essere trattato come una matrice con una colonna per soddisfare i requisiti delle dimensioni contraenti negli iperparametri del modello.

---

**Glossario Tecnico:**
*   **JAX**: Libreria per il calcolo numerico che permette la differenziazione automatica e la compilazione JIT (Just-In-Time) su GPU/TPU.
*   **$\epsilon$-Insensitive Tube**: Regione in cui l'errore di regressione è ignorato.
*   **Soft Margin**: Approccio che permette alcune violazioni del margine o del tubo per ottenere una migliore generalizzazione.
*   **Bias Term**: L'intercetta di un modello lineare, spesso aggiunta come colonna di 1 nella matrice delle feature.

---

## ⏱️ Min 120-150

Ecco degli appunti universitari dettagliati basati sulla sessione di laboratorio presentata nel video.

---

# Appunti del Laboratorio: Ottimizzazione con JAX e Implementazione di SVM

**Argomento:** Implementazione da zero di una Support Vector Machine (SVM) lineare in formulazione primale utilizzando la libreria JAX per il calcolo dei gradienti e l'ottimizzazione.
**Tecnologie utilizzate:** Python, JAX (`jax.numpy`), Matplotlib.

---

## 1. Fondamenti Matematici: La Funzione di Perdita (Loss Function)

L'obiettivo è implementare una SVM lineare. La funzione di costo (loss) per una SVM è composta da due parti principali: la **Hinge Loss** (perdita a cerniera) e un termine di **regolarizzazione L2**.

### Formulazione della Loss:
$$L(w, b) = \lambda \|w\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w^T x_i + b))$$

Dove:
- $w$: Vettore dei pesi (weights).
- $b$: Termine di bias.
- $\lambda$: Parametro di regolarizzazione.
- $y_i$: Etichetta reale della classe ($\{-1, 1\}$).
- $x_i$: Vettore delle feature per il campione $i$.
- $w^T x_i + b$: Funzione di decisione.

---

## 2. Implementazione della Classe `SVM`

La classe `SVM` incapsula la logica del modello, inclusi il calcolo della loss, l'addestramento tramite discesa del gradiente e le predizioni.

### Metodo `__init__`
Inizializza il parametro di regolarizzazione `lambda`. Inizialmente, i pesi `self.w` sono impostati a `None`.

### Metodo `loss`
Questo metodo calcola il valore della funzione obiettivo.
- **Funzione di decisione:** Viene calcolata come prodotto matriciale tra i dati `X` e i pesi (escluso l'ultimo elemento) più il bias (l'ultimo elemento dei parametri). 
- **Reshape tecnico:** Viene utilizzato `params[:-1].reshape(-1, 1)` per assicurare che il prodotto matriciale con `X` (matrice di campioni) avvenga correttamente riga per riga.
- **Hinge Loss:** Implementata tramite `jnp.maximum(0, 1 - y * decision)`.
- **Regolarizzazione:** Implementata come `jnp.sum(params**2) * self.lambda`.

### Metodo `train`
Gestisce il processo di ottimizzazione.
1. **Inizializzazione:** I parametri (pesi + bias) vengono inizializzati a zero utilizzando `jnp.zeros`.
2. **Calcolo del Gradiente:** Utilizza `jax.grad(self.loss)` per ottenere automaticamente la funzione che calcola il gradiente rispetto ai parametri.
3. **Loop di Ottimizzazione:** Esegue un numero fissato di iterazioni (`max_iter`).
4. **JAX JIT Compilation:** Il passo di aggiornamento (`step`) è decorato con `@jax.jit` per compilare il codice in codice macchina altamente efficiente.

> **Nota tecnica importante su `@jax.jit` e le "Impure Functions":** 
> L'istruttore sottolinea un punto critico: all'interno di una funzione JIT, è fondamentale passare i parametri da aggiornare (come `w`) come argomenti espliciti della funzione invece di accedervi tramite `self.w`. Questo perché JAX richiede funzioni "pure". Se si usasse `self.w` all'interno del JIT, JAX potrebbe memorizzare nella cache il valore iniziale e non aggiornarlo correttamente durante le iterazioni.

### Metodo `predict`
Restituisce la classe predetta per nuovi dati.
- Calcola il valore della funzione di decisione.
- Utilizza `jnp.sign(decision)` per mappare il risultato a `-1` o `1`.

---

## 3. Workflow Sperimentale

### Generazione dei Dati Sintetici
Vengono generati 100 campioni in uno spazio 2D con valori tra 0 e 10. Le etichette vengono assegnate in base alla condizione $z_1 + z_2 > 10$. Le classi sono rappresentate dai valori `1` e `-1`.

### Split del Dataset
Il dataset viene diviso in set di addestramento (**training set**) e set di test (**test set**) con una proporzione 80/20 utilizzando `train_test_split`.

### Addestramento e Tuning
Il modello viene istanziato con un valore di $\lambda$ (es. 0.001) e addestrato chiamando `svm.train(X_train, y_train, lr=1e-1, max_iter=5000)`. 
*Osservazione dal video:* L'istruttore mostra che se l'accuratezza iniziale è bassa, potrebbe essere necessario correggere eventuali bug nel calcolo della loss (es. confusione tra `jnp.mean` e `jnp.sum`) o regolare gli iperparametri per far convergere il modello correttamente.

---

## 4. Valutazione e Visualizzazione

### Accuratezza
L'accuratezza viene calcolata confrontando le predizioni del modello con le etichette reali sia sul set di training che su quello di test.
$$Accuracy = \frac{\text{Numero di Predizioni Corrette}}{\text{Totale Campioni}}$$

### Visualizzazione dei Risultati
Viene utilizzato Matplotlib per plottare:
- I punti di addestramento e di test (colorati per classe).
- Il **confine di decisione** (decision boundary): la linea dove la funzione di decisione è uguale a zero.
- Le regioni di appartenenza delle classi (spesso visualizzate tramite `plt.contourf`).

---

## Termini Tecnici Chiave
- **Primal Formulation:** Risoluzione del problema SVM ottimizzando direttamente i pesi e il bias.
- **Hinge Loss:** Funzione di perdita specifica per SVM che penalizza non solo le classificazioni errate, ma anche quelle corrette che si trovano troppo vicine al margine.
- **L2 Regularization:** Tecnica per prevenire l'overfitting penalizzando i pesi con valori elevati.
- **Automatic Differentiation (JAX):** Capacità della libreria di calcolare derivate di funzioni Python/NumPy complesse in modo automatico.
- **JIT (Just-In-Time) Compilation:** Trasformazione dinamica del codice Python in codice eseguibile ottimizzato per CPU/GPU/TPU.

---

## ⏱️ Min 150-180

Certamente! Ecco degli appunti universitari dettagliati basati sul segmento video analizzato, incentrati sull'implementazione di un **Linear Support Vector Classifier (Linear-SVC)** utilizzando la libreria **JAX**.

---

# Appunti di Machine Learning: Implementazione di Linear-SVC con JAX

## 1. Introduzione all'SVM (Support Vector Machine)
L'obiettivo della sessione è l'implementazione da zero di un classificatore SVM lineare. JAX viene utilizzato per sfruttare la differenziazione automatica (`jax.grad`) e la compilazione Just-In-Time (`jax.jit`) per ottimizzare le prestazioni.

### Concetti Chiave:
*   **Funzione di Decisione:** $f(x) = Xw + b$, dove $w$ sono i pesi e $b$ è il bias.
*   **Hinge Loss (Funzione di Perdita a Cerniera):** Utilizzata per massimizzare il margine tra le classi.
*   **Regolarizzazione L2:** Aggiunta alla funzione di perdita per prevenire l'overfitting, controllata dal parametro $\lambda$ (lambda).

---

## 2. Struttura della Classe `SVM`

### Inizializzazione (`__init__`)
```python
def __init__(self, lambda_param=1.0):
    self.lambda_param = lambda_param
    self.w = None
```
Il parametro `lambda_param` determina la forza della regolarizzazione.

### Funzione di Perdita (`loss`)
La funzione di perdita è composta da due parti:
1.  **Termine di errore (Hinge Loss):** $\max(0, 1 - y \cdot \text{decision})$
2.  **Termine di regolarizzazione:** $\lambda \cdot \|w\|^2$

**Dettaglio Tecnico (Debugging):**
Durante il video, viene discusso un errore comune riguardante lo *shaping* dei parametri. Se il bias $b$ è incluso nell'ultimo elemento del vettore dei parametri (`params[-1]`), la moltiplicazione matriciale deve escluderlo:
`decision = X @ params[:-1] + params[-1]`

### Addestramento (`train`)
L'addestramento avviene tramite **Discesa del Gradiente (SGD)**.
*   **JAX Autodiff:** `grad_fn = jax.grad(self.loss, argnums=0)` calcola automaticamente il gradiente della perdita rispetto ai pesi.
*   **JIT Compilation:** L'uso del decoratore `@jax.jit` sulla funzione `step` velocizza drasticamente l'esecuzione compilando il codice per la specifica architettura (CPU/GPU).

---

## 3. Workflow Sperimentale

### Generazione di Dati Sintetici
Viene creato un dataset 2D per la classificazione binaria:
*   **Feature ($X$):** Punti campionati uniformemente in uno spazio $[0, 10]^2$.
*   **Target ($y$):** Etichette $(-1, 1)$ basate sulla condizione $z_1 + z_2 > 10$.

### Suddivisione del Dataset (Split)
I dati vengono divisi in **Training Set** e **Test Set** (tipicamente 80/20) utilizzando `train_test_split` di scikit-learn, per valutare la capacità di generalizzazione del modello.

---

## 4. Valutazione e Visualizzazione

### Accuratezza
L'accuratezza viene calcolata come la proporzione di previsioni corrette rispetto al totale:
$$\text{Accuracy} = \frac{\text{Numero di predizioni corrette}}{\text{Totale delle predizioni}}$$
Nel video, il modello raggiunge un'ottima accuratezza (circa **95%**).

### Visualizzazione del Confine di Decisione (Decision Boundary)
Per visualizzare come il modello separa le classi, si utilizza un **Meshgrid**.

**Tecnica di Plotting:**
*   `plt.contour`: Disegna solo le linee di livello (il confine tra le classi).
*   `plt.contourf` (f = *filled*): Riempie le aree con colori diversi per rappresentare le zone di decisione del modello.

**Interpretazione del Grafico:**
*   **Dots (Punti):** Rappresentano i dati di addestramento.
*   **Crosses (Croci):** Rappresentano i dati di test.
*   **Sfondo Colorato:** Mostra la regione in cui il modello predice $+1$ o $-1$. Il confine lineare riflette la natura "Linear-SVC" dell'implementazione.

---

## 5. Note del Docente / Osservazioni Finali
*   **Handling del Bias:** Prestare attenzione alla distinzione tra i pesi delle feature e il termine di bias durante il calcolo del prodotto scalare.
*   **Vantaggi di JAX:** La facilità nel calcolare gradienti complessi e la velocità del JIT rendono JAX ideale per prototipare algoritmi di ottimizzazione per ML.
*   **Errore di Indentazione:** All'inizio del video viene mostrato un `IndentationError`, a dimostrazione dell'importanza della formattazione rigorosa in Python.

---

