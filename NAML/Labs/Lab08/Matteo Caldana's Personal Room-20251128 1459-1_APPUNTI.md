# 📝 Appunti: Matteo Caldana's Personal Room-20251128 1459-1
**Modello:** gemini-3-flash-preview | **Data:** 24/01/2026

---

## ⏱️ Min 0-30

Ecco degli appunti universitari dettagliati basati sul segmento video fornito, riguardante i metodi di ottimizzazione del primo ordine per le reti neurali.

---

# Metodi di Ottimizzazione del Primo Ordine per il Training di Reti Neurali

## 1. Panoramica del Corso e Obiettivi del Laboratorio
Il laboratorio si concentra sull'approfondimento delle tecniche di ottimizzazione per le reti neurali artificiali (ANN). L'obiettivo è passare da implementazioni di base a metodi più sofisticati utilizzati nelle librerie moderne.

### Argomenti Trattati:
1.  **Ottimizzatori del primo ordine:** Implementazione di varianti avanzate come il **Momentum** e **RMSprop**.
2.  **Ottimizzatori del secondo ordine:** Metodi basati sull'algoritmo di **Newton**. Questi sono particolarmente rilevanti in matematica applicata e per le **PINNs** (Physics-Informed Neural Networks) utilizzate per risolvere equazioni differenziali alle derivate parziali (PDE).
3.  **Regolarizzazione:** Studio dei termini aggiuntivi nella funzione di costo che penalizzano parametri di grandi dimensioni (es. Regressione Lasso, Support Vector Machines) per migliorare la perdita di validazione.

---

## 2. Definizione del Problema: Approssimazione di Funzione
L'obiettivo pratico è utilizzare una rete neurale per approssimare una funzione unidimensionale definita sull'intervallo $[0, 10]$:

$$f(x) = e^{-x/3} \sin(x) + \frac{1}{10} \cos(\pi x)$$

### Generazione dei Dati
Viene generato un set di dati di training aggiungendo rumore gaussiano alla funzione originale per simulare dati reali:
*   **Punti di training:** 100 punti campionati.
*   **Rumore:** Distribuzione normale con media zero e deviazione standard $\sigma = 0.05$.

---

## 3. Architettura della Rete Neurale (ANN)
L'implementazione utilizza la libreria **JAX** per il calcolo dei gradienti e la compilazione JIT (Just-In-Time).

*   **Struttura:** 2 layer nascosti con 5 neuroni ciascuno (Configurazione: `[1, 5, 5, 1]`).
*   **Funzione di Attivazione:** Tangente iperbolica (`tanh`) applicata a tutti i layer tranne l'ultimo.
*   **Funzione di Costo (Loss):** Errore Quadratico Medio (Mean Squared Error - MSE).
    $$L(x, y, \theta) = \frac{1}{m} \sum_{i=1}^n (y_i - \text{ANN}(x_i, \theta))^2$$
    *Dove $m$ è il numero di campioni e $\theta$ sono i parametri della rete.*

---

## 4. Metodi di Ottimizzazione Implementati

### A. Gradient Descent (GD) Standard (Full Batch)
Aggiornamento dei parametri basato sull'intero set di dati ad ogni iterazione (epoca).
*   **Direzione:** Massima pendenza negativa del gradiente.
*   **Osservazione:** Il grafico della loss può mostrare incrementi temporanei a causa della natura **non convessa** della funzione di costo, ma tende a diminuire nel lungo periodo.

### B. Stochastic Gradient Descent (SGD) con Linear Decay
Introduce l'uso di mini-batch e una strategia di decadimento della velocità di apprendimento (learning rate).
*   **Decadimento Lineare:** Il learning rate $\lambda$ diminuisce progressivamente per evitare "overshooting" (superamento del minimo) quando ci si avvicina alla soluzione ottimale.
    $$\lambda_k = \max \left( \lambda_{min}, \lambda_{max} \left( 1 - \frac{k}{K} \right) \right)$$
    *Dove $k$ è l'epoca corrente e $K$ è il parametro di decadimento.*

### C. SGD con Momentum
Simula una "palla che rotola" lungo una valle.
*   **Velocità ($v$):** Mantiene una memoria dei gradienti passati per accelerare nelle direzioni costanti e ridurre le oscillazioni.

### D. AdaGrad e RMSprop
Metodi di apprendimento adattivo che regolano il learning rate per ogni singolo parametro.
*   **RMSprop:** Utilizza una media mobile esponenziale dei quadrati dei gradienti per "dimenticare" i gradienti molto vecchi (parametro di decadimento $\rho$).

---

## 5. Note Tecniche di Implementazione (JAX)
*   **`jax.jit`:** Utilizzato per velocizzare la valutazione della loss e del gradiente tramite compilazione.
*   **`jax.grad`:** Utilizzato per la differenziazione automatica rispetto ai parametri.
*   **Visualizzazione:** Viene utilizzata una classe `Callback` personalizzata per aggiornare dinamicamente i grafici (Loss vs Epoche e Fit della ANN vs Dati Reali) durante il processo di training.

---

## 6. Risultati Sperimentali e Analisi
Durante la dimostrazione video del **Full Batch Gradient Descent**:
*   **Oscillazioni:** Si osservano piccole oscillazioni nella loss, tipiche dei metodi del primo ordine.
*   **Non-convessità:** La perdita può aumentare temporaneamente se l'algoritmo finisce in una "valle" diversa o attraversa regioni ad alta curvatura.
*   **Convergenza:** Dopo circa 2000 epoche, la loss finale si stabilizza intorno a $5 \times 10^{-3}$, mostrando un buon adattamento della curva arancione (ANN) ai punti verdi (dati rumorosi).

---

## ⏱️ Min 30-60

Certo! Ecco degli appunti universitari dettagliati basati sul segmento video fornito, che copre vari algoritmi di ottimizzazione per l'addestramento di reti neurali.

---

# Appunti di Ottimizzazione per il Deep Learning

Questo modulo esplora l'implementazione pratica e i concetti teorici alla base dello **Stochastic Gradient Descent (SGD)** e delle sue varianti avanzate: **Momentum**, **AdaGrad** e **RMSProp**.

## 1. Stochastic Gradient Descent (SGD) con Mini-Batch

Il video inizia con una discussione sull'uso dei mini-batch nell'SGD. 

### Concetti Chiave:
*   **Epoca (Epoch):** Tradizionalmente, un'epoca implica il passaggio attraverso l'intero set di dati. Tuttavia, nell'implementazione mostrata, il termine "epoca" è usato in modo più flessibile per indicare un singolo aggiornamento basato su un mini-batch estratto casualmente.
*   **Vantaggio Computazionale:** L'uso dei mini-batch riduce drasticamente il costo computazionale per ogni aggiornamento (epoca nel codice), permettendo più iterazioni con la stessa quantità totale di dati elaborati rispetto al Full Batch Gradient Descent.
*   **Confronto delle Perdite (Loss):** È più facile confrontare l'andamento della funzione di perdita tra i casi "full batch" e "mini-batch" se si mantiene lo stesso numero totale di aggiornamenti dei parametri.

### Implementazione Python (snippet):
```python
# Estrazione casuale degli indici per il mini-batch
idxs = np.random.choice(n_training_points, batch_size)
x_batch = xx[idxs, :]
y_batch = yy[idxs, :]

# Calcolo dei gradienti sul mini-batch
grads = grad_jit(x_batch, y_batch, params)

# Aggiornamento dei parametri
for i in range(len(params)):
    params[i] = params[i] - learning_rate * grads[i]
```

---

## 2. SGD con Momentum

L'aggiunta del momentum aiuta l'ottimizzatore a superare i minimi locali e a ridurre le oscillazioni nella direzione del gradiente.

### Formula Matematica:
Dati i parametri $\theta$ e la velocità $v$:
1.  $\mathbf{v}^{(k+1)} = \alpha \mathbf{v}^{(k)} - \lambda \mathbf{g}^{(k)}$
2.  $\theta^{(k+1)} = \theta^{(k)} + \mathbf{v}^{(k+1)}$

Dove:
*   $\alpha$ è il coefficiente di momentum (es. 0.9), che rappresenta la "memoria" delle direzioni passate.
*   $\lambda$ è il learning rate.
*   $\mathbf{g}^{(k)}$ è il gradiente calcolato al passo $k$.

### Intuizione:
L'aggiornamento non dipende solo dal gradiente corrente, ma anche da una media pesata dei gradienti passati. Questo crea un effetto inerziale che accelera la discesa lungo direzioni consistenti e smorza le oscillazioni.

### Risultati Sperimentali:
Nel video, l'SGD con momentum raggiunge una perdita finale di circa $2.55 \times 10^{-3}$, significativamente inferiore all'SGD semplice ($4.76 \times 10^{-3}$) a parità di sforzo computazionale.

---

## 3. AdaGrad (Adaptive Gradient Algorithm)

AdaGrad introduce un learning rate adattivo per ogni singolo parametro.

### Formula Matematica:
1.  $\mathbf{r}^{(k+1)} = \mathbf{r}^{(k)} + \mathbf{g}^{(k)} \odot \mathbf{g}^{(k)}$ (accumulo del quadrato dei gradienti)
2.  $\theta^{(k+1)} = \theta^{(k)} - \frac{\lambda}{\delta + \sqrt{\mathbf{r}^{(k+1)}}} \odot \mathbf{g}^{(k)}$

Dove:
*   $\odot$ indica il prodotto componente per componente (element-wise).
*   $\delta$ è una costante piccola (es. $10^{-7}$) per evitare la divisione per zero.
*   $\mathbf{r}$ agisce come un fattore di scala che riduce il learning rate per i parametri che hanno ricevuto gradienti elevati in passato.

### Limitazioni:
Poiché $\mathbf{r}$ cresce monotonicamente, il learning rate adattivo può diventare eccessivamente piccolo, interrompendo prematuramente l'esplorazione del modello prima di raggiungere un ottimo globale.

---

## 4. RMSProp (Root Mean Square Propagation)

RMSProp è un'evoluzione di AdaGrad progettata per risolvere il problema della riduzione eccessiva del learning rate.

### Formula Matematica:
1.  $\mathbf{r}^{(k+1)} = \rho \mathbf{r}^{(k)} + (1 - \rho) \mathbf{g}^{(k)} \odot \mathbf{g}^{(k)}$
2.  $\theta^{(k+1)} = \theta^{(k)} - \frac{\lambda}{\delta + \sqrt{\mathbf{r}^{(k+1)}}} \odot \mathbf{g}^{(k)}$

### Differenza con AdaGrad:
Invece di accumulare semplicemente il quadrato dei gradienti, RMSProp utilizza una **media mobile esponenziale**. Il parametro di decadimento $\rho$ (decay rate, es. 0.9) permette all'algoritmo di "dimenticare" la storia molto vecchia dei gradienti, mantenendo il learning rate adattivo più flessibile durante tutto l'addestramento.

### Implementazione Python (RMSProp):
```python
# Aggiornamento della media mobile del quadrato dei gradienti
cumulated_square_grad[i] = decay_rate * cumulated_square_grad[i] + (1 - decay_rate) * grads[i]**2

# Calcolo del learning rate adattato
learning_rate_adapted = learning_rate / (delta + jnp.sqrt(cumulated_square_grad[i]))

# Aggiornamento parametri
params[i] = params[i] - learning_rate_adapted * grads[i]
```

---

## Osservazioni sulla Regolarizzazione

*   **Oscillazioni:** Gli algoritmi basati su mini-batch introducono intrinsecamente del rumore durante l'addestramento, visibile come oscillazioni nella curva della funzione di perdita.
*   **Early Stopping:** A causa di queste oscillazioni, è cruciale monitorare la perdita sul set di validazione e utilizzare la tecnica dell'early stopping per salvare il modello quando raggiunge il minimo sulla validazione, evitando di fermarsi in un punto sfavorevole a causa di una fluttuazione casuale nell'ultima epoca.

---

## ⏱️ Min 60-90

Certamente! Ecco degli appunti universitari dettagliati basati sul segmento video fornito, riguardanti le tecniche avanzate di ottimizzazione per l'apprendimento automatico.

---

# Appunti di Ottimizzazione per il Machine Learning

## 1. Algoritmo RMSProp (Root Mean Square Propagation)

L'RMSProp è un metodo di ottimizzazione adattivo progettato per affrontare le limitazioni di AdaGrad, in particolare il problema della decrescita eccessiva del tasso di apprendimento.

### 1.1 Confronto con AdaGrad
*   **Problema di AdaGrad:** AdaGrad accumula i quadrati di tutti i gradienti passati fin dall'inizio dell'addestramento. Di conseguenza, il termine di accumulo ($r$) cresce indefinitamente, portando il tasso di apprendimento adattivo verso lo zero e bloccando prematuramente l'addestramento.
*   **Soluzione di RMSProp:** Introduce un **tasso di decadimento ($\rho$)** (solitamente 0.9). Invece di una somma cumulativa, RMSProp utilizza una **media mobile esponenziale** dei gradienti al quadrato.
    *   *Effetto:* L'algoritmo "dimentica" gradualmente la storia antica, impedendo al termine di accumulo di diventare troppo grande e permettendo all'addestramento di continuare efficacemente anche dopo molte epoche.

### 1.2 Evoluzione verso Adam
*   Sebbene l'RMSProp sia potente, nella pratica moderna viene spesso sostituito da **Adam (Adaptive Moment Estimation)**.
*   **Adam = RMSProp + Momentum:** Combina l'idea di RMSProp (scaling dei gradienti basato sulla loro magnitudo recente) con il Momentum (accumulo della direzione del gradiente).

---

## 2. Metodo di Newton (Ottimizzazione del Secondo Ordine)

I metodi di Newton utilizzano le informazioni sulla curvatura della funzione di perdita (derivate di secondo ordine) per trovare il minimo.

### 2.1 Caratteristiche e Vantaggi
*   **Convergenza:** Teoricamente molto più veloce dei metodi del primo ordine (come il Gradient Descent) perché la direzione punta direttamente verso il minimo di una funzione quadratica.
*   **Applicazioni:** Trova impiego in casi specifici come le **PINN (Physics-Informed Neural Networks)**, ma è raramente utilizzato per le reti neurali profonde standard.

### 2.2 Svantaggi Principali
1.  **Costo di Memoria:** Richiede il calcolo e la memorizzazione della **matrice Hessiana** ($\nabla^2 \mathcal{L}$), che ha dimensioni $N \times N$ (dove $N$ è il numero di parametri). Se $N$ è grande, il costo diventa quadratico ($O(N^2)$), saturando rapidamente la RAM.
2.  **Garanzie di Convergenza:** Le garanzie di convergenza valgono solo se si inizia vicino a un punto di minimo. Lontano dal minimo o in presenza di punti di sella, il metodo di Newton può divergere o comportarsi peggio dei metodi del primo ordine.

---

## 3. Calcolo Efficiente in JAX

JAX offre strumenti avanzati per gestire le derivate di ordine superiore in modo efficiente.

### 3.1 Implementazione della Matrice Hessiana
In JAX, l'Hessiana può essere calcolata combinando la differenziazione in modalità forward e reverse per ottimizzare i costi computazionali:
```python
hess = jax.jacfwd(jax.jacrev(loss))
```
Questa combinazione (`jacfwd` su `jacrev`) è generalmente la più efficiente per funzioni con molti parametri.

### 3.2 Il "Trucco" dell'Hessian-Vector Product (HVP)
Per evitare il costo $O(N^2)$ della memorizzazione dell'intera Hessiana, si può calcolare direttamente il prodotto della Hessiana per un vettore ($v$).

*   **Identità Matematica:**
    $$\nabla^2 \mathcal{L}(x) \cdot v = \nabla_x (\nabla \mathcal{L}(x) \cdot v)$$
*   **Concetto:** Invece di calcolare la matrice completa, si calcola il gradiente del prodotto scalare tra il gradiente della funzione e il vettore $v$.
*   **Vantaggi:**
    *   **Memoria:** Non si costruisce mai la matrice Hessiana; si lavora sempre con campi scalari o vettoriali ($O(N)$).
    *   **Velocità:** È significativamente più veloce (nel video, circa 25 volte più rapido rispetto alla versione con matrice completa) grazie alla riduzione del movimento di dati tra RAM e CPU/GPU.

### 3.3 Risultati Sperimentali
Il video evidenzia come l'implementazione "matrix-free" (HVP) riduca drasticamente la latenza (da microsecondi a frazioni di essi) e renda i metodi del secondo ordine effettivamente utilizzabili su scale più ampie.

---
*Note: Questi appunti coprono i concetti di decadimento esponenziale nel learning rate, i limiti computazionali delle derivate di secondo ordine e le tecniche di differenziazione automatica per l'ottimizzazione efficiente.*

---

## ⏱️ Min 90-120

Ecco degli appunti universitari dettagliati basati sul segmento video fornito, focalizzati sull'ottimizzazione numerica e l'implementazione pratica in Python.

---

# Appunti di Ottimizzazione Numerica e Machine Learning

## 1. Risoluzione di Sistemi Lineari nel Metodo di Newton
Il cuore del metodo di Newton risiede nel calcolo dell'incremento risolvendo il sistema lineare:
$$H \cdot \text{incr} = -G$$
Dove $H$ è la matrice Hessiana e $G$ è il gradiente.

### 1.1 Approccio Diretto vs Iterativo
*   **Approccio Diretto:** Utilizza funzioni come `np.linalg.solve(H, -G)`. 
    *   **Limitazioni:** Richiede il calcolo e la memorizzazione dell'intera matrice Hessiana, il che comporta un elevato costo computazionale ($O(n^3)$ per la risoluzione) e di memoria ($O(n^2)$), proibitivo per problemi ad alta dimensione.
*   **Approccio Matrix-Free (Gradiente Coniugato - CG):** Utilizza metodi iterativi per risolvere il sistema senza costruire esplicitamente $H$.
    *   **Vantaggio:** Richiede solo l'azione della Hessiana su un vettore (Hessian-Vector Product - HVP).
    *   **Requisiti:** Il Gradiente Coniugato è particolarmente efficiente quando la Hessiana è una matrice simmetrica e definita positiva.

### 1.2 Implementazione JAX (Metodo Matrix-Free)
Per implementare Newton in modalità "matrix-free", si utilizza `jax.scipy.sparse.linalg.cg`. Invece di passare la matrice $H$, si passa una funzione lambda che calcola il prodotto Hessiana-vettore:

```python
# Utilizzo del Gradiente Coniugato per risolvere H * incr = -G
incr, info = jax.scipy.sparse.linalg.cg(
    lambda y: hvp_jit(x, y), # Azione della Hessiana valutata in x sul vettore y
    -G,                      # Lato destro del sistema
    tol=eps                  # Tolleranza per la convergenza
)
```
*Nota: In JAX, `hvp_jit(x, y)` calcola efficientemente $\nabla^2 f(x) \cdot y$ senza istanziare la matrice completa.*

---

## 2. Metodi Quasi-Newton: L'Algoritmo BFGS
L'algoritmo **BFGS** (Broyden-Fletcher-Goldfarb-Shanno) è una variante del metodo di Newton che evita completamente il calcolo della Hessiana.

### 2.1 Concetti Chiave
*   **Approssimazione dell'Inversa:** BFGS approssima direttamente l'inversa della Hessiana ($B^{-1}$) in modo iterativo.
*   **Inizializzazione:** Solitamente $B^{-1}$ viene inizializzata come la matrice identità $I$. Inizialmente, l'algoritmo si comporta come una Discesa del Gradiente e converge gradualmente verso il comportamento di Newton man mano che l'approssimazione migliora.
*   **Aggiornamento di Rango 1/2:** Ad ogni iterazione, l'approssimazione viene raffinata aggiungendo correzioni basate sullo spostamento ($s$) e sulla differenza dei gradienti ($y$):
    *   $s = x_{new} - x_{old}$
    *   $y = \nabla f_{new} - \nabla f_{old}$
*   **Formula di Sherman-Morrison:** Viene utilizzata per aggiornare l'inversa della Hessiana in modo efficiente.

### 2.2 Line Search
Nei metodi del secondo ordine, la scelta del learning rate (step size $\alpha$) è critica. Si utilizza spesso una **Line Search** (tramite `sp.optimize.line_search`) per trovare il valore ottimale di $\alpha$ lungo la direzione di ricerca $p$ che minimizza la funzione di perdita.

---

## 3. Implementazione Pratica e Data Preprocessing
Nella transizione alla risoluzione di problemi reali (es. predizione del consumo di carburante - Auto MPG), la gestione dei dati è fondamentale.

### 3.1 Pulizia dei Dati con Pandas
Prima di addestrare un modello (es. una rete neurale artificiale - ANN), è necessario ispezionare e pulire il dataset.

1.  **Ispezione dei Valori Mancanti:**
    ```python
    print(data.isna().sum())
    ```
    Nel dataset Auto MPG, la colonna `Horsepower` presenta spesso valori mancanti (NaN).
2.  **Rimozione (Drop) dei Dati:** Se il numero di righe con dati mancanti è piccolo rispetto al totale, la tecnica più semplice è la rimozione:
    ```python
    data = data.dropna() # Rimuove le righe contenenti valori nulli
    ```
3.  **Data Inspection:** Utilizzare `data.head()`, `data.info()` e `data.describe()` per comprendere la distribuzione e i tipi di dati (es. distinguere tra variabili numeriche e categoriali).

### 3.2 Regolarizzazione
La regolarizzazione consiste nell'aggiungere un termine extra alla funzione di perdita (Loss Function) per penalizzare i pesi della rete neurale troppo grandi. Questo aiuta a prevenire l'overfitting e viene gestito tramite il **tuning degli iperparametri**, monitorando la perdita sul set di validazione.

---

## ⏱️ Min 120-150

# Appunti Universitari: Regolarizzazione nelle Reti Neurali Artificiali (ANN)

## 1. Analisi Esplorativa dei Dati (EDA) e Visualizzazione
Prima di implementare il modello, è fondamentale comprendere la struttura del dataset. Nel caso studio (previsione del consumo di carburante MPG - Miles Per Gallon), vengono utilizzate le librerie **Pandas** e **Seaborn**.

*   **Ispezione dei Metadati:** Utilizzo di `data.info()` per verificare i tipi di dati (es. `float64`, `int64`) e la presenza di valori nulli.
*   **Statistiche Descrittive:** `data.describe()` fornisce una panoramica su media, deviazione standard, quartili e valori estremi.
*   **Distribuzione del Target:** Un `sns.displot` sulla variabile MPG permette di verificare se la distribuzione è ragionevole e identificare eventuali **outlier** (valori anomali) che potrebbero necessitare di rimozione.
*   **Analisi delle Correlazioni:**
    *   **Heatmap:** Utilizzata per visualizzare i coefficienti di correlazione tra le variabili (es. numero di cilindri, cilindrata, potenza, peso).
    *   **Insights:** Esiste una forte correlazione positiva tra potenza/peso e il numero di cilindri, e una correlazione negativa tra questi fattori e l'efficienza del carburante (MPG).
*   **Pairplot:** Consente di osservare le relazioni lineari o quadratiche tra coppie di variabili (es. relazione quasi lineare tra cilindrata e potenza; relazione quadratica tra potenza ed efficienza).

## 2. Pre-processing dei Dati
*   **Normalizzazione:** È essenziale applicare una trasformazione affine per garantire che ogni caratteristica abbia media zero e deviazione standard unitaria. Questo facilita la convergenza della rete neurale.
*   **Violin Plot:** Utilizzati dopo la normalizzazione per verificare che tutte le distribuzioni siano ben educate, senza "code lunghe" e confinate in un intervallo simile.
*   **Train-Validation Split:** Il dataset viene diviso tipicamente in un set di addestramento (80%) e un set di validazione (20%) per monitorare la capacità di generalizzazione del modello.

## 3. Configurazione della Rete Neurale (ANN)
Il modello implementato è una rete **feedforward** standard.
*   **Inizializzazione dei Parametri:** Viene utilizzata l'inizializzazione **Glorot Normal** (o Xavier) per i pesi, campionando da una distribuzione gaussiana con media zero e varianza specifica basata sul numero di neuroni di input/output. I bias sono inizializzati a zero.
*   **Funzione di Attivazione:** Viene impiegata la **ReLU** (Rectified Linear Unit) nei layer nascosti.
*   **Output:** Trattandosi di un problema di regressione scalare (predizione di un singolo valore MPG), il layer di output ha un solo neurone.

## 4. Implementazione della Regolarizzazione $L^2$
Il nucleo della lezione riguarda l'aggiunta di un termine di penalità alla funzione di costo per evitare l'overfitting.

### 4.1 Mean Sum of Weights (MSW)
Il termine di regolarizzazione MSW è la media dei quadrati di tutti i pesi della rete:
$$MSW = \frac{1}{n_{weights}} \sum_{i=1}^{n_{weights}} w_i^2$$
*Nota: Nel calcolo dell'MSW vengono inclusi solo i pesi ($w$), escludendo i bias ($b$).*

### 4.2 Funzione di Loss Totale
La funzione di perdita totale ($L$) diventa la somma dell'errore quadratico medio (MSE) e del termine MSW pesato:
$$\mathcal{L} = MSE + \beta \cdot MSW$$
*   **$\beta$ (Beta):** È l'iperparametro di penalizzazione. Controlla il bilanciamento tra l'aderenza ai dati (MSE) e la semplicità del modello (MSW).

## 5. Training e Ottimizzazione
L'addestramento avviene tramite **Stochastic Gradient Descent (SGD)** con le seguenti caratteristiche:
*   **Momentum ($\alpha=0.9$):** Per accelerare la convergenza e superare minimi locali.
*   **Learning Rate Decay:** Il tasso di apprendimento diminuisce linearmente nel tempo.
*   **Monitoraggio:** Vengono plottate le curve di loss sia per il training che per la validazione. Tipicamente, la loss di validazione è leggermente superiore a quella di training.

## 6. Hyperparameter Tuning: Ricerca su Griglia (Grid Search)
Per trovare il valore ottimale di $\beta$, viene eseguita una ricerca sistematica (Grid Search) in un intervallo definito (es. da 0 a 2 con step di 0.25).

### 6.1 Effetti della penalizzazione Beta
Dall'analisi dei risultati (tabella `hyper_tuning_df`) si osserva che:
*   **All'aumentare di $\beta$:**
    *   **MSW diminuisce:** La rete è costretta a mantenere pesi di magnitudo inferiore (modello più semplice).
    *   **Training MSE aumenta:** Il modello ha meno libertà di "memorizzare" perfettamente i dati di addestramento a causa dei vincoli aggiuntivi.
    *   **Validation MSE:** Inizialmente diminuisce, indicando una migliore generalizzazione (riduzione dell'overfitting), per poi risalire se la regolarizzazione diventa eccessiva.

## 7. La Curva L di Tikhonov
Viene introdotta la **Curva L di Tikhonov**, che mette in relazione l'errore di addestramento (`MSE_train`) e la complessità del modello (`MSW`).
*   Questa curva rappresenta una sorta di **frontiera di Pareto**.
*   Dimostra visivamente il **trade-off**: riducendo la magnitudo dei pesi (MSW), l'errore sui dati di training tende inevitabilmente a salire.
*   L'obiettivo della regolarizzazione è trovare il "gomito" della curva dove si ottiene il miglior compromesso per la generalizzazione.

---

