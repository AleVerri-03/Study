# 📝 Appunti: Matteo Caldana's Personal Room-20251121 1455-1
**Modello:** gemini-3-flash-preview | **Data:** 23/01/2026

---

## ⏱️ Min 0-30

Ecco degli appunti universitari dettagliati basati sul segmento video riguardante l'implementazione di un modello di regressione con reti neurali utilizzando il dataset **California Housing**.

---

# Note del Corso: Regressione con Reti Neurali (Caso Studio: California Housing)

## 1. Introduzione al Progetto
L'obiettivo della sessione è implementare da zero un problema di **regressione** utilizzando una rete neurale. Il dataset utilizzato è il celebre **California Housing dataset**, derivante dal censimento statunitense del 1990.

### Obiettivo Finale
Predire il costo di un'abitazione (`median_house_value`) sulla base di diverse caratteristiche (feature) socio-economiche e geografiche.

---

## 2. Setup dell'Ambiente e Librerie
Per la manipolazione e visualizzazione dei dati vengono utilizzate le seguenti librerie standard in ambiente Python/Google Colab:

*   **Pandas (`pd`)**: Manipolazione di tabelle (DataFrame) e serie temporali.
*   **Matplotlib (`plt`)** & **Seaborn (`sns`)**: Visualizzazione statistica dei dati.
*   **NumPy (`np`)** & **Jax**: Calcolo numerico e differenziazione automatica (Jax è ottimizzato per il machine learning).

---

## 3. Ispezione e Pulizia dei Dati (Exploratory Data Analysis - EDA)

### Ispezione Iniziale
Vengono utilizzati i metodi standard di Pandas per comprendere la struttura del dataset:
*   `data.head()`: Visualizza le prime righe. Le feature includono `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income` e il target `median_house_value`.
*   `data.info()`: Conferma che ci sono **17.000 entry** e che tutte le variabili sono di tipo `float64`. Questo è ideale per una rete neurale poiché non richiede conversioni di tipi categorici o stringhe in questa fase.
*   `data.describe()`: Fornisce statistiche descrittive (media, deviazione standard, min/max, quartili) utili per identificare anomalie o outlier.

### Analisi del Target e Troncamento dei Dati
Visualizzando la distribuzione del `median_house_value` tramite un istogramma con **KDE (Kernel Density Estimation)**:
*   **Anomalia rilevata**: Si nota un picco innaturale a $500.000$. Questo indica che i dati originali sono stati troncati: ogni casa con valore superiore a mezzo milione è stata raggruppata forzatamente in questa categoria.
*   **Azione di Cleaning**: Per non confondere il modello durante l'apprendimento, le righe con valore superiore o uguale a $500.001$ vengono rimosse tramite una maschera booleana:
    ```python
    data = data[data['median_house_value'] < 500001]
    ```

---

## 4. Analisi delle Correlazioni
L'analisi della matrice di correlazione (`data.corr()`) permette di capire quali feature influenzano maggiormente il target.

### Visualizzazione tramite Heatmap
Una **Heatmap** di Seaborn facilita l'identificazione visiva dei coefficienti di correlazione di Pearson (compresi tra -1 e 1):
*   **Target vs Feature**: Il valore mediano della casa ha una forte correlazione positiva (~0.65) con il `median_income` (reddito medio).
*   **Multicollinearità**: Si notano forti correlazioni tra `total_rooms` e `total_bedrooms` (~0.93) e tra `population` e `households` (~0.91). Questo è logico: più stanze ci sono, più è probabile che ci siano più camere da letto.

---

## 5. Visualizzazione Geografica
Utilizzando uno **scatterplot** con `longitude` (asse X) e `latitude` (asse Y), e mappando il colore (`hue`) al `median_house_value`:
*   Si ricostruisce chiaramente la forma geografica della California.
*   **Insight**: Le case più costose (punti più scuri/intensi) sono concentrate lungo la **costa** e nelle aree metropolitane di San Francisco e Los Angeles.
*   **Feature Engineering (Idea)**: Un suggerimento per migliorare il modello futuro è creare variabili sintetiche, come la "distanza da San Francisco" o la "distanza da Los Angeles".

---

## 6. Pre-elaborazione: Normalizzazione dei Dati
Le reti neurali faticano a convergere se le feature hanno ordini di grandezza molto differenti.

### Standardizzazione (Z-score Normalization)
Viene applicata una trasformazione affine per garantire che ogni feature abbia **media zero** e **varianza unitaria**:
$$\hat{x} = \frac{x - \mu}{\sigma}$$
Dove $\mu$ è la media e $\sigma$ la deviazione standard.

### Verifica tramite Violin Plot
Il **Violin Plot** permette di visualizzare la densità di probabilità dei dati standardizzati per tutte le colonne contemporaneamente. Dopo la normalizzazione, la maggior parte delle distribuzioni risulta centrata sullo zero, rendendo l'addestramento della rete più stabile ed efficiente.

---

## 7. Suddivisione del Dataset (Train-Validation Split)
Per una valutazione corretta, il dataset viene diviso in tre parti:

1.  **Training Set (80%)**: Utilizzato per calcolare il gradiente e aggiornare i pesi della rete.
2.  **Validation Set (20%)**: Utilizzato per il **Tuning dei Iperparametri** (es. learning rate, numero di layer). Serve a monitorare l'overfitting senza guardare i dati di test.
3.  **Test Set**: (Caricato separatamente) Serve per la valutazione finale del modello su dati mai visti.

**Procedura**:
*   Viene fissato un `seed` (casuale) per garantire la riproducibilità.
*   I dati vengono mescolati (`np.random.shuffle`) prima dello split per evitare bias derivanti dall'ordinamento originale del dataset.

---

## ⏱️ Min 30-60

Ecco degli appunti universitari dettagliati basati sul segmento video riguardante la configurazione di una Rete Neurale Artificiale (ANN) per un problema di regressione (California Housing dataset).

---

# Appunti di Machine Learning: Configurazione e Implementazione di una ANN

## 1. Preparazione dei Dati
Prima di definire l'architettura della rete, è fondamentale suddividere il dataset in sottoinsiemi per l'addestramento e la validazione.

*   **Shuffling:** Utilizzo di `np.random.shuffle` per mescolare i dati in modo casuale, evitando bias dovuti all'ordinamento originale.
*   **Split:** 
    *   **80% Training Set:** Utilizzato per l'apprendimento dei parametri del modello.
    *   **20% Validation Set:** Utilizzato per monitorare le performance e prevenire l'overfitting.
*   **Convenzione sulle Matrici:**
    *   `x_train` / `x_valid`: Contengono le feature (tutte le colonne tranne l'ultima).
    *   `y_train` / `y_valid`: Contengono il target (l'ultima colonna, ovvero il valore da predire).

---

## 2. Inizializzazione dei Parametri (`initialize_params`)
L'inizializzazione corretta dei pesi è cruciale per la convergenza della rete. Pesi troppo piccoli possono causare la scomparsa del gradiente, mentre pesi troppo grandi possono portarlo all'esplosione.

### 2.1 Strategia di Inizializzazione di Xavier/Glorot
Il video introduce l'uso dell'**Inizializzazione di Glorot** (o Xavier), preferibile rispetto a una distribuzione normale standard semplice. L'obiettivo è mantenere costante la varianza delle attivazioni e dei gradienti attraverso i layer.

*   **Pesi ($W$):** Estratti da una distribuzione normale con media zero e deviazione standard:
    $$\sigma = \sqrt{\frac{2}{n + m}}$$
    Dove:
    *   $n$: numero di neuroni in input al layer.
    *   $m$: numero di neuroni in output del layer.
*   **Bias ($b$):** Inizializzati a zero (`np.zeros`).

### 2.2 Implementazione del Codice
La funzione riceve in input `layers_sizes`, una lista che definisce la struttura della rete (es. `[input_size, hidden_1, ..., output_size]`).

```python
def initialize_params(layers_sizes):
    np.random.seed(0) # Per riproducibilità
    params = []
    
    # Iterazione sulle coppie di layer (n, m)
    for i in range(len(layers_sizes) - 1):
        n = layers_sizes[i+1] # Neuroni output layer corrente
        m = layers_sizes[i]   # Neuroni input layer corrente
        
        # Coefficiente di Glorot
        coef = np.sqrt(2 / (n + m))
        
        # Inizializzazione Pesi e Bias
        W = np.random.randn(n, m) * coef
        b = np.zeros((n, 1))
        
        # Salvataggio come coppia (W, b)
        params.append([W, b])
        
    return params
```

---

## 3. Implementazione del Forward Pass (`ANN`)
La funzione `ANN(x, params)` definisce come i dati fluiscono attraverso la rete dall'input all'output.

### 3.1 Trasformazioni Lineari e Attivazioni
Per ogni layer, la rete calcola una trasformazione affine seguita da una funzione di attivazione non lineare (nel video viene usata la tangente iperbolica `tanh`).

*   **Equazione del layer:** $a = \sigma(W \cdot x + b)$
*   **Gestione della Convenzione:** Poiché i dati sono spesso forniti come (campioni $\times$ feature), il codice esegue inizialmente una trasposizione per operare su colonne come vettori sample, per poi trasporre nuovamente il risultato finale.

### 3.2 Eccezione per l'Ultimo Layer (Regressione)
In un problema di **regressione**, l'ultimo layer **non deve avere una funzione di attivazione** (o deve avere un'attivazione lineare). 
*   **Motivazione:** Se usassimo `tanh` nell'ultimo layer, l'output sarebbe limitato tra $[-1, 1]$. Per predire prezzi di case o altri valori reali illimitati, l'ultima trasformazione deve essere puramente lineare ($W \cdot layer + b$).

### 3.3 Implementazione del Codice
Utilizzo della funzione `enumerate` per identificare l'ultimo layer durante il loop.

```python
def ANN(x, params):
    # Trasposizione iniziale per convenzione (feature per riga)
    layer = x.T 
    
    for i, (W, b) in enumerate(params):
        # Trasformazione lineare
        layer = W @ layer + b
        
        # Attivazione tanh solo per layer interni (n-1)
        if i < len(params) - 1:
            layer = np.tanh(layer)
            
    # Trasposizione finale per tornare al formato originale
    return layer.T 
```

---

## Glossario Tecnico
*   **Forward Pass:** Il processo di calcolo dell'output della rete a partire dall'input.
*   **Glorot/Xavier Initialization:** Tecnica di inizializzazione che adatta la scala dei pesi in base alla dimensione dei layer connessi.
*   **Linear Layer (Affine):** Operazione $Wx + b$ che proietta i dati in un nuovo spazio vettoriale.
*   **Activation Function:** Funzione non lineare necessaria per permettere alla rete di apprendere relazioni complesse (non lineari).
*   **Broadcasting:** (In NumPy) Meccanismo che permette di eseguire operazioni aritmetiche tra array di forme diverse (usato per sommare il vettore bias a ogni colonna della matrice dei pesi).

---

## ⏱️ Min 60-90

Ecco degli appunti universitari dettagliati basati sul video tutorial sull'implementazione di una Rete Neurale Artificiale (ANN) utilizzando la libreria **JAX**.

---

# Appunti: Implementazione e Training di una ANN con JAX

## 1. Funzioni di Attivazione
Le funzioni di attivazione introducono la non-linearità nel modello, permettendo alla rete di apprendere relazioni complesse.

*   **Tanh (Tangente Iperbolica):** Molto comune, schiaccia i valori nell'intervallo $[-1, 1]$.
    *   Implementazione: `activation_fn = jnp.tanh`
*   **ReLU (Rectified Linear Unit):** Definisce l'output come il massimo tra 0 e l'input.
    *   Definizione matematica: $f(x) = \max(0, x)$
    *   Implementazione: `activation_fn = lambda x: jnp.maximum(0.0, x)`
*   **Nota Tecnica su JAX:** È fondamentale utilizzare i moduli di JAX (es. `jnp`) per le operazioni matematiche all'interno della funzione ANN. Se si utilizzano librerie esterne (come NumPy standard), la **differenziazione automatica** di JAX fallirà durante il calcolo dei gradienti.

## 2. Architettura della Rete (Forward Pass)
Il video mostra una funzione `ANN(x, params)` che itera attraverso i livelli della rete:
1.  Calcola il prodotto matriciale tra input e pesi, aggiungendo il bias: `layer = w @ layer + b`.
2.  Applica la funzione di attivazione a tutti i livelli tranne l'ultimo.
3.  **L'ultimo livello:** Spesso non ha attivazione (attivazione lineare) nei problemi di regressione per permettere alla rete di produrre qualsiasi valore reale, potenzialmente superiore a 1 (limite della Tanh).

### Verifica delle Dimensioni (Shapes)
Prima dell'addestramento, è prassi verificare le dimensioni degli array:
*   **Input ($x_{train}$):** $(12948, 8)$ $\rightarrow$ 12948 campioni con 8 feature ciascuno.
*   **Output previsto:** $(12948, 1)$ $\rightarrow$ un valore predetto per ogni campione.

## 3. Funzione di Perdita (Loss Function)
La funzione di perdita misura la discrepanza tra le predizioni del modello e i valori target reali.

### Mean Squared Error (MSE)
Per problemi di regressione, la scelta standard è l'errore quadratico medio:
$$\mathcal{L}(x, y, \theta) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \text{ANN}(x_i, \theta))^2$$

*   **Implementazione:**
    ```python
    def loss(x, y, params):
        model_pred = ANN(x, params)
        error = model_pred - y
        return jnp.mean(error * error)
    ```
*   **Alternativa L1:** Si potrebbe usare il valore assoluto (`jnp.abs`), meno sensibile agli outlier rispetto al quadrato, ma l'MSE è generalmente preferito per la sua convessità e differenziabilità.

## 4. Addestramento: Full Batch Gradient Descent
L'obiettivo è minimizzare la funzione di perdita aggiornando i parametri $\theta$.

### Iperparametri del Training
*   `layers_size`: Lista che definisce i neuroni per livello (es: `[8, 20, 20, 1]`).
*   `num_epochs`: Numero di iterazioni totali sull'intero dataset.
*   `lr` (Learning Rate): La velocità di aggiornamento dei parametri.

### Il Loop di Ottimizzazione
1.  **Calcolo del Gradiente:** Si usa `jax.grad` (o una sua versione ottimizzata con JIT).
2.  **Aggiornamento dei Parametri:** $\theta_{nuovo} = \theta_{vecchio} - \lambda \cdot \nabla_\theta \mathcal{L}$.
3.  **Monitoraggio:** Si salvano i valori della loss per i set di *training* e *validation* per ogni epoca.

## 5. Strumenti Avanzati di JAX

### JAX JIT (Just-In-Time Compilation)
L'uso di `@jax.jit` o `jax.jit(funzione)` compila la funzione per l'esecuzione accelerata su CPU, GPU o TPU, velocizzando drasticamente il calcolo dei gradienti e della loss.

### JAX Tree Utility (`tree_map`)
Quando i parametri della rete sono memorizzati in strutture complesse (liste di tuple, dizionari annidati), l'aggiornamento manuale con loop `for` è inefficiente e prono a errori.
`jax.tree_util.tree_map` permette di applicare una funzione (come l'aggiornamento del gradiente) a ogni singolo elemento della struttura "albero" dei parametri in un'unica riga:
```python
params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
```

## 6. Analisi dei Risultati
Il monitoraggio della curva di perdita su scala **Log-Log** è utile per visualizzare i miglioramenti, che spesso sono molto rapidi all'inizio e rallentano esponenzialmente col tempo.
*   **Convergenza:** Se la curva del training e quella della validazione scendono insieme, il modello sta imparando bene.
*   **Overfitting:** Se la curva di validazione inizia a salire o si stabilizza mentre quella di training continua a scendere, il modello sta memorizzando i dati di training invece di generalizzare.

---
**Termini Tecnici Chiave:**
*   **Forward Pass:** Calcolo dell'output partendo dall'input.
*   **Automatic Differentiation:** Capacità di calcolare derivate esatte di funzioni implementate in codice.
*   **Vanishing Gradient:** Problema potenziale con attivazioni come Tanh se i pesi non sono inizializzati correttamente.
*   **Log-Log Scale:** Grafico con assi logaritmici su entrambi i lati, ideale per visualizzare ordini di grandezza della loss.

---

## ⏱️ Min 90-120

Ecco degli appunti universitari dettagliati basati sulla lezione video riguardante l'implementazione dello **Stochastic Gradient Descent (SGD)** con il framework **JAX**.

---

# Appunti di Machine Learning: Stochastic Gradient Descent (SGD) e JAX

## 1. Introduzione allo Stochastic Gradient Descent (SGD)
L'obiettivo principale del passaggio dal Gradient Descent "vanilla" (o Batch Gradient Descent) allo **Stochastic Gradient Descent (SGD)** è l'efficienza computazionale. Mentre il Gradient Descent standard calcola il gradiente utilizzando l'intero dataset per ogni singolo aggiornamento dei parametri, l'SGD utilizza solo una piccola porzione di dati, denominata **minibatch**.

### Vantaggi dell'SGD:
*   **Velocità:** Molto più rapido per iterazione rispetto al Batch GD.
*   **Esplorazione:** La natura stocastica (casuale) aiuta l'algoritmo a sfuggire ai minimi locali, permettendo una migliore esplorazione della superficie di costo (Loss Function).
*   **Scalabilità:** Indispensabile quando il dataset è troppo grande per essere caricato interamente in memoria (RAM).

## 2. Implementazione Tecnica con JAX
Il video illustra come implementare l'SGD in Python utilizzando la libreria **JAX**, nota per le sue capacità di differenziazione automatica e compilazione JIT (Just-In-Time).

### Funzioni Chiave di JAX utilizzate:
*   `jax.jit`: Compila le funzioni per un'esecuzione accelerata su CPU/GPU/TPU.
*   `jax.grad`: Calcola automaticamente il gradiente della funzione di perdita rispetto ai parametri.
*   `jax.tree_util.tree_map`: Permette di applicare una funzione (come l'aggiornamento dei pesi) a intere strutture dati complesse (PyTrees), come i dizionari di parametri dei layer neurali.

### Definizione degli Iperparametri
Per l'SGD sono fondamentali i seguenti parametri:
*   `num_epochs`: Numero totale di passaggi attraverso l'intero dataset.
*   `learning_rate_max` e `learning_rate_min`: Definiscono il range di aggiornamento dei pesi.
*   `batch_size`: La dimensione del minibatch (es. 1000 campioni).
*   `learning_rate_decay`: Una strategia per ridurre il passo di apprendimento nel tempo.

## 3. Dinamiche del Learning Rate (Decay)
Un concetto cruciale trattato è il **Linear Decay del Learning Rate ($\lambda$)**. La formula utilizzata è:
$$\lambda_k = \max \left( \lambda_{min}, \lambda_{max} \left( 1 - \frac{k}{K} \right) \right)$$
dove $k$ è l'epoca corrente e $K$ è il numero totale di epoche.

**Perché è necessario il decadimento?**
All'inizio dell'addestramento, vogliamo un learning rate elevato per esplorare rapidamente la funzione di perdita (**Exploration**). Tuttavia, poiché l'SGD calcola gradienti approssimativi basati sui minibatch, un learning rate costantemente alto causerebbe continue oscillazioni attorno al minimo senza mai convergere stabilmente. Riducendo il learning rate, "congeliamo" il modello vicino al minimo ottimale (**Exploitation/Greediness**).

## 4. Struttura dell'Algoritmo (Loop di Training)
L'implementazione segue questi step logici per ogni epoca:

1.  **Permutazione dei dati:** All'inizio di ogni epoca, gli indici dei dati vengono rimescolati casualmente (`np.random.permutation`). Questo garantisce che ogni minibatch sia diverso tra le epoche, riducendo il rischio di overfitting e bias sistematici.
2.  **Iterazione per Minibatch:** Il dataset viene suddiviso in blocchi di dimensione `batch_size`.
3.  **Calcolo del Gradiente:** Si calcola il gradiente (`grad_jit`) solo sui campioni del minibatch corrente.
4.  **Aggiornamento dei Parametri:** Si aggiornano i pesi sottraendo il gradiente moltiplicato per il learning rate corrente ($\theta = \theta - \lambda \nabla L$).
5.  **Monitoraggio:** Si calcola e si salva la Loss (sia per il training che per la validazione) per monitorare la convergenza.

## 5. Analisi dei Risultati e Grafici
Il video mostra un confronto visivo tra la Loss con e senza decadimento:
*   **Senza Decadimento:** Il grafico della Loss mostra forti oscillazioni finali, simili a "salti" continui tra diverse valli della funzione di costo non convessa.
*   **Con Decadimento:** Le oscillazioni diminuiscono progressivamente, portando a una curva più fluida e a un valore di perdita finale più basso e stabile.

## 6. Fase di Test
Una volta addestrato il modello (ottenuti i parametri ottimali), viene effettuato il test su un dataset indipendente (`california_housing_test.csv`).
*   **Normalizzazione:** I dati di test devono subire lo stesso processo di normalizzazione (mean/std) applicato ai dati di training.
*   **Valutazione:** Viene calcolato il **RMSE (Root Mean Square Error)** per quantificare l'errore medio nelle predizioni dei prezzi delle case.

---

### Glossario Tecnico:
*   **Epoch (Epoca):** Un ciclo completo di addestramento su tutti i campioni del dataset.
*   **Minibatch:** Un sottoinsieme casuale di campioni utilizzato per un singolo aggiornamento dei parametri.
*   **Stochasticity (Stocasticità):** L'elemento di casualità introdotto dall'uso di campioni parziali, che permette di approssimare il gradiente reale.
*   **JIT (Just-In-Time):** Tecnica di compilazione che trasforma il codice Python in codice macchina altamente ottimizzato al momento dell'esecuzione.

---

## ⏱️ Min 120-150

Certamente. Ecco degli appunti universitari dettagliati basati sulla lezione di programmazione e data science presentata nel video.

---

# Appunti di Data Science: Analisi, Visualizzazione e Valutazione di Modelli di Regressione

**Argomento:** Pre-elaborazione, predizione e valutazione di una rete neurale applicata al set di dati *California Housing*.
**Ambiente di sviluppo:** Google Colab.
**Linguaggio:** Python.
**Librerie principali:** Pandas, NumPy, Matplotlib, Seaborn, JAX.

---

## 1. Caricamento e Pulizia dei Dati di Test

La fase iniziale prevede il caricamento del dataset di test e la rimozione di eventuali valori anomali (*outlier*) per garantire la coerenza con le operazioni effettuate sul dataset di addestramento.

```python
# Caricamento del dataset di test
data_test = pd.read_csv('./sample_data/california_housing_test.csv')

# Filtraggio degli outlier: rimozione dei valori massimi troncati (es. 500.001)
data_test = data_test[data_test['median_house_value'] < 500001]
```

## 2. Normalizzazione dei Dati (Standardizzazione)

Un concetto fondamentale nel Machine Learning è che **la normalizzazione dei dati di test deve basarsi esclusivamente sui parametri del set di addestramento** (media e deviazione standard del training set). 

### 2.1 Perché usare i parametri del training set?
L'utilizzo dei parametri del test set comporterebbe un errore metodologico noto come **Data Leakage** (fuga di dati). Il modello non deve avere alcuna informazione sulla distribuzione dei dati che dovrà "predire" in fase di test.

```python
# Normalizzazione Z-score del dataset di test
# NOTA: data_mean e data_std provengono dal training set
data_test_normalized = (data_test - data_mean) / data_std
```

### 2.2 Preparazione per la Rete Neurale
Dopo la normalizzazione, i dati vengono convertiti in formato NumPy e suddivisi in feature di input ($X$) e target ($Y$).

```python
# Conversione in array NumPy
data_test_normalized_np = data_test_normalized.to_numpy()

# Suddivisione in input (tutte le colonne tranne l'ultima) e target (ultima colonna)
x_test = data_test_normalized_np[:, :-1]
y_test = data_test_normalized_np[:, -1:]
```

## 3. Fase di Predizione e Denormalizzazione

Il modello di rete neurale (`ANN`) riceve l'input normalizzato e produce una predizione, anch'essa in formato normalizzato. Per interpretare il risultato in termini reali (valore monetario in dollari), è necessario invertire la normalizzazione.

### 3.1 Formula di Denormalizzazione
Dato un valore standardizzato $z$, la formula per recuperare il valore originale $x$ è:
$$x = (z \cdot \sigma_{training}) + \mu_{training}$$

```python
# Predizione tramite il modello ANN
y_pred_norm = ANN(x_test, params)

# Recupero dei parametri originali della colonna target (median_house_value)
std = data['median_house_value'].std()
avg = data['median_house_value'].mean()

# Denormalizzazione delle predizioni
Y_pred = (y_pred_norm * std) + avg

# Denormalizzazione del target reale per confronto
Y_test = data_test['median_house_value']
```

## 4. Visualizzazione dei Risultati

La visualizzazione è cruciale per comprendere qualitativamente la bontà del modello.

### 4.1 Scatter Plot con Matplotlib
Si confrontano i valori reali ($Y_{test}$) con le predizioni ($Y_{pred}$). In un modello perfetto, tutti i punti dovrebbero giacere sulla **bisettrice del primo quadrante**.

*   **`plt.axis('equal')`**: Assicura che la scala degli assi X e Y sia identica, evitando distorsioni visive nella pendenza della bisettrice.

### 4.2 Analisi Avanzata con Seaborn
L'uso di `sns.jointplot` permette una visualizzazione più ricca:
*   **Scatter Plot centrale**: mostra la correlazione tra target e predizione.
*   **Istogrammi marginali**: mostrano la distribuzione delle frequenze per entrambe le variabili sui rispettivi assi.

```python
import seaborn as sns
import pandas as pd

# Creazione di un DataFrame temporaneo per Seaborn
test_df = pd.DataFrame({'target': Y_test.flatten(), 'predicted': Y_pred.flatten()})

# Visualizzazione con JointPlot
sns.jointplot(data=test_df, x='target', y='predicted')
```

## 5. Valutazione Quantitativa: RMSE

L'**RMSE (Root Mean Square Error)** è la metrica principale utilizzata per quantificare l'errore medio del modello in unità monetarie.

$$RMSE = \sqrt{\text{mean}((Y_{pred} - Y_{test})^2)}$$

```python
# Calcolo dell'errore (usando JAX/NumPy)
error = Y_pred.flatten() - Y_test.to_numpy().flatten()
rmse = jnp.sqrt(jnp.mean(error * error))

print(f"RMSE: {rmse / 1000.0:.2f} k$")
```

---

### Nota tecnica: Ottimizzazione del calcolo delle potenze
Il docente sottolinea che, dal punto di vista dell'efficienza del codice, calcolare il quadrato tramite moltiplicazione semplice (`error * error`) è preferibile rispetto all'operatore di potenza (`error ** 2`). 
*   L'operatore `**` è una procedura generale che gestisce esponenti in virgola mobile, risultando più lenta.
*   La moltiplicazione diretta è una delle operazioni più veloci eseguibili dalla CPU/GPU.

---

