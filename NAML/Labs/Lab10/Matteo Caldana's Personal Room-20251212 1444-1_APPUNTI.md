# 📝 Appunti: Matteo Caldana's Personal Room-20251212 1444-1
**Modello:** gemini-3-flash-preview | **Data:** 24/01/2026

---

## ⏱️ Min 0-30

Ecco degli appunti universitari dettagliati basati sulla lezione di laboratorio relativa al riconoscimento della scrittura autografa tramite reti neurali convoluzionali (CNN).

---

# Laboratorio di Machine Learning: Handwriting Recognition con CNN (JAX & Flax)

## 1. Obiettivi del Laboratorio
L'obiettivo principale è la costruzione di una **Reti Neurale Convoluzionale (CNN)** partendo da zero per classificare cifre scritte a mano utilizzando il dataset **MNIST**. A differenza dei laboratori precedenti basati solo su JAX, qui vengono introdotte librerie di alto livello per facilitare la gestione di architetture complesse.

---

## 2. Stack Tecnologico
*   **JAX:** Motore principale per il calcolo numerico ad alte prestazioni e la differenziazione automatica.
*   **Flax (`flax.linen`):** Libreria per la definizione di architetture di reti neurali in JAX. Offre un'interfaccia più semplice per comporre layer (convoluzionali, densi, ecc.).
*   **Optax:** Libreria per l'implementazione di ottimizzatori (es. Adam, RMSProp). Si passa dal classico SGD (Stochastic Gradient Descent) a ottimizzatori "state-of-the-art".
*   **Matplotlib:** Utilizzata per la visualizzazione dei dati e dei risultati.
*   **NumPy:** Utilizzata per la manipolazione iniziale del dataset prima della conversione in tensori JAX.

---

## 3. Configurazione dell'Ambiente (Google Colab)
Per garantire l'efficienza computazionale delle CNN, è fondamentale modificare il tipo di runtime in Google Colab selezionando un acceleratore hardware (**GPU o TPU**). Senza questo passaggio, l'esecuzione del codice risulterebbe estremamente lenta.

---

## 4. Ingestione e Analisi Preliminare dei Dati
Il dataset utilizzato è una versione ridotta di MNIST (`mnist_train_small.csv`), caricato tramite `np.genfromtxt`.
*   **Shape iniziale:** Matrix di $(20000, 785)$.
    *   $20.000$ righe (campioni/immagini).
    *   $785$ colonne.
*   **Struttura delle colonne:**
    *   La **prima colonna** contiene le etichette (*labels*), ovvero la cifra corretta ($0-9$).
    *   Le restanti **784 colonne** rappresentano i pixel dell'immagine ($28 \times 28$).

---

## 5. Pre-processing dei Dati

### 5.1 Separazione e Reshaping
Le immagini devono essere trasformate da vettori piatti a tensori di quarto ordine per essere processate correttamente dai layer convoluzionali:
*   **Target Shape:** `(campioni, x-pixel, y-pixel, canali)`.
*   Nel caso specifico: `(20000, 28, 28, 1)`. Il canale è singolo poiché le immagini sono in scala di grigio.
*   **Normalizzazione:** I valori dei pixel (originariamente $0-255$) vengono divisi per $255$ per scalarli nell'intervallo $[0, 1]$. Questo migliora la convergenza durante l'addestramento.

### 5.2 One-Hot Encoding delle Etichette
Per la classificazione multiclasse, le etichette interre vengono convertite in vettori di probabilità.
*   **Definizione:** Un vettore dove l'unica componente non nulla è quella corrispondente alla classe target.
*   **Esempio:** La cifra `3` diventa `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.
*   **Implementazione efficiente:** Invece di iterare su ogni riga (costoso in Python), si preferisce l'uso di **maschere booleane** (*vectorized approach*). Si crea una matrice di zeri di shape `(num_campioni, 10)` e si assegna $1$ dove `labels == i` per ogni classe $i$.
*   **Validazione:** È necessario verificare che la somma di ogni riga della matrice risultante sia esattamente $1$.

---

## 6. Visualizzazione del Dataset
Viene effettuato un plot dei primi 30 campioni per verificare l'integrità del pre-processing.
*   **Metodo:** Utilizzo di `plt.subplots(ncols=10, nrows=3)`.
*   Le immagini vengono visualizzate in scala di grigio (`cmap='gray'`).
*   Ogni immagine riporta come titolo l'etichetta corretta per validare visivamente il dataset. Alcuni campioni possono risultare ambigui (es. un $6$ con tratti mancanti), sottolineando la difficoltà del compito di classificazione.

---

## 7. Definizione dell'Architettura della Rete (CNN)
La rete viene definita come una classe che eredita da `nn.Module` di Flax.

### 7.1 Layer e Componenti
1.  **`nn.Conv`:** Layer convoluzionale. Estraggono feature spaziali (linee verticali, orizzontali, bordi).
    *   `features=32`: Numero di filtri (kernel) da apprendere.
    *   `kernel_size=(3, 3)`: Dimensione della finestra di convoluzione.
2.  **`nn.relu`:** Funzione di attivazione (Rectified Linear Unit). Introduce non-linearità. È lo standard di riferimento per le CNN.
3.  **`nn.avg_pool`:** Layer di pooling. Riduce la dimensionalità spaziale mantenendo le informazioni salienti.
    *   `window_shape=(2, 2)`, `strides=(2, 2)`.
4.  **Flattening:** L'output multidimensionale dei layer convoluzionali viene "appiattito" in un vettore tramite `x.reshape(x.shape[0], -1)`.
5.  **`nn.Dense`:** Layer completamente connessi (Fully Connected) per la classificazione finale basata sulle feature estratte.

### 7.2 Il Decoratore `@nn.compact`
Viene utilizzato il decoratore `@nn.compact` all'interno della classe. Questo permette di definire i layer direttamente all'interno del metodo `__call__` (il metodo che definisce il passaggio in avanti o *forward pass*), rendendo il codice più leggibile ed evitando di definire i parametri separatamente in un costruttore.

---

**Termini Tecnici Chiave:**
*   **Tensor:** Array multidimensionale (generalizzazione di matrici).
*   **Kernel/Filtro:** Matrice di pesi piccola applicata all'immagine per estrarre caratteristiche.
*   **Stride:** Il "passo" con cui il filtro scorre sull'immagine.
*   **Activation Function:** Funzione applicata all'output di un neurone per determinare se deve attivarsi.
*   **Softmax:** Spesso usata nell'ultimo layer per trasformare l'output in una distribuzione di probabilità che somma a 1.

---

## ⏱️ Min 30-60

Ecco degli appunti universitari dettagliati basati sulla lezione video sulle **Reti Neurali Convoluzionali (CNN)** e la loro implementazione pratica.

---

# Appunti di Deep Learning: Architettura e Implementazione di CNN

## 1. Fondamenti dell'Architettura CNN
Le Reti Neurali Convoluzionali (CNN) sono progettate per elaborare dati con struttura a griglia, come le immagini. L'obiettivo principale è estrarre caratteristiche (features) sempre più complesse riducendo al contempo la dimensione spaziale dei dati per gestire la complessità computazionale.

### Il Trade-off tra Risoluzione e Canali
*   **Aumento dei Canali:** Ogni strato convoluzionale applica molteplici filtri (kernel), aumentando il numero di canali (es. da 1 canale MNIST a 32 o 64 canali di feature). Questo permette alla rete di apprendere diverse tipologie di pattern (linee verticali, orizzontali, texture, ecc.).
*   **Riduzione Spaziale (Downsampling):** Per compensare l'esplosione dei dati dovuta all'aumento dei canali, si riduce la risoluzione spaziale dell'immagine. È più vantaggioso avere molteplici caratteristiche a bassa risoluzione che poche caratteristiche ad alta risoluzione.

## 2. Operazioni Core e Layer
L'architettura tipica segue il flusso: **Input -> [Convoluzione -> Attivazione -> Pooling] x N -> Fully Connected -> Output.**

### Convoluzione (`nn.Conv`)
*   Utilizza un **Kernel** (es. 3x3) che scorre sull'immagine.
*   **Parametri chiave:** Numero di features (canali in uscita) e dimensione del kernel. Questi sono iperparametri fondamentali da ottimizzare.

### Pooling (`nn.avg_pool` / `nn.max_pool`)
*   **Scopo:** Ridurre le dimensioni della matrice (downsampling) mantenendo le informazioni salienti.
*   **Average Pooling:** Calcola la media dei pixel in una finestra specifica (es. 2x2).
*   **Max Pooling:** Seleziona il valore massimo.
*   **Stride:** Indica il "salto" compiuto dalla finestra di pooling. Uno stride di 2 su una finestra 2x2 dimezza le dimensioni spaziali.

### Flattening e Dense Layer
*   Prima di passare ai layer densi (Fully Connected), i dati multidimensionali devono essere "appiattiti" in un vettore monodimensionale tramite un'operazione di **reshape**.
*   **Dense Layer (`nn.Dense`):** Esegue la classica moltiplicazione matriciale $W \cdot x + b$. L'ultimo layer denso deve avere un numero di output pari al numero di classi (es. 10 per MNIST).

## 3. Implementazione con Flax (JAX)
Nel video viene analizzata una classe Python che eredita da `nn.Module`:

```python
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # ... ulteriori strati ...
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=10)(x) # Output layer (Logits)
        return x
```

## 4. Stabilità Numerica e Softmax
Un punto critico riguarda la gestione dell'output finale per la classificazione multiclasse.

*   **Logits:** L'output dell'ultimo layer lineare può variare tra $-\infty$ e $+\infty$.
*   **Softmax:** Trasforma i logits in probabilità (valori tra 0 e 1 che sommano a 1) usando l'esponenziale: $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$.
*   **Problema di Instabilità:** Calcolare esplicitamente l'esponenziale di numeri grandi può portare a errori di precisione nei numeri a virgola mobile (floating-point).
*   **Soluzione:** Utilizzare funzioni integrate come `softmax_cross_entropy` (dalla libreria **Optax**). Queste funzioni gestiscono internamente i logaritmi e gli esponenziali in modo numericamente stabile (es. log-sum-exp trick).

## 5. Analisi del Modello e Parametri
L'uso del metodo `.tabulate()` permette di visualizzare il sommario della rete:
*   **Shape degli Input/Output:** Mostra come le dimensioni spaziali si riducano (es. da 28x28 a 14x14) mentre i canali aumentano.
*   **Conteggio Parametri:** La maggior parte dei parametri risiede solitamente nei layer **Dense** dopo il flattening, poiché connettono ogni feature estratta a ogni neurone del layer successivo. Il pooling è quindi essenziale per evitare un numero eccessivo di parametri che saturerebbero la memoria della GPU.

## 6. Logica di Training e Metriche
Vengono definite funzioni di utilità per il ciclo di addestramento:

### Calcolo delle Metriche (`compute_metrics`)
*   Calcola la **Loss** (Cross-Entropy).
*   Calcola l'**Accuracy** confrontando l'indice del valore massimo dei logits (`argmax`) con le etichette reali.

### Gradient Update (`train_step`)
*   Viene utilizzata `jax.value_and_grad` per ottenere simultaneamente il valore della loss e i gradienti, ottimizzando le prestazioni.
*   **`has_aux=True`:** Necessario quando la funzione di perdita restituisce valori ausiliari (come i logits) oltre allo scalare della loss.
*   **Optax:** Gestisce l'aggiornamento dei parametri (`state.apply_gradients`) includendo logiche complesse come il momentum o tassi di apprendimento adattivi.

---
*Note: La programmazione orientata agli oggetti in questo contesto serve a trasformare una definizione astratta (Classe) in un'istanza concreta del modello su cui eseguire il calcolo.*

---

## ⏱️ Min 60-90

Ecco degli appunti universitari dettagliati basati sul video riguardante l'addestramento di una rete neurale convoluzionale (CNN) utilizzando le librerie **JAX**, **Flax** e **Optax**.

---

# Appunti: Addestramento e Valutazione di una CNN con JAX/Flax

## 1. Funzione di Valutazione del Modello (`eval_model`)
La funzione `eval_model` viene utilizzata per calcolare le prestazioni del modello su un determinato dataset (tipicamente di validazione o test) senza aggiornare i parametri.

*   **Input**:
    *   `state`: L'oggetto `TrainState` che contiene i parametri correnti e la funzione di applicazione del modello.
    *   `dataset`: Un dizionario contenente le chiavi `"image"` (input $X$) e `"label"` (target $Y$).
*   **Logica**:
    1.  Esegue il forward pass tramite `state.apply_fn` per ottenere i **logits**.
    2.  Calcola le metriche (loss e accuracy) richiamando una funzione di utilità `compute_metrics`.
    3.  Restituisce le metriche calcolate per l'intero dataset.

## 2. Definizione dell'Epoca di Addestramento (`train_epoch`)
Questa funzione gestisce l'iterazione completa su tutto il dataset di addestramento per una singola epoca.

### A. Suddivisione in Mini-Batch
L'addestramento non avviene sull'intero dataset contemporaneamente per motivi di memoria ed efficienza computazionale (Stochastic Gradient Descent).
*   Viene calcolato il numero di step per epoca: `step_per_epoch = train_ds_size // batch_size`.

### B. Shuffling (Permutazione)
Per garantire che il modello non impari l'ordine dei campioni, i dati vengono rimescolati all'inizio di ogni epoca.
*   Si utilizza `jax.random.permutation` per generare un ordine casuale di indici basato su una chiave PRNG (Pseudo-Random Number Generator).

### C. Aggregazione delle Metriche
Le metriche di addestramento (loss e accuracy) vengono raccolte per ogni batch in una lista `batch_metrics`. Invece di ricalcolare la loss sull'intero dataset a fine epoca (operazione costosa), si effettua una **media delle statistiche dei batch**.
*   **Vantaggio**: Riduzione del costo computazionale.
*   **Svantaggio**: È una stima, poiché i parametri cambiano leggermente tra un batch e l'altro.

## 3. Preparazione dei Dati: Split Train-Validation
È prassi standard dividere i dati originali per monitorare l'overfitting e ottimizzare gli iperparametri.
*   **Split Ratio**: 80% Training, 20% Validation.
*   **Trasferimento su GPU**: JAX opera su array propri. Mentre NumPy risiede su CPU, gli array JAX (`jnp.array`) vengono automaticamente allocati sulla GPU (se disponibile) per accelerare i calcoli matriciali.

## 4. Inizializzazione del Modello e dell'Ottimizzatore

### Iperparametri definiti:
*   `num_epochs`: 10
*   `batch_size`: 64
*   `learning_rate`: 0.001

### Componenti di Flax/Optax:
1.  **Inizializzazione Parametri**: `cnn.init` genera i pesi iniziali (es. usando la distribuzione di Xavier/Glorot) partendo da un input di esempio (`jnp.zeros`) per definire gli shape delle matrici.
2.  **Ottimizzatore Adam**: Viene utilizzato l'ottimizzatore **Adam** (`optax.adam`).
    *   *Nota tecnica*: Adam combina i concetti di **Momentum** (accumulo dei gradienti passati) e **RMSProp** (scaling del learning rate basato sulla magnitudo dei gradienti), rendendo l'addestramento più stabile.
3.  **TrainState**: È una classe di Flax che raggruppa i parametri del modello, lo stato dell'ottimizzatore e la funzione di forward pass, facilitando la gestione degli aggiornamenti.

## 5. Loop di Addestramento Principale
Il ciclo itera per il numero di epoche specificato:
1.  **Aggiornamento PRNG**: Si utilizza `jax.random.split` per generare una nuova chiave casuale a ogni epoca, garantendo che le permutazioni siano diverse.
2.  **Chiamata a `train_epoch`**: Aggiorna lo `state` e restituisce le medie delle metriche di addestramento.
3.  **Valutazione**: Viene chiamata `eval_model` sul dataset di validazione.
4.  **Logging**: I risultati vengono stampati a video.
    *   *Dettaglio di formattazione*: L'uso di f-string come `f"{loss:.4e}"` permette di visualizzare la loss in notazione scientifica con 4 decimali, rendendo la tabella dei log facilmente leggibile.

## 6. Analisi dei Risultati
Nel video, dopo 10 epoche, si osservano i seguenti risultati:
*   **Training Accuracy**: > 99%
*   **Validation Accuracy**: ~ 98%

Questa piccola differenza (gap) tra addestramento e validazione indica che il modello ha generalizzato molto bene senza incorrere in un overfitting significativo. Il processo di conversione dei dati in `float32` assicura la compatibilità con i requisiti di calcolo ad alte prestazioni della GPU.

---

## ⏱️ Min 90-120

Ecco degli appunti universitari dettagliati basati sul seminario tecnico riguardante il Deep Learning, le Reti Neurali Convoluzionali (CNN) e la sicurezza dei modelli (Attacchi Avversari).

---

# Appunti del Corso: Deep Learning e Visione Artificiale
## Argomento: Ottimizzazione, Valutazione e Vulnerabilità delle CNN

### 1. Monitoraggio e Analisi dell'Addestramento
Durante la fase di addestramento di una CNN sul dataset MNIST (cifre scritte a mano), è fondamentale monitorare l'evoluzione della **Loss** (funzione di perdita) e della **Accuracy** (accuratezza) sia sul set di addestramento che su quello di validazione.

*   **Convergenza**: Nel video si osserva che dopo 10 epoche (da `0000` a `0009`), l'accuratezza di validazione raggiunge circa il **98.05%**. Il punto di minima perdita e massima accuratezza si attesta verso l'ultima epoca monitorata.
*   **Efficienza computazionale**: Una CNN moderna è estremamente efficiente rispetto a una rete densa (Fully Connected). Con soli 10 passaggi (epoche) e pochi secondi di calcolo, è possibile processare migliaia di immagini con un'accuratezza elevatissima.
*   **Confronto teorico**: Una rete neurale densa (feed-forward) richiederebbe molti più parametri e un costo computazionale significativamente superiore per raggiungere lo stesso livello di precisione su dati spaziali come le immagini.

### 2. Valutazione (Testing) e Previsione Probabilistica
Una volta completato l'addestramento, il modello viene valutato su un **Test Set** mai visto prima per verificarne la capacità di generalizzazione.

*   **Metriche di Test**: Nel caso analizzato, l'accuratezza di test è del **98.39%**, leggermente inferiore a quella di validazione (98.53% in un punto precedente), il che è fisiologico poiché gli iperparametri vengono spesso "tirati" sul validation set.
*   **Interpretazione probabilistica**: L'output finale della rete non è una singola cifra, ma una distribuzione di probabilità (solitamente ottenuta tramite funzione *Softmax*).
    *   Esempio: In un'immagine della cifra "5", la rete potrebbe essere molto sicura (istogramma alto sul 5), ma mostrare una piccola incertezza residua verso il "6" a causa di somiglianze morfologiche.
*   **Visualizzazione**: Gli istogrammi delle probabilità per ogni classe (0-9) permettono di comprendere quanto il classificatore sia "sicuro" della propria scelta.

### 3. Attacchi Avversari (Adversarial Attacks)
Un attacco avversario consiste nel modificare in modo quasi impercettibile un'immagine di input con l'obiettivo di indurre il classificatore in errore.

#### A. Il Metodo FGSM (Fast Gradient Sign Method)
L'algoritmo principale discusso è l'**FGSM**. La logica è opposta a quella dell'addestramento standard:
1.  **Training standard**: Si calcola il gradiente della funzione di perdita rispetto ai *parametri* ($\theta$) per minimizzare l'errore.
2.  **Attacco avversario**: Si calcola il gradiente della funzione di perdita rispetto all'*input* ($x$, l'immagine) per **massimizzare** l'errore.

**Formula euristica**:
$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \text{Loss})$$

*   **Epsilon ($\epsilon$)**: Rappresenta l'entità della perturbazione. Deve essere abbastanza piccola da non essere notata dall'occhio umano, ma sufficiente a spostare l'immagine oltre il confine di decisione della rete.
*   **Funzione Sign**: Si utilizza il segno del gradiente per garantire una magnitudo uniforme del cambiamento su tutta l'immagine, evitando che gradienti molto piccoli rendano nullo l'attacco.

#### B. Casi di Studio
*   **Esempio Classico**: Un'immagine di un panda, con l'aggiunta di un rumore calcolato (perturbazione dello 0.007), viene classificata dalla rete come un "gibbone" con una confidenza del 99.3%, pur rimanendo identica a un panda per un umano.
*   **Applicazione su MNIST**: Nel video, una cifra "4" viene manipolata fino a essere riconosciuta come un "9". Una cifra "6" viene vista come uno "0".

### 4. Estrazione delle Caratteristiche (Feature Extraction)
Per comprendere cosa "vede" effettivamente la rete, è possibile visualizzare l'output dei singoli layer convoluzionali.

*   **Layer Convoluzionali**: I primi strati della rete fungono da filtri che estraggono caratteristiche di basso livello come bordi, angoli o curvature.
*   **Struttura dei Dati**: In JAX/Flax, i parametri sono gestiti come dizionari o strutture ad albero (*pytrees*). Per visualizzare l'output del primo layer, bisogna isolare i parametri specifici (`Conv_0`) e applicare la funzione di attivazione (spesso *ReLU*).
*   **Interpretazione**: Visualizzando queste immagini intermedie, si nota come la rete enfatizzi determinati contrasti o contorni della cifra originale, trasformando l'immagine in una serie di mappe di attivazione che verranno poi interpretate dagli strati successivi.

---
**Terminologia tecnica chiave:**
*   **Hyperparameter Tuning**: Ottimizzazione dei parametri esterni al modello (es. learning rate).
*   **Overfitting**: Rischio in cui il modello impara troppo bene il training set ma fallisce sui nuovi dati (evitabile monitorando la validation loss).
*   **One-hot Encoding**: Metodo per rappresentare categorie (0-9) come vettori binari.
*   **Clipping**: Operazione per assicurarsi che i valori dei pixel rimangano nell'intervallo valido [0, 1] dopo l'attacco.

---

## ⏱️ Min 120-150

Certamente! Ecco degli appunti universitari dettagliati basati sul segmento video fornito:

# Visualizzazione delle Feature Map in una CNN (Convolutional Neural Network)

Il video illustra come estrarre e visualizzare l'output del primo strato convoluzionale di una rete neurale addestrata per il riconoscimento di cifre scritte a mano (dataset MNIST). L'obiettivo è comprendere come i diversi filtri (kernel) apprendano a estrarre caratteristiche specifiche dall'immagine di input.

## 1. Analisi del Codice Sorgente

Il processo viene implementato in Python, utilizzando probabilmente la libreria **Flax/JAX** (evinto dall'uso esplicito di `params` e `apply`).

### Definizione della Funzione di Estrazione
```python
def get_first_layer_output(params, x):
    # Definizione dello strato convoluzionale (32 filtri, kernel 3x3)
    conv_layer = nn.Conv(features=32, kernel_size=(3, 3), name="Conv_0")
    
    # Applicazione dello strato utilizzando i parametri appresi dal modello
    conv_output = conv_layer.apply({'params': params['params']['Conv_0']}, x)
    
    # Applicazione della funzione di attivazione ReLU
    return nn.relu(conv_output)
```
*   **Strato Convoluzionale:** Definito con 32 filtri di dimensione 3x3.
*   **ReLU (Rectified Linear Unit):** Utilizzata per introdurre non linearità, azzerando i valori negativi e mantenendo quelli positivi.

### Preparazione dei Dati e Inferenza
```python
idx = 0
# Estrazione di un'immagine di test
x = test_ds["image"][idx : idx + 1]

# Ottenimento dell'output (feature maps)
output_conv0 = get_first_layer_output(state.params, x)

# Controllo della forma dell'output
# Output: (1, 28, 28, 32) -> [Batch, Altezza, Larghezza, Canali/Filtri]
print(output_conv0.shape)
```
La dimensione dell'output conferma che per una singola immagine di input (28x28), sono state generate **32 feature map** distinte, una per ogni filtro dello strato.

## 2. Visualizzazione e Interpretazione dei Risultati

Per visualizzare cosa ha "visto" ogni filtro, si itera attraverso i 32 canali di output:

```python
for i in range(output_conv0.shape[-1]):
    plt.figure()
    # Visualizzazione dell'i-esima feature map in scala di grigi
    plt.imshow(output_conv0[0, :, :, i], cmap="gray")
```

### Analisi delle Feature Map (Esempio: Cifra "7")
Dalla visualizzazione delle feature map relative alla cifra "7", si possono trarre le seguenti conclusioni:

*   **Estrazione di Caratteristiche Specifiche:** Ogni filtro si specializza nel riconoscere un particolare pattern geometrico.
*   **Esempi di Filtri osservati nel video:**
    *   **Rilevatore di Linee Oblique/Curve:** Alcuni filtri mostrano un'alta attivazione (bianco) sulle linee diagonali del "7", mentre la linea orizzontale superiore appare quasi nera (bassa attivazione).
    *   **Rilevatore di Linee Orizzontali:** Altri filtri mostrano un'attivazione marcata sulla barra superiore del "7", ignorando la parte diagonale.
    *   **Rilevatore di Contorni (Edge Detection):** Alcune mappe evidenziano i bordi esterni della cifra, agendo in modo simile a un filtro di sharpening o di esaltazione della nitidezza.

## 3. Concetti Chiave

1.  **Feature Map (Mappa delle Caratteristiche):** È il risultato dell'applicazione di un filtro convoluzionale. Rappresenta la presenza e la posizione di una determinata caratteristica nell'immagine di input.
2.  **Apprendimento dei Filtri:** I pesi dei kernel non sono predefiniti manualmente, ma vengono appresi dalla rete durante la fase di training. La rete "capisce" autonomamente che per distinguere un "7" da un "5" è utile estrarre linee orizzontali e diagonali.
3.  **Gerarchia delle Caratteristiche:** Sebbene il video si concentri sul primo strato (che estrae caratteristiche semplici come bordi e linee), negli strati successivi di una CNN queste caratteristiche vengono combinate per riconoscere forme sempre più complesse (angoli, cerchi, parti di oggetti).

---
**Nota finale:** La visualizzazione conferma che i primi strati di una rete convoluzionale funzionano in modo analogo ai filtri classici di elaborazione delle immagini, ma con la capacità critica di adattarsi specificamente ai dati del problema.

---

