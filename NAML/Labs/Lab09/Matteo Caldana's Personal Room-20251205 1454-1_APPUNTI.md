# 📝 Appunti: Matteo Caldana's Personal Room-20251205 1454-1
**Modello:** gemini-3-flash-preview | **Data:** 22/01/2026

---

## ⏱️ Min 0-30

Ecco dei dettagliati appunti universitari basati sul segmento video fornito, che copre i concetti fondamentali delle convoluzioni e della Trasformata di Fourier Veloce (FFT).

---

# Appunti del Laboratorio: Convoluzioni e Trasformata di Fourier (FFT)

## 1. Introduzione alle Convoluzioni
La convoluzione è un'operazione matematica fondamentale nell'elaborazione dei segnali e dei dati strutturati (vettori, matrici, tensori). Nel contesto del deep learning, è lo strumento principale per le **Convolutional Neural Networks (CNN)**, utilizzate per compiti come la classificazione del dataset MNIST.

### Meccanica Visiva della Convoluzione
Un'operazione di convoluzione coinvolge tre componenti principali:
1.  **Kernel (Filtro):** Un piccolo set di pesi (rappresentato in blu nel video) che definisce la caratteristica da estrarre.
2.  **Segnale (Input):** I dati grezzi (rappresentati in rosso), come un segnale audio 1D o un'immagine 2D.
3.  **Risultato (Feature Map):** L'output ottenuto applicando il kernel al segnale.

**Processo:**
*   Il kernel "scorre" lungo il segnale.
*   In ogni posizione, viene calcolato il **prodotto elemento per elemento** tra il kernel e la porzione corrispondente del segnale.
*   I prodotti vengono **sommati** per produrre un singolo valore nel segnale di uscita.
*   *Esempio:* Una convoluzione con un kernel costante agisce come una **media mobile**, attenuando le punte e smussando il segnale.

---

## 2. Il Concetto di Padding
Il padding determina come vengono gestiti i bordi del segnale durante la convoluzione. Esistono tre tipologie principali:

1.  **Valid Padding:** Non viene aggiunto alcun padding. Il kernel viene applicato solo dove si sovrappone completamente al segnale. Il risultato è **più piccolo** dell'input.
2.  **Same Padding:** Viene aggiunto un padding sufficiente (solitamente zeri) affinché l'output abbia la **stessa dimensione** dell'input.
3.  **Full Padding:** Il kernel scorre dal primo punto di contatto parziale all'ultimo. Il risultato è **più grande** dell'input, testando il kernel in ogni posizione possibile.

---

## 3. Rappresentazioni Matriciali
Le convoluzioni possono essere espresse come moltiplicazioni tra matrici utilizzando strutture speciali:

*   **Matrice di Toeplitz:** Una matrice in cui ogni diagonale discendente da sinistra a destra è costante. In Python, si può generare con `scipy.linalg.toeplitz`. È usata per mappare l'operazione di convoluzione 1D in una forma $y = Ax$.
*   **Matrice Circolante:** Un tipo speciale di matrice di Toeplitz quadrata in cui ogni riga è uno spostamento ciclico della riga precedente.

---

## 4. La Trasformata di Fourier (FFT)

### Teoria e Differenze Tecniche
*   **DFT (Discrete Fourier Transform):** L'operatore matematico che trasforma un segnale dal dominio spaziale/temporale al **dominio della frequenza**.
*   **FFT (Fast Fourier Transform):** L'algoritmo efficiente per calcolare la DFT. Ha una complessità computazionale di $O(N \log N)$, un miglioramento massiccio rispetto al $O(N^2)$ della DFT ingenua.

L'FFT è essenzialmente un **cambio di base**. Invece di descrivere il segnale tramite la sua ampiezza nel tempo, lo descrive come una combinazione di seni e cosini a diverse frequenze.

### Implementazione in NumPy (`np.fft`)
Funzioni standard utilizzate:
*   `fft(a)` / `ifft(a)`: Calcola la FFT 1D e la sua inversa.
*   `fftfreq(n, d)`: Restituisce i centri delle frequenze per i componenti della FFT.
*   `fftshift(x)`: Sposta la componente a frequenza zero al centro dello spettro (utile per la visualizzazione).
*   `fft2(a)`: FFT per segnali bidimensionali.

### Analisi Pratica di un Segnale 1D
Prendiamo un segnale composto dalla somma di due onde sinusoidali (es. 5Hz e 13.5Hz):
1.  **Dominio del Tempo:** Il segnale appare complesso e difficile da interpretare visivamente.
2.  **Applicazione FFT:** Il risultato è un vettore di **numeri complessi**.
3.  **Magnitudo (Valore Assoluto):** Plottando il valore assoluto dei coefficienti FFT rispetto alle frequenze calcolate, appaiono dei picchi netti in corrispondenza delle frequenze costituenti (5 e 13.5).
4.  **Ricostruzione:** Utilizzando l'Inversa della FFT (`ifft`), possiamo tornare al segnale originale.
    *   *Nota tecnica:* A causa della precisione numerica, l'output di `ifft` avrà una parte immaginaria minuscola (prossima all'**epsilon di macchina**, circa $10^{-16}$). Per la ricostruzione visiva si utilizza solo la parte reale.

---

## 5. FFT in 2D
Per i segnali 2D (immagini), la FFT rivela la periodicità spaziale.
*   Un segnale con strisce verticali produrrà dei "poli" (punti luminosi) sull'asse delle frequenze corrispondente nella FFT 2D.
*   Segnali circolari (onde concentriche) producono cerchi nello spazio delle frequenze.

---

## 6. Esercitazione Pratica: Implementazione della Convoluzione
L'obiettivo finale del laboratorio è implementare la convoluzione 1D in due modi diversi:
1.  **Tramite Matrice di Toeplitz:** Definendo una matrice $K$ associata al kernel $k$ e calcolando il prodotto $k * v = Kv$.
2.  **Tramite Definizione Diretta:** Utilizzando due cicli `for` annidati per calcolare la somma dei prodotti (approccio "sum of products").

**Importanza dei Kernel nel Deep Learning:**
Nelle reti neurali convoluzionali, i kernel non sono predefiniti dall'uomo, ma vengono **appresi automaticamente** tramite la discesa del gradiente. La rete scopre quali filtri (es. rilevatori di bordi verticali, texture) sono più utili per il compito specifico.

---

## ⏱️ Min 30-60

Ecco degli appunti universitari dettagliati basati sulla lezione video sulle convoluzioni 1D in Python.

---

# Note di Lezione: Convoluzioni 1D - Teoria e Implementazione in Python

## 1. Introduzione alla Convoluzione 1D
La convoluzione è un'operazione fondamentale nell'elaborazione dei segnali e nell'apprendimento automatico. Formalmente, per un segnale $v$ e un kernel $k$, la convoluzione può essere espressa come un'operazione lineare che può essere rappresentata tramite un prodotto matrice-vettore:
$$k * v = K v$$
dove $K$ è una speciale matrice strutturata chiamata **Matrice di Toeplitz**.

---

## 2. Metodo 1: Matrice di Toeplitz
In questo approccio, la convoluzione viene vista come una trasformazione lineare.

### 2.1 Proprietà della Matrice di Toeplitz
*   Una matrice di Toeplitz ha le diagonali costanti.
*   Ogni riga della matrice rappresenta il kernel $k$ traslato di una posizione rispetto alla riga precedente.
*   Questo permette di calcolare l'intero segnale convoluto attraverso un singolo prodotto tra matrici.

### 2.2 Implementazione in Python
Per implementare questo metodo, utilizziamo `scipy.linalg.toeplitz`.

```python
import numpy as np
from scipy import linalg

# Definizione del segnale v e del kernel k (es. moving average)
# k_padded viene creato per gestire le dimensioni corrette
length_out = len(k) + len(v) - 1
k_padded = np.zeros(length_out)
k_padded[:len(k)] = k

# Creazione della matrice di Toeplitz
# La prima colonna inizia con il kernel, seguita da zeri
first_col = k_padded
first_row = np.zeros(len(v))
first_row[0] = k_padded[0] # La prima cella deve coincidere

K = linalg.toeplitz(first_col, first_row)

# Calcolo della convoluzione
v_conv1 = K @ v
```

---

## 3. Varianti del Kernel e Applicazioni Fisiche
La lezione esplora come diverse forme di kernel $k$ influenzino il segnale di output $v$.

### 3.1 Kernel Media Mobile (Moving Average)
*   **Definizione:** Un vettore di valori costanti (es. `np.ones(10) / 10`).
*   **Effetto:** "Smussa" (smooth) il segnale, riducendo il rumore ad alta frequenza ma sfocando i fronti di salita/discesa.

### 3.2 Kernel Gaussiano
*   **Implementazione:** `scipy.signal.windows.gaussian(size, std)`.
*   **Nota Tecnica:** Nelle versioni recenti di SciPy, la funzione `gaussian` è stata spostata nel modulo `windows`.
*   **Effetto:** Simile alla media mobile ma dà più importanza ai valori centrali. Produce una transizione molto più dolce e graduale rispetto al kernel costante.

### 3.3 Kernel a Differenze Finite (Derivata)
*   **Definizione:** Un kernel con valori negativi e positivi, ad esempio `[-1, 2, -1]`.
*   **Effetto:** Agisce come un operatore di derivata prima o seconda.
*   **Risultato:** In corrispondenza dei gradini del segnale, produce dei "picchi" (impulsi), mentre sulle parti costanti il risultato è zero. È fondamentale per il **edge detection** (rilevamento dei contorni).

---

## 4. Metodo 2: Definizione Diretta (Somma di Prodotti)
Questo metodo implementa la formula matematica classica della convoluzione discreta senza utilizzare strutture matriciali esplicite.

### 4.1 Formula Matematica
L'elemento $i$-esimo del segnale convoluto è dato da:
$$(k * v)_i = \sum_{j=0}^{n_k-1} k_{n_k-1-j} \cdot v_{i+j}$$

### 4.2 Implementazione con Cicli For (Nested Loops)
Sebbene meno efficiente del calcolo vettorializzato in Python, è utile per comprendere l'algoritmo alla base.

```python
l_out = len(v) - len(k) + 1 # Lunghezza senza boundary layer
v_conv2 = np.zeros(l_out)

for i in range(l_out):
    for j in range(len(k)):
        # Applica la formula: kernel invertito moltiplicato per il segnale
        v_conv2[i] += k[len(k) - 1 - j] * v[i + j]
```

---

## 5. Analisi delle Differenze: Padding e Bordi
Durante la lezione viene evidenziata una differenza cruciale tra i due metodi:

1.  **Metodo Toeplitz (con Padding):** Calcola la convoluzione considerando anche i momenti in cui il kernel "entra" ed "esce" dal segnale. L'output è tipicamente più lungo del segnale originale.
2.  **Definizione Diretta (senza Padding):** Calcola il risultato solo dove il kernel è completamente sovrapposto al segnale (spesso chiamata modalità "valid"). L'output è più corto e non presenta il "boundary layer".

## 6. Considerazioni sulle Prestazioni
*   **Vettorializzazione:** L'uso di matrici di Toeplitz o funzioni predefinite come `np.convolve` è ordini di grandezza più veloce dei cicli `for` annidati in Python.
*   **Complessità:** Per segnali molto grandi, la convoluzione viene solitamente calcolata nel dominio della frequenza utilizzando la **FFT (Fast Fourier Transform)**, dove la convoluzione diventa una semplice moltiplicazione punto a punto.

---
*Fine degli appunti.*

---

## ⏱️ Min 60-90

Ecco degli appunti universitari dettagliati basati sul segmento video fornito, che tratta l'implementazione e l'ottimizzazione delle convoluzioni 1D e 2D in ambiente Python.

---

# Analisi Numerica per il Machine Learning: Convoluzioni 1D e 2D

## 1. Ottimizzazione della Convoluzione 1D via Vettorizzazione
L'approccio iniziale basato su cicli `for` annidati è computazionalmente inefficiente in Python. Per migliorare le prestazioni, si ricorre alla **vettorizzazione** sfruttando la libreria NumPy.

### Sostituzione dei cicli con `np.sum`
Invece di iterare manualmente su ogni elemento del kernel e del segnale, possiamo calcolare ogni elemento del segnale di output ($v_{conv}$) come una somma vettoriale.
```python
# Versione ottimizzata con np.sum
for i in range(l_out):
    v_conv2[i] = np.sum(np.flip(k) * v[i : i + len(k)])
```
**Nuance tecnica:** È necessario invertire il kernel ($k$) utilizzando `np.flip(k)` per rispettare la definizione matematica di convoluzione, oppure operare su un subset del segnale $v$ della stessa dimensione di $k$.

### Rappresentazione tramite Prodotto Scalare (`np.dot`)
Una forma ancora più compatta e potenzialmente più veloce consiste nell'utilizzare il prodotto punto (o scalare), che combina intrinsecamente la moltiplicazione elemento per elemento e la somma:
```python
v_conv2[i] = np.dot(np.flip(k), v[i : i + len(k)])
```

---

## 2. Convoluzione nel Dominio della Frequenza (DFT)
Un metodo fondamentale per calcolare la convoluzione sfrutta il **Teorema della Convoluzione**, il quale afferma che la convoluzione nel dominio del tempo/spazio equivale alla moltiplicazione puntuale (prodotto di Hadamard) nel dominio della frequenza.

### Il Teorema
Dati un segnale $v$ e un kernel $k$, la loro convoluzione $\mathbf{v} * \mathbf{k}$ è data da:
$$\mathbf{v} * \mathbf{k} = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{v}) \odot \mathcal{F}(\mathbf{k}))$$
Dove:
*   $\mathcal{F}$ è la Trasformata di Fourier Discreta (DFT).
*   $\odot$ rappresenta il prodotto elemento per elemento (Hadamard).

### Implementazione e Zero-Padding
Per poter moltiplicare le trasformate, i vettori devono avere la stessa dimensione. NumPy permette di gestire automaticamente lo **zero-padding** passando la lunghezza desiderata come secondo argomento alla funzione `fft`.

```python
# Procedura DFT
v_fft = np.fft.fft(v)
k_fft = np.fft.fft(k, n=len(v)) # Padding automatico di k alla lunghezza di v

# Prodotto nel dominio della frequenza
vk_fft = v_fft * k_fft

# Ritorno al dominio spaziale e rimozione parte immaginaria residua
v_conv3 = np.real(np.fft.ifft(vk_fft))
```
**Nota:** L'uso di `np.real` è necessario perché, a causa di errori di arrotondamento numerico, il risultato della trasformata inversa potrebbe presentare una componente immaginaria infinitesimale anche se il segnale atteso è puramente reale.

---

## 3. Utilizzo di `scipy.signal.convolve`
Per scopi applicativi, si utilizzano le funzioni di libreria ottimizzate. La funzione `scipy.signal.convolve` offre tre modalità principali (`mode`) che determinano la dimensione dell'output:
1.  **`full`**: Restituisce la convoluzione completa (dimensione $N + M - 1$).
2.  **`valid`**: Restituisce solo le parti del segnale dove il kernel e il segnale si sovrappongono completamente senza bisogno di padding (dimensione $\max(M, N) - \min(M, N) + 1$).
3.  **`same`**: Restituisce un output della stessa dimensione del segnale in ingresso, centrato rispetto al risultato `full`.

---

## 4. Estensione alle Convoluzioni 2D (Image Processing)
Nelle convoluzioni 2D, il segnale è tipicamente un'immagine (una matrice di pixel).

### Pre-processing dell'immagine
*   **Caricamento**: Utilizzo di `imread` da `matplotlib.image`.
*   **Conversione in scala di grigi**: Poiché le immagini RGB sono tensori 3D (H, W, Canali), si può ottenere una versione in scala di grigi mediando i canali:
    ```python
    v = np.mean(imread(image_path), axis=2)
    ```

### Tipologie di Kernel 2D Comuni
Le convoluzioni 2D vengono utilizzate per applicare filtri alle immagini tramite specifici kernel (matrici):

1.  **Blur (Sfocatura)**:
    *   *Uniform Blur*: Una matrice di soli 1 normalizzata (es. `np.ones((10, 10)) / 100`).
    *   *Gaussian Blur*: Pesi decrescenti dal centro verso l'esterno per una sfocatura più naturale.
2.  **Edge Detection (Rilevamento Bordi)**: Kernel progettati per evidenziare brusche variazioni di intensità luminosa.
3.  **Sobel Operators**:
    *   *Sobel Verticale/Orizzontale*: Utilizzati per rilevare i gradienti lungo assi specifici.
4.  **Sharpen (Affilatura)**: Aumenta il contrasto dei dettagli e delle linee sottili.
5.  **Drunkard's Walk (Random)**: Utilizzo di numeri casuali nel kernel per effetti di distorsione stocastica.

---

## Riepilogo Termini Tecnici
*   **Vettorizzazione**: Tecnica di programmazione che sostituisce cicli espliciti con operazioni su interi array per sfruttare le ottimizzazioni di basso livello.
*   **Zero-Padding**: Aggiunta di zeri alle estremità di un segnale per raggiungere una lunghezza specifica (necessario per la DFT o per controllare la dimensione dell'output).
*   **Prodotto di Hadamard**: Moltiplicazione elemento per elemento tra due matrici/vettori della stessa dimensione.
*   **Dominio della Frequenza**: Rappresentazione di un segnale in termini di ampiezze e fasi delle sue componenti sinusoidali.

---

## ⏱️ Min 90-120

Ecco degli appunti universitari dettagliati basati sul video riguardante l'elaborazione dei segnali e le convoluzioni 2D in Python.

---

# Appunti di Elaborazione Digitale delle Immagini: Convoluzione e Filtri

## 1. Introduzione ai Kernel di Convoluzione
Il video introduce il concetto di **kernel** (o maschera) come una piccola matrice utilizzata per applicare effetti a un segnale 2D (immagine).

### Caratteristiche del Blurring Kernel (Sfocatura)
Prendendo come esempio `kernel_blur1`, si osservano le seguenti proprietà:
*   **Struttura**: Possiede un valore centrale elevato (0.25) circondato da valori più piccoli (0.125, 0.0625).
*   **Meccanismo**: Calcola una media pesata dell'intorno di ogni pixel. Maggiore è il peso del centro, maggiore è l'importanza della posizione attuale del pixel rispetto ai vicini.
*   **Normalizzazione**: La somma di tutti gli elementi del kernel deve essere pari a **1**. Questo assicura che l'intensità luminosa globale dell'immagine rimanga invariata dopo l'operazione (evitando che l'immagine diventi troppo chiara o troppo scura).
*   **Effetto Visivo**: Poiché "mescola" le informazioni di un intorno (neighborhood), l'effetto risultante è una perdita di dettaglio nitido, ovvero una sfocatura.

### Altri Kernel Menzionati (Snippet di Codice)
*   **Edge Detection**: `kernel_edge` (es. Sobel, Laplaciani) per identificare i bordi.
*   **Sharpening**: Per aumentare la nitidezza.
*   **Drunk Kernel**: Generato casualmente (`np.random.randn`), produce un rumore o una distorsione imprevedibile.

---

## 2. Convoluzione Diretta (Somma di Prodotti)

La convoluzione spaziale tra un segnale $v$ di dimensioni $M \times N$ e un kernel $k$ di dimensioni $m \times n$ è definita formalmente come:

### Definizione Matematica
Sia $v \in \mathbb{R}^{M,N}$ e $k \in \mathbb{R}^{m,n}$.
1.  **Dimensione dell'output**: Per una convoluzione senza "boundary layers" (padding), la dimensione è:
    *   $S_z = M - m + 1$
    *   $S_p = N - n + 1$
2.  **Kernel Reverse**: Per la convoluzione è necessario invertire il kernel:
    $$k^*_{a,b} = k_{m-1-a, n-1-b}$$
3.  **Equazione di Convoluzione**:
    $$(v * k)_{i,j} = \sum_{a=0}^{m-1} \sum_{b=0}^{n-1} k^*_{a,b} \cdot v_{i+a, j+b}$$

### Implementazione Algoritmica in Python
L'approccio "naïve" prevede **4 cicli for annidati**:
1.  Due cicli esterni per scorrere i pixel dell'immagine di output ($i, j$).
2.  Due cicli interni per scorrere gli elementi del kernel ($ki, kj$).

**Nota Tecnica**: In Python, l'uso di 4 cicli annidati è estremamente inefficiente a causa dell'overhead dell'interprete. Si preferisce la vettorializzazione tramite NumPy.

#### Ottimizzazione con NumPy
È possibile ridurre i cicli interni utilizzando le funzioni di slicing e somma di NumPy:
```python
# Versione ottimizzata (2 cicli anziché 4)
for i in range(S_out[0]):
    for j in range(S_out[1]):
        # np.flip(k) inverte il kernel su entrambi gli assi
        v_conv[i, j] = np.sum(np.flip(k) * v[i : i+m, j : j+n])
```
*   **`np.flip(k)`**: Con `axis=None` (default), inverte l'ordine degli elementi lungo tutti gli assi, realizzando il "kernel reversal" necessario.

---

## 3. Convoluzione tramite Trasformata Discreta di Fourier (DFT)

Il teorema della convoluzione afferma che la convoluzione nel dominio spaziale equivale alla moltiplicazione puntuale nel dominio della frequenza.

### Principio Teorico
$$\widehat{v * k} = \hat{v} \odot \hat{k}$$
Dove:
*   $\hat{a}$ è la DFT del segnale $a$.
*   $\odot$ indica il prodotto di Hadamard (prodotto elemento per elemento).

### Passaggi per l'implementazione (FFT2)
1.  **DFT del Segnale**: Calcolare la trasformata 2D dell'immagine: `v_fft = fft2(v)`.
2.  **DFT del Kernel con Padding**: È fondamentale che il kernel abbia la stessa dimensione dell'immagine per poter eseguire il prodotto di Hadamard. Si utilizza il parametro `shape` in `fft2`: `k_fft = fft2(k, v.shape)`.
3.  **Moltiplicazione**: `vk_fft = v_fft * k_fft`.
4.  **Inversione**: Tornare nel dominio spaziale con la IFFT 2D: `v_conv = np.real(ifft2(vk_fft))`.

**Vantaggio**: La complessità computazionale passa da $O(N^2 \cdot M^2)$ a $O(N \log N)$, rendendo questo metodo molto più veloce per kernel di grandi dimensioni.

---

## 4. Errori Comuni e Best Practices
*   **Assegnazione Indici**: Durante l'accumulo manuale della convoluzione, assicurarsi di assegnare il valore calcolato alla cella specifica `v_conv[i, j]` e non all'intera matrice, per evitare incrementi computazionali esponenziali e risultati errati.
*   **Documentazione NumPy**: Il video sottolinea l'importanza di leggere la documentazione (es. per `np.flip` e `np.fft.fft2`) per comprendere come vengono gestiti gli assi e il padding automatico.
*   **Visualizzazione**: Per visualizzare correttamente il risultato della FFT, viene spesso utilizzato `fftshift` per centrare le basse frequenze e la scala logaritmica (`np.log10(np.absolute(...))`) per gestire il range dinamico dei valori.

---

## ⏱️ Min 120-150

Ecco delle note universitarie dettagliate basate sul segmento video fornito, che tratta dell'elaborazione delle immagini tramite convoluzione, analisi nel dominio della frequenza (FFT) e implementazioni a basso livello con JAX.

---

# Appunti del Corso: Elaborazione delle Immagini e Calcolo Scientifico

## 1. Visualizzazione della Trasformata di Fourier (FFT)
Nell'analizzare le immagini nel dominio della frequenza, è fondamentale utilizzare una scala logaritmica per la visualizzazione delle ampiezze.

*   **Perché la scala logaritmica?** Le ampiezze delle frequenze possono variare di molti ordini di grandezza (es. da 0 a 1000). Usando una scala lineare, le differenze minori verrebbero "schiacciate", rendendo l'immagine quasi uniforme.
*   **Funzione Python:** Si utilizza `np.log10(np.absolute(v_fft))` per comprimere l'intervallo dinamico e rendere visibili i dettagli delle frequenze.

---

## 2. Analisi dei Kernel nel Dominio della Frequenza
L'effetto di un kernel (filtro) può essere meglio compreso osservando la sua FFT traslata (`fftshift`).

### Filtro Passa-Basso (Blurring)
*   **Comportamento Spaziale:** Un kernel di sfocatura (es. media pesata) ha valori più alti al centro.
*   **Comportamento in Frequenza:** La FFT mostra un valore elevato (giallo) al centro (basse frequenze) e valori bassi (blu) verso i bordi (alte frequenze).
*   **Effetto:** Rimuove le alte frequenze (bordi nitidi, rumore), producendo un'immagine più liscia.

### Filtro Passa-Alto (Edge Detection)
*   **Esempio:** Kernel Laplaciano (es. centro positivo, intorno negativo).
*   **Comportamento in Frequenza:** La FFT mostra valori bassi al centro e valori alti verso i bordi.
*   **Effetto:** Rimuove le basse frequenze (zone uniformi) e mantiene solo le variazioni repentine (bordi).

---

## 3. Filtri Direzionali e Compositi

### Filtri di Sobel (Orizzontale e Verticale)
Vengono utilizzati per estrarre bordi in direzioni specifiche.
*   **Sobel Orizzontale:** Il kernel ignora le variazioni sulla riga centrale. Nella FFT, si osservano due "poli" sull'asse Y. Esalta i dettagli orizzontali.
*   **Sobel Verticale:** Il kernel ha una colonna centrale di zeri. Nella FFT, i poli sono sull'asse X. Esalta i dettagli verticali (es. colonne di un edificio).

### Filtro di Sharpening (Nitidezza)
*   **Definizione:** Il kernel di nitidezza è essenzialmente una combinazione del filtro Laplaciano più la matrice identità (identificata dal valore centrale del kernel).
*   **Logica:** Si prende l'immagine originale e si aggiungono i bordi estratti dal Laplaciano.
*   **Risultato:** L'immagine appare più definita rispetto all'originale, con bordi più marcati.

---

## 4. Implementazione della Convoluzione in JAX (Livello LAX)
Il passaggio da librerie ad alto livello come `scipy.signal` a JAX (livello `lax`) richiede una gestione esplicita delle dimensioni dei tensori.

### Requisiti di Input (Tensori 4D)
JAX richiede che sia l'immagine che il kernel siano tensori a 4 dimensioni per gestire standard tipici delle reti neurali:
1.  **Immagine (v):** `(batch_size, altezza, larghezza, canali_in)`.
    *   In scala di grigi, `canali_in = 1`.
2.  **Kernel (k):** `(altezza_filtro, larghezza_filtro, canali_in, canali_out)`.

### Parametri Tecnici della Funzione `jax.lax.conv_general_dilated`
*   **Padding:** Per mantenere le dimensioni dell'output uguali all'input (`mode='same'`), il padding deve essere calcolato manualmente: `pad = (dimensione_kernel - 1) // 2`.
*   **Stride:** Indica il salto che il kernel compie durante la convoluzione. Uno stride di 1 (`window_strides=(1, 1)`) elabora ogni pixel.
*   **Dimension Numbers:** Una stringa che definisce l'ordine delle dimensioni (es. `"NHWC"` per l'input: Batch, Height, Width, Channel).
*   **Precisione:** JAX utilizza di default la precisione singola (`float32`), il che può introdurre piccoli errori di arrotondamento (ordine di $10^{-6}$) rispetto a Scipy.

### Workflow in JAX:
1.  Aggiunta di dimensioni fittizie tramite `None` o `np.expand_dims`.
2.  Calcolo del padding simmetrico.
3.  Esecuzione di `lax.conv_general_dilated`.
4.  Rimozione delle dimensioni extra per tornare a una matrice 2D visualizzabile.

---

## Glossario Tecnico
*   **FFT (Fast Fourier Transform):** Algoritmo per trasformare un segnale dal dominio spaziale a quello della frequenza.
*   **Kernel/Filtro:** Matrice di piccole dimensioni applicata a un'immagine tramite convoluzione per ottenere effetti specifici.
*   **Laplaciano:** Operatore differenziale del secondo ordine utilizzato per l'edge detection.
*   **Batch Size:** Numero di campioni elaborati simultaneamente (fondamentale nel Deep Learning).
*   **Stride:** Passo di scorrimento del filtro sull'immagine.

---

