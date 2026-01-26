# 📝 Appunti: Matteo Caldana's Personal Room-20251003 1412-1
**Modello:** gemini-3-flash-preview | **Data:** 22/01/2026

---

## ⏱️ Min 0-30

Ecco degli appunti universitari dettagliati basati sul segmento video, focalizzati sulla Decomposizione ai Valori Singolari (SVD), le sue applicazioni e l'implementazione pratica in ambiente Python.

---

# Appunti del Laboratorio: Decomposizione ai Valori Singolari (SVD) e Compressione

**Docente:** Matteo Caldana  
**Argomenti:** SVD per la compressione, Decomposizione ai Valori Singolari Randomizzata (rSVD), Implementazione in Python (NumPy/SciPy).

---

## 1. Revisione Teorica della SVD

La **Decomposizione ai Valori Singolari (SVD)** scompone una matrice $A \in \mathbb{R}^{m \times n}$ nel prodotto di tre matrici:

$$A = U \Sigma V^T$$

- **$U \in \mathbb{R}^{m \times m}$**: Matrice ortogonale i cui vettori colonna sono i vettori singolari sinistri. ($U U^T = I$).
- **$\Sigma \in \mathbb{R}^{m \times n}$**: Matrice pseudo-diagonale contenente i valori singolari $\sigma_i$ ordinati in modo decrescente.
- **$V \in \mathbb{R}^{n \times n}$**: Matrice ortogonale i cui vettori colonna sono i vettori singolari destri. ($V V^T = I$).

### 1.1 SVD per la Compressione dei Dati
Sfruttando il **teorema di Eckart-Young**, possiamo approssimare $A$ con una matrice di rango inferiore $k$ ($k < \text{rango}(A)$):

$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

- **Proprietà:** $A_k$ è la migliore approssimazione di rango $k$ di $A$ sia nella norma 2 che nella norma di Frobenius.
- **Efficienza di memorizzazione:** Invece di memorizzare $m \times n$ elementi, memorizziamo solo $(m + n + 1)k$ valori. Questo riduce la complessità dello storage da quadratica a lineare rispetto alle dimensioni della matrice (per un $k$ fisso).

---

## 2. Decomposizione ai Valori Singolari Randomizzata (rSVD)

Quando le dimensioni della matrice sono estremamente grandi, il calcolo della SVD completa è computazionalmente oneroso. La **rSVD** utilizza tecniche di algebra lineare randomizzata per trovare un'approssimazione accurata in modo molto più veloce.

### Fasi dell'Algoritmo rSVD:
1.  **Creazione di una Matrice di Sketch:** Si genera una matrice casuale $P \in \mathbb{R}^{n \times k}$.
2.  **Campionamento dello Spazio delle Colonne:** Si calcola $Z = AP$, proiettando $A$ in uno spazio di dimensione inferiore.
3.  **Base Ortonormale:** Si esegue la decomposizione QR su $Z$ ($QR = Z$) per ottenere una base ortonormale $Q$ per lo spazio delle colonne di $A$.
4.  **Proiezione:** Si proietta $A$ sulla base $Q$: $Y = Q^T A$. $Y$ è ora una matrice molto più piccola.
5.  **SVD su Matrice Ridotta:** Si calcola la SVD standard di $Y$: $U_Y \Sigma V^T = Y$.
6.  **Lifting (Ritorno allo Spazio Originale):** Si recupera la matrice $U$ originale calcolando $U = Q U_Y$.

---

## 3. Implementazione in Python (NumPy e SciPy)

### 3.1 Setup e Riproducibilità
Per garantire risultati consistenti in esperimenti che coinvolgono numeri casuali, è fondamentale impostare un **seed**.
```python
import numpy as np
from scipy import linalg as la

np.random.seed(0) # Garantisce la riproducibilità
A = np.random.rand(5, 4)
```

### 3.2 Funzioni SVD
Sia NumPy che SciPy offrono implementazioni quasi identiche tramite il sottomodulo `linalg`.

- **Full SVD:** Restituisce matrici $U$ e $V$ complete.
- **Thin SVD:** Restituisce solo le prime $k$ colonne/righe necessarie, risparmiando memoria.
```python
# Thin SVD
u, s, vt = np.linalg.svd(A, full_matrices=False)
```
**Note tecniche:**
- `s` viene restituito come un **vettore 1D** dei soli valori singolari, non come una matrice diagonale.
- `vt` corrisponde già alla matrice $V$ trasposta ($V^T$).

### 3.3 Ricostruzione della Matrice
Per ricostruire $A$ dai suoi componenti, è necessario convertire il vettore `s` in una matrice diagonale compatibile.
```python
# Utilizzando np.diag
A_reconstructed = u @ np.diag(s) @ vt

# Utilizzando la funzione specifica di SciPy per matrici non quadrate
S = la.diagsvd(s, A.shape[0], A.shape[1])
A_reconstructed = u @ S @ vt
```

---

## 4. Ottimizzazione delle Prestazioni: Vettorizzazione vs Loop

Il docente dimostra l'importanza della **vettorizzazione** in Python rispetto all'uso dei cicli `for`.

### Esperimento di Benchmarking:
- **Metodo Loop:** Ricostruire $A_k$ iterando sui primi $k$ prodotti esterni ($\sigma_i u_i v_i^T$) e sommandoli.
- **Metodo Vettorizzato:** Utilizzare l'operatore `@` (moltiplicazione di matrici) ottimizzato da librerie BLAS/LAPACK.

**Risultati:** Per una matrice $1000 \times 1500$, il metodo vettorizzato è **ordini di grandezza più veloce** (frazioni di secondo contro diversi secondi).

### Il concetto di Broadcasting in NumPy
Un metodo ancora più avanzato e performante per moltiplicare le colonne di $U$ per i rispettivi scalari in `s` senza creare una matrice diagonale esplicita:
```python
# Moltiplica ogni riga di U per il vettore s (element-wise) grazie al broadcasting
# Poi esegue il prodotto matriciale con vt
A_vec = (u * s) @ vt
```
NumPy adatta automaticamente le dimensioni del vettore `s` per corrispondere a quelle della matrice `u`, rendendo l'operazione estremamente efficiente dal punto di vista della memoria e del calcolo.

---

## ⏱️ Min 30-60

Ecco degli appunti universitari dettagliati basati sul segmento video:

---

# Analisi Numerica per il Machine Learning: SVD e Ottimizzazione Computazionale

## 1. Ottimizzazione della Ricostruzione: Loop vs. Vettorializzazione

L'implementazione algoritmica può influenzare drasticamente le prestazioni in termini di tempo e memoria. Nel caso della ricostruzione di una matrice tramite SVD ($A = U \cdot S \cdot V^T$):

### Vantaggi della Vettorializzazione
L'uso del **broadcasting** di NumPy permette di evitare la creazione esplicita della matrice diagonale $S$, portando a:
- **Efficienza di Memoria:** Memorizzare $S$ come matrice $n \times n$ ha un costo spaziale di $O(n^2)$. Per una matrice $1000 \times 1000$, ciò significa gestire un milione di numeri. Memorizzare solo il vettore dei valori singolari $s$ riduce drasticamente questo costo.
- **Efficienza Computazionale:** Si evita l'operazione di sommatoria esplicita tipica dei cicli `for`.

### Benchmarking e Prestazioni
I risultati mostrano una differenza di prestazioni significativa:
- **Naive For Loop:** ~7 secondi (eseguito interamente nell'interprete Python).
- **Operatore `@` Ottimizzato:** ~0.11 secondi (>30 volte più veloce).
- **Metodo Vettorializzato (Broadcasting):** Ulteriore miglioramento (~1.8 volte più veloce dell'operatore `@`).

**Ragione Tecnica:** L'operatore `@` e le funzioni NumPy chiamano librerie sottostanti scritte in **C** o **Fortran**, già compilate e ottimizzate. Python puro, essendo interpretato, soffre di un elevato overhead, specialmente nei cicli, dovuto a controlli di tipo e di intervallo (*range checking*) eseguiti a ogni iterazione.

---

## 2. Compressione Immagini tramite SVD

L'SVD è uno strumento fondamentale per la riduzione della dimensionalità e la compressione dei dati.

### Caricamento e Rappresentazione dei Dati
Le immagini vengono caricate utilizzando `matplotlib.image` e gestite come array NumPy tridimensionali (**ndarrays**):
- **Dimensioni:** `(Altezza, Larghezza, Canali)`.
- **Canali (RGB):** Il terzo asse (indice 2) rappresenta i colori Rosso, Verde e Blu.
- **Esempio:** Un'immagine di $567 \times 630 \times 3$ contiene circa 1 milione di pixel totali.

### Pre-processing: Conversione in Scala di Grigi
Per semplificare l'applicazione dell'SVD (che opera su matrici 2D), l'immagine viene convertita in scala di grigi mediando i canali colore.

**Operazione NumPy:** `X = np.mean(A, axis=2)`
- **Parametro `axis=2`:** Indica a NumPy di calcolare la media lungo il terzo asse (quello dei canali RGB).
- **Risultato:** Si passa da un tensore 3D a una matrice 2D di dimensioni $567 \times 630$.

### Visualizzazione con Matplotlib
Per una corretta visualizzazione della matrice risultante come immagine in bianco e nero, è necessario specificare la mappa di colori (*colormap*):
```python
img = plt.imshow(X)
img.set_cmap('gray')
plt.show()
```
Senza `set_cmap('gray')`, Matplotlib utilizza di default 'viridis', che mappa i valori scalari su una scala di colori (giallo-viola), rendendo l'immagine simile a una mappa termica o a un plot di superficie.

---

## 3. Analisi della Decomposizione e Varianza Spiegata

Eseguendo `u, s, vt = np.linalg.svd(X, full_matrices=False)`, otteniamo i valori singolari che determinano l'importanza di ogni "componente" dell'immagine.

### Criteri di Taglio (Troncamento)
Per comprimere l'immagine, dobbiamo decidere quanti valori singolari ($k$) mantenere. Si analizzano tre grafici:
1. **Andamento dei Valori Singolari ($\sigma_k$):** Spesso visualizzato in scala semi-logaritmica per evidenziare il decadimento rapido dopo i primi valori dominanti.
2. **Frazione Cumulata dei Valori Singolari:** Mostra quanta informazione "totale" stiamo catturando al crescere di $k$.
3. **Varianza Spiegata:** $\frac{\sum_{i=1}^k \sigma_i^2}{\sum \sigma_i^2}$. È un indicatore standard nel machine learning (simile alla PCA) per determinare quanta energia o varianza del segnale originale viene preservata.

**Osservazione pratica:** Spesso si sceglie un valore di $k$ tale da spiegare il **95% o il 99% della varianza**. Nel video viene mostrato come, per l'immagine della nebulosa, il 90% della varianza si raggiunga già con circa 300 valori singolari su oltre 500 disponibili.

---

## Glossario Tecnico
- **Vectorization (Vettorializzazione):** Tecnica di programmazione che sostituisce cicli espliciti con operazioni su array, sfruttando ottimizzazioni a basso livello (SIMD).
- **Broadcasting:** Capacità di NumPy di eseguire operazioni aritmetiche su array di forme diverse.
- **Machine Epsilon:** Il più piccolo numero che, aggiunto a 1.0, produce un risultato diverso da 1.0 (limite della precisione di macchina).
- **Explained Variance (Varianza Spiegata):** Rapporto tra la varianza catturata dalle componenti selezionate e la varianza totale dei dati.

---

## ⏱️ Min 60-90

Ecco degli appunti universitari dettagliati basati sul segmento video riguardante la compressione delle immagini tramite la Decomposizione ai Valori Singolari (SVD).

---

# Appunti di Calcolo Numerico: Compressione delle Immagini via SVD

## 1. Approssimazione di Rango Inferiore ($A_k$)
L'obiettivo primario della compressione tramite SVD è approssimare una matrice originale $A$ (l'immagine) con una matrice $A_k$ di rango $k < \text{rank}(A)$. Matematicamente, questo si ottiene troncando la somma della decomposizione:

$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

### Implementazione in Python (NumPy)
Per ricostruire la matrice $A_k$ mantenendo i primi $k$ valori singolari, si utilizza la seguente sintassi di *slicing*:

```python
# U, s, VT sono ottenuti da np.linalg.svd(A)
# Ricostruzione per un dato valore di k
A_k = np.matmul(U[:, :k], np.matmul(np.diag(s[:k]), VT[:k, :]))
```

**Osservazioni dalla visualizzazione:**
*   **$k$ bassi ($k=1, 2, 5$):** L'immagine appare estremamente sfuocata e irriconoscibile, catturando solo le variazioni di intensità più grossolane.
*   **$k$ medi ($k=50, 100$):** Le strutture principali diventano visibili, ma l'immagine presenta ancora molti artefatti e appare "pixelata" in modo rettangolare.
*   **$k$ alti ($k=300, 500$):** L'approssimazione è quasi indistinguibile dall'originale a occhio nudo. 

**Nota tecnica sul dataset:** Nel video viene utilizzata l'immagine di una galassia. Il docente sottolinea che immagini con texture complesse e molti dettagli casuali (simili a rumore) sono più difficili da comprimere efficacemente rispetto a immagini geometriche (es. un quadro di Mondrian), poiché richiedono un valore di $k$ più elevato per mantenere la fedeltà visiva.

---

## 2. Visualizzazione delle Matrici di Rango 1
Ogni componente della somma SVD, ovvero $u_k v_k^T$, è una matrice di rango 1. La matrice originale è una combinazione lineare di queste "basi" pesata dai valori singolari $\sigma_k$.

### Codice per la $k$-esima componente:
```python
# Prodotto esterno del k-esimo vettore di U e del k-esimo di VT
ukvk = np.outer(U[:, k-1], VT[k-1, :])
```

**Analisi delle componenti:**
*   Le prime componenti (associate ai $\sigma$ maggiori) mostrano strutture a blocchi orizzontali e verticali molto ampie (frequenze spaziali basse).
*   All'aumentare di $k$, le matrici di rango 1 mostrano dettagli sempre più fini e granulari.
*   Le componenti con $k$ molto alto appaiono come rumore puro. Questo spiega perché l'SVD è utile per il *denoising*: eliminando le componenti associate ai valori singolari più piccoli, si rimuove di fatto il rumore dall'immagine.

---

## 3. Decomposizione ai Valori Singolari Randomizzata (rSVD)
Per matrici di dimensioni molto elevate, il calcolo della SVD standard è computazionalmente oneroso. La **Randomized SVD** è un'alternativa efficiente che utilizza tecniche di proiezione casuale per trovare un sottospazio a bassa dimensionalità che approssimi bene l'intervallo della matrice originale.

### Passaggi dell'Algoritmo rSVD:
1.  **Campionamento (Sketching):** Si crea una matrice casuale $P$ di dimensioni $n \times k$.
2.  **Proiezione:** Si proietta la matrice originale $A$ su $P$ per ottenere una matrice "sommario" $Z = AP$.
3.  **Ortogonalizzazione:** Si esegue la decomposizione QR su $Z$ per ottenere una base ortonormale $Q$ ($Z = QR$).
4.  **Riduzione:** Si proietta $A$ nel sottospazio più piccolo definito da $Q$: $Y = Q^T A$.
5.  **SVD Locale:** Si calcola la SVD della matrice ridotta $Y$ (molto più piccola di $A$): $Y = U_y \Sigma V^T$.
6.  **Recupero:** Si ottiene la matrice $U$ originale proiettando $U_y$ indietro: $U = Q U_y$.

### Implementazione in Python:
Il video guida alla scrittura di una funzione `randomized_SVD(A, k)`:

```python
import numpy as np

def randomized_SVD(A, k):
    # 1. Ottenere le dimensioni
    n = A.shape[1]
    
    # 2. Matrice casuale (distribuzione normale)
    # Nota: np.random.randn genera da Gaussiana, rand genera Uniforme
    P = np.random.randn(n, k)
    
    # 3. Proiezione e QR
    Z = A @ P
    Q, _ = np.linalg.qr(Z)
    
    # 4. SVD nel sottospazio ridotto
    Y = Q.T @ A
    Uy, s, VT = np.linalg.svd(Y, full_matrices=False)
    
    # 5. Ricostruzione del vettore U originale
    U = Q @ Uy
    
    return U, s, VT
```

### Termini Tecnici Chiave:
*   **`np.random.randn`**: Utilizzato per generare rumore Gaussiano, preferibile in contesti di proiezione casuale rispetto alla distribuzione uniforme.
*   **Decomposizione QR**: Utilizzata per estrarre una base ortonormale da una matrice campionata.
*   **Matrice di Sketch ($P$)**: La matrice casuale utilizzata per "comprimere" le informazioni di $A$ prima della decomposizione.
*   **Sottospazio di Krylov**: Sebbene non citato esplicitamente in questo frammento, è il concetto teorico alla base del miglioramento delle performance delle proiezioni iterate.

---

## ⏱️ Min 90-120

Ecco degli appunti universitari dettagliati basati sul segmento video fornito, che trattano l'applicazione della Decomposizione ai Valori Singolari (SVD) nella compressione delle immagini e nella rimozione del rumore.

---

# Appunti di Calcolo Numerico: SVD, Approssimazione di Basso Rango e Denoising

## 1. Decomposizione ai Valori Singolari Randomizzata (rSVD)
La rSVD viene utilizzata per computare un'approssimazione di rango $k$ di una matrice generica $A$ con un costo computazionale notevolmente inferiore rispetto alla SVD completa.

### Implementazione in Python (Sintesi)
Il metodo si basa sulla proiezione della matrice originale in uno spazio di dimensione inferiore utilizzando una matrice casuale, seguita dall'esecuzione della SVD sulla matrice proiettata.

```python
def randomized_SVD(A, k):
    # n = numero di colonne di A
    P = np.random.randn(n, k) # Matrice di proiezione casuale
    Z = A @ P
    Q, _ = np.linalg.qr(Z)    # Decomposizione QR per trovare base ortonormale
    Y = Q.T @ A
    Uy, sy, VTy = np.linalg.svd(Y, full_matrices=False)
    U = Q @ Uy
    return U, sy, VTy
```

### Osservazioni Chiave:
- **Accuratezza vs. Rango ($k$):** Maggiore è il valore di $k$, migliore è l'approssimazione dei valori singolari reali. 
- **Matching dei Valori Singolari:** Nel confronto tra SVD esatta e rSVD, i primi valori singolari (quelli di ordine superiore, più significativi) tendono a coincidere quasi perfettamente. Con $k=100$, l'approssimazione rimane solida fino a circa i primi 30-40 valori singolari, dopodiché la rSVD inizia a divergere dall'esatta a causa della perdita di informazione nello spazio rimosso.
- **Vantaggio Computazionale:** Il risparmio in termini di tempo di calcolo è il driver principale per l'utilizzo della rSVD in dataset di grandi dimensioni.

---

## 2. Effetto delle Caratteristiche Geometriche sulla Compressione
L'efficacia della SVD nella compressione (approssimazione di basso rango) dipende fortemente dalla natura dell'immagine.

### Esempio: Il Quadro di Mondrian
Il quadro di Mondrian presenta caratteristiche geometriche nette (linee orizzontali e verticali) perfettamente allineate con gli assi della matrice dei pixel.
- **Risultato:** La compressione è estremamente efficiente. Con un rango $k$ molto basso (es. $k=50$), la ricostruzione è virtualmente indistinguibile dall'originale ("quasi lossless").
- **Valori Singolari:** Il grafico dei valori singolari mostra un drop repentino verso lo zero (prossimo all'epsilon di macchina), indicando che poche componenti catturano quasi tutta la varianza dell'immagine.

### L'Impatto della Rotazione
Se si ruota la stessa immagine di Mondrian di un piccolo angolo (es. 20°):
- **Decadimento delle Performance:** L'allineamento geometrico con gli assi viene perso. 
- **Valori Singolari:** Non decadono più rapidamente verso lo zero. La curva diventa molto più "piatta", indicando che sono necessari molti più valori singolari per ottenere una ricostruzione accettabile.
- **Conclusione:** SVD predilige feature geometriche semplici e allineate. Anche una piccola rotazione "rompe" la struttura di basso rango che rendeva efficiente la compressione.

---

## 3. Rimozione del Rumore (Denoising) tramite SVD
Il rumore in un'immagine è solitamente caratterizzato da "feature fini" ad alta frequenza che non hanno una struttura coerente di basso rango.

### Logica del Denoising
Nella SVD, il segnale principale è catturato dai valori singolari più grandi, mentre il rumore è distribuito tra i valori singolari più piccoli (la "coda" dello spettro). troncando la decomposizione ad un certo valore soglia, è possibile rimuovere gran parte del rumore preservando l'informazione essenziale.

### Setup Sperimentale
1. **Trasformazione in Scala di Grigi:** Si lavora su una matrice $X$ 2D.
2. **Normalizzazione:** Valori scalati tra $[0, 1]$ per consistenza algoritmica.
3. **Aggiunta di Rumore:** $X_{noisy} = X + \gamma \cdot \text{rumore\_gaussiano}$, dove $\gamma$ rappresenta la magnitudo del rumore.

### Metodi di Thresholding (Soglia)
Per decidere dove tagliare i valori singolari, si utilizzano diverse strategie:
1. **Soglia Hard Ottimale (Rumore Noto):** Se la magnitudo del rumore è nota, si usa una formula analitica:
   $$\tau = \frac{4}{\sqrt{3}}\sqrt{n} \cdot \gamma$$
2. **Soglia Hard Ottimale (Rumore Ignoto):** Si stima la magnitudo basandosi sulla mediana dei valori singolari osservati ($\sigma_{med}$).
3. **Soglia Energetica (Benchmark):** Si mantiene una percentuale fissa dell'energia totale (es. il 90% della somma cumulativa dei valori singolari).

### Procedura Algoritmica
- Calcolare la SVD della matrice rumorosa.
- Identificare l'indice $k$ tale per cui $\sigma_k > \text{Soglia}$.
- Ricostruire la matrice utilizzando solo le prime $k$ componenti: $A_k = U_k \Sigma_k V_k^T$.

---

## Termini Tecnici Chiave
- **Low-rank approximation:** Rappresentazione di una matrice complessa tramite una somma di matrici di rango 1.
- **Cumulative fraction of singular values:** Misura di quanta "energia" o informazione è contenuta nelle prime $k$ componenti.
- **Machine Epsilon ($\epsilon$):** Il più piccolo numero rappresentabile in virgola mobile; nel grafico indica quando un valore singolare è numericamente nullo.
- **Aspect Ratio ($\beta$):** Rapporto tra righe e colonne della matrice ($m/n$), influenza il calcolo della soglia ottimale.

---

## ⏱️ Min 120-150

Certamente. Ecco degli appunti universitari dettagliati basati sulla lezione video riguardante l'applicazione della **Decomposizione ai Valori Singolari (SVD)** per la rimozione del rumore e l'estrazione del background.

---

# Appunti di Analisi Numerica: SVD per il Denoising e Background Removal

## 1. Introduzione al Denoising tramite SVD
La rimozione del rumore (denoising) basata su SVD sfrutta la proprietà che le informazioni significative di un segnale (o di un'immagine) sono solitamente concentrate nei primi valori singolari (quelli più alti), mentre il rumore è distribuito tra i valori singolari più bassi.

Dato un sistema $X = U S V^T$, l'obiettivo è trovare un valore di soglia (**threshold**) tale da troncare i valori singolari che rappresentano il rumore, mantenendo solo quelli "puliti".

---

## 2. Strategie di Thresholding (Soglia)
Durante la lezione sono state discusse tre tecniche principali per calcolare la soglia ottimale di troncamento:

### A. Soglia Hard Ottimale (Matrici Quadrate)
Utilizzata quando la matrice è quadrata e il livello di rumore $\gamma$ è noto.
*   **Formula:** $\tau = \frac{4}{\sqrt{3}} \sqrt{n} \cdot \gamma$
*   **Implementazione Python:**
    ```python
    cutoff = (4 / np.sqrt(3)) * np.sqrt(n) * gamma
    ```

### B. Soglia con Rumore Ignoto (Matrici Non Quadrate)
Basata sulla teoria di Gavish e Donoho per matrici di dimensioni $m \times n$.
*   **Formula:** $\tau = \omega(\beta) \cdot s_{median}$
*   Dove $\beta = m/n$ e $\omega(\beta)$ è un'approssimazione polinomiale:
    $\omega(\beta) \approx 0.56\beta^3 - 0.95\beta^2 + 1.82\beta + 1.43$
*   **Implementazione Python:**
    ```python
    beta = m / n
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    cutoff_unknown = omega * np.median(s)
    ```

### C. Soglia Energetica (90% dell'Energia)
Consiste nel mantenere un numero di valori singolari sufficiente a spiegare il 90% della varianza totale (somma cumulativa dei valori singolari).
*   **Logica:** Si calcola la somma cumulativa normalizzata e si identifica l'indice in cui si supera lo 0.9.
*   **Implementazione Python:**
    ```python
    cs = np.cumsum(s) / np.sum(s)
    r90 = np.where(cs > 0.9)[0][0] # Prende il primo indice che supera 0.9
    cutoff90 = s[r90]
    ```

---

## 3. Visualizzazione e Analisi dei Risultati
Un metodo empirico molto efficace discusso è la **Regola del Gomito (Elbow Rule)**. Plottando i valori singolari su scala logaritmica (`plt.semilogy`), si nota spesso un calo repentino seguito da una "coda" piatta. Il punto di flessione (gomito) indica visivamente dove il segnale termina e inizia il rumore.

*   **Confronto degli Errori:** Per valutare la bontà del denoising, si sottrae l'immagine ricostruita dall'originale "pulita" (se disponibile).
*   **Osservazione:** La soglia del 90% dell'energia è spesso troppo conservativa (mantiene troppo rumore), mentre le soglie statistiche (Gavish-Donoho) tendono a posizionarsi più precisamente vicino al "gomito".

---

## 4. Applicazione: Background Removal nei Video
L'SVD può essere utilizzata per separare lo sfondo statico dagli oggetti in movimento in un video di sorveglianza.

### Procedura Tecnica:
1.  **Costruzione della Matrice:** Ogni frame del video viene linearizzato (flattened) e inserito come colonna in una grande matrice $X$.
    *   Righe: Pixel totali di un frame.
    *   Colonne: Numero di frame campionati.
2.  **Pre-processing:** Conversione in scala di grigi e ridimensionamento (resizing) per ridurre il carico computazionale.
3.  **Applicazione SVD:** Poiché lo sfondo è la parte più costante del video, esso corrisponde alla componente principale con il valore singolare più alto (Rank-1 approximation).
4.  **Estrazione:**
    *   **Background:** Ricostruito usando solo il primo valore singolare ($k=1$).
    *   **Foreground (Oggetti in movimento):** Ottenuto sottraendo il background dal video originale o analizzando le componenti di rango superiore.

### Risultato:
*   La prima "eigen-image" rappresenta la media stazionaria del video (lo sfondo).
*   Le differenze tra i frame originali e questa ricostruzione di rango 1 isolano efficacemente le persone o i veicoli in movimento.

---

## Termini Tecnici Chiave
*   **SVD (Singular Value Decomposition):** Fattorizzazione di una matrice in $U$, $S$ e $V^T$.
*   **Valori Singolari ($s$):** Elementi diagonali della matrice $S$, indicano l'importanza di ogni componente.
*   **Rank-k Approximation:** Ricostruzione di una matrice usando solo i primi $k$ valori singolari.
*   **Cumulative Sum (Somma Cumulativa):** Utilizzata per misurare la ritenzione energetica del modello.
*   **Log-scale plot:** Grafico in scala logaritmica fondamentale per visualizzare il decadimento dei valori singolari.

---

## ⏱️ Min 150-180

Certamente! Ecco una serie di appunti universitari dettagliati basati sul segmento video fornito.

---

# 📚 Appunti di Analisi Numerica e Machine Learning

## 1. Applicazione della SVD all'Elaborazione Video
Il segmento inizia con una dimostrazione pratica di come la **Singular Value Decomposition (SVD)** possa essere utilizzata per separare gli elementi statici da quelli dinamici in un video (es. sorveglianza).

### 1.1 Scomposizione Sfondo/Primo Piano
*   **Concetto:** In una sequenza video fissa, lo sfondo rimane pressoché costante in tutti i frame. Matematicamente, questo significa che l'informazione dello sfondo è contenuta nel primo valore singolare (quello dominante).
*   **Ricostruzione dello Sfondo:** Utilizzando solo il primo valore singolare e i relativi vettori singolari ($k=1$), è possibile ricostruire un'immagine che rappresenta la media temporale della scena, ovvero lo sfondo pulito.
*   **Estrazione del Primo Piano:** Sottraendo l'immagine di "sfondo" ricostruita dal video originale, si ottiene il "foreground", che contiene gli elementi in movimento (le persone).

### 1.2 Ottimizzazione con Randomized SVD (rSVD)
*   **Problema:** Calcolare la SVD completa per matrici di grandi dimensioni (video ad alta risoluzione o molti frame) è computazionalmente oneroso.
*   **Soluzione:** L'uso della **Randomized SVD**.
*   **Vantaggi:**
    *   **Velocità:** Può essere ordini di grandezza più veloce della SVD classica.
    *   **Efficienza:** Poiché ci interessano solo i primi valori singolari (per lo sfondo ne basta spesso uno), la rSVD approssima perfettamente lo spazio di interesse con un costo ridotto.
    *   **Parametri:** Impostando un rango di approssimazione $k$ (es. tra 10 e 20), si ottiene una ricostruzione di alta qualità in frazioni di secondo.

---

## 2. Approfondimenti Tecnici (Q&A Post-Lezione)

### 2.1 Gestione dell'Errore e Normalizzazione
Durante la visualizzazione dell'errore di ricostruzione, sorge il problema dei valori negativi.
*   **Visualizzazione:** Le librerie grafiche spesso riscalano i valori tra il minimo e il massimo (0-255).
*   **Interpretazione:** In una mappa di errore, uno zero (nessun errore) potrebbe apparire grigio, mentre errori positivi o negativi appaiono come bianco o nero. Per analizzare l'intensità dell'errore, è spesso preferibile considerare il valore assoluto.

### 2.2 Pre-processing: Grayscale vs Color
*   **Perché il Grayscale?** Riduce la dimensione dei dati di un fattore 3 (un solo canale invece di tre: R, G, B).
*   **Metodo:** Si effettua una media pesata dei canali colore. È una tecnica standard che mantiene la coerenza visiva degli oggetti semplificando drasticamente il calcolo matriciale.

### 2.3 Metodo delle Potenze (Power Iteration) nella rSVD
Un punto cruciale discusso riguarda l'uso di iterazioni di potenza per migliorare l'accuratezza della rSVD.
*   **Funzionamento:** Si moltiplica la matrice originale per la sua trasposta diverse volte ($A(A^TA)^q$).
*   **Scopo:** Questo processo "amplifica" i valori singolari dominanti rispetto al rumore (i valori singolari più piccoli), rendendo la base ortogonale scelta molto più precisa.
*   **Trade-off:**
    *   **Vantaggio:** Maggiore precisione nell'approssimazione del range della matrice.
    *   **Svantaggio:** Aumenta il costo computazionale e può introdurre instabilità numerica se non gestito correttamente (necessità di ri-ortogonalizzazione).

### 2.4 Struttura dei Dati: "Flattening" dei Frame
*   **Formattazione della Matrice $A$:** Ogni colonna della matrice rappresenta un intero frame del video.
*   **Processo:** Un'immagine 2D (pixel $x, y$) viene "appiattita" (flattened) in un unico vettore colonna. Se il video ha $n$ frame, la matrice avrà $n$ colonne.

---

## 3. Contesto Accademico e Applicazioni Future
La SVD non è solo uno strumento per le immagini, ma una tecnica fondamentale di riduzione della dimensionalità.

*   **Principal Component Analysis (PCA):** La SVD è il motore computazionale della PCA, utilizzata in statistica per analizzare dataset complessi (es. dati genomici o oncologici).
*   **Equazioni alle Derivate Parziali (PDE):** In analisi numerica avanzata, è possibile trattare la soluzione di una PDE come un'immagine/matrice e applicare tecniche di riduzione dell'ordine del modello per accelerare le simulazioni.
*   **Interdisciplinarità:** Il docente cita corsi correlati come "Deep Learning" (Borrachi/Matteucci) e "Metodi Numerici per PDE" (Pagani/Manzoni), sottolineando come la decomposizione spettrale sia un filo conduttore tra l'informatica e la matematica applicata.

---
*Note: Questi appunti integrano la parte teorica della lezione con le risposte pratiche fornite dal docente ai dubbi degli studenti.*

---

