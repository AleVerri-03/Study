# 📝 Appunti: Matteo Caldana's Personal Room-20251017 1403-1
**Modello:** gemini-3-flash-preview | **Data:** 23/01/2026

---

## ⏱️ Min 0-30

Ecco degli appunti universitari dettagliati basati sulla lezione di laboratorio di **Analisi Numerica per il Machine Learning**.

---

# Laboratorio di Analisi Numerica per il Machine Learning

## 1. Obiettivi della Sessione
La lezione si divide in due parti principali:
1.  **Analisi Esplorativa e PCA**: Applicazione della Principal Component Analysis (PCA) su un dataset reale di diagnostica proteomica.
2.  **Regressione e Algebra Lineare Numerica**: Implementazione efficiente della pseudo-inversa di Moore-Penrose tramite Decomposizione a Valori Singolari (SVD).

---

## 2. Caso di Studio: Diagnostica Proteomica del Siero
Il dataset analizzato proviene dal programma di proteomica clinica FDA-NCI.
*   **Struttura**: 216 pazienti (campioni).
*   **Feature**: 2000 o 4000 biomarcatori proteici (intensità ionica a specifici valori di massa-carica MZ).
*   **Target**: Variabile binaria (Paziente Sano vs. Paziente con Cancro Ovarico).

### Pre-processing e Analisi Preliminare
*   **Bilanciamento del Dataset**: È fondamentale verificare la distribuzione delle classi. In questo caso, abbiamo 121 campioni "Cancer" e 95 "Normal". Il dataset è considerato bilanciato, il che evita bias durante l'addestramento del modello.
*   **Codifica delle Etichette**: Trasformazione delle etichette testuali in vettori booleani/numerici (0 e 1) utilizzando la funzione `np.where` di NumPy per facilitare il calcolo computazionale.
*   **Problema della Visualizzazione Diretta**: Uno scatter plot di due o tre proteine scelte casualmente non permette di distinguere le classi. I dati appaiono sovrapposti e non linearmente separabili nello spazio delle feature originali.

---

## 3. Analisi delle Componenti Principali (PCA)
La PCA viene utilizzata per la riduzione della dimensionalità e la visualizzazione.

### Implementazione Matematica in Python
1.  **Calcolo della Media**: `A_mean = A.mean(axis=1)` (media lungo i campioni).
2.  **Centratura dei Dati**: Sottrazione della media dalla matrice originale per ottenere la matrice centrata $\bar{A}$.
3.  **SVD (Singular Value Decomposition)**: Applicazione di `np.linalg.svd` sulla matrice centrata.
    *   `U`: Direzioni principali (autovettori della matrice di covarianza).
    *   `s`: Singolar valori (correlati alla varianza spiegata).

### Analisi dei Singolar Valori
*   **Elbow Rule (Regola del Gomito)**: Osservando il grafico dei singolar valori in scala logaritmica, si cerca il punto di flesso ("gomito") dove l'importanza dei componenti decade drasticamente. Nel dataset in esame, il taglio ottimale è suggerito intorno alla 25ª componente.
*   **Varianza Spiegata**: Il grafico della frazione cumulata della varianza mostra quanta informazione del dataset originale è conservata riducendo le dimensioni.

### Visualizzazione Proiettata
Proiettando i dati sulle prime due o tre componenti principali (PC1, PC2, PC3), le classi "Cancer" e "Normal" iniziano a formare cluster distinti, rendendo possibile la classificazione.

---

## 4. Classificazione con SVC (Support Vector Classification)
Dopo la riduzione dimensionale tramite PCA, viene applicato un classificatore **SVC** con kernel lineare.
*   **Metodologia**: Si traccia un iperpiano (un piano in 3D) che separa al meglio i campioni sani da quelli malati nello spazio delle componenti principali.
*   **Performance**: Il modello raggiunge un'accuratezza di circa l'**83%**. Questo dimostra l'efficacia della PCA nel sintetizzare feature rilevanti per la diagnosi medica.

---

## 5. Fondamenti Numerici: Pseudo-inversa di Moore-Penrose
La seconda parte della lezione si focalizza sul calcolo della pseudo-inversa ($A^\dagger$), essenziale per risolvere problemi di regressione ai minimi quadrati.

### Implementazione tramite SVD
La formula utilizzata è: $A^\dagger = V \Sigma^\dagger U^T$.

Vengono confrontate tre metodologie:
1.  **Full SVD**: Utilizza matrici complete. Meno efficiente per la memoria.
2.  **Thin SVD**: (`full_matrices=False`) Calcola solo le parti necessarie delle matrici $U$ e $V$, riducendo drasticamente il costo computazionale.
3.  **Ottimizzazione tramite Broadcasting**: Invece di costruire esplicitamente la matrice diagonale $\Sigma^\dagger$, si utilizza il broadcasting di NumPy per moltiplicare i vettori, velocizzando ulteriormente l'operazione.

### Analisi delle Prestazioni (`%timeit`)
Il test di benchmarking mostra che:
*   L'implementazione personalizzata basata su **Thin SVD** e **Broadcasting** risulta significativamente più veloce (spesso il doppio) rispetto all'implementazione standard di `np.linalg.pinv` per matrici di determinate dimensioni.
*   Questo evidenzia l'importanza di conoscere la struttura del problema (es. matrici diagonali) per scrivere codice performante in ambito Machine Learning.

---

## Termini Tecnici Chiave
*   **Biomarcatore**: Indicatore biologico misurabile (proteina).
*   **SVD (Singular Value Decomposition)**: Decomposizione di una matrice in $U \Sigma V^T$.
*   **Least Squares Regression**: Metodo per trovare la linea di miglior fitting minimizzando la somma dei quadrati dei residui.
*   **Broadcasting**: Tecnica di NumPy per eseguire operazioni su array di forme diverse.
*   **Machine Epsilon**: Il più piccolo numero che, aggiunto a 1, dà un risultato diverso da 1 (limite di precisione numerica).

---

## ⏱️ Min 30-60

Certamente. Ecco degli appunti universitari dettagliati basati sul segmento video riguardante la regressione lineare e i minimi quadrati.

---

# Appunti di Machine Learning: Regressione Lineare e Metodo dei Minimi Quadrati

## 1. Definizione Matematica del Problema
Il problema dei minimi quadrati (*Least Squares Problem*) mira a trovare i parametri ottimali di un modello lineare che meglio approssima un insieme di dati rumorosi.

### Il Modello
Dato un insieme di punti, vogliamo determinare la retta di regressione definita da:
$$y = mx + q$$
Dove:
*   **$m$**: Coefficiente angolare (pendenza).
*   **$q$**: Intercetta (termine noto).

### Rappresentazione Matriciale
Per risolvere il problema in modo efficiente, utilizziamo una matrice dei pesi $w$ e una matrice di design $\Phi$:
*   **Vettore dei pesi**: $w = [m, q]^T$
*   **Matrice di Design ($\Phi$)**: $\Phi = [X, \mathbf{1}] \in \mathbb{R}^{N \times 2}$. Ogni riga contiene la coordinata $x_i$ e un valore costante $1$ per permettere il calcolo dell'intercetta $q$.
*   **Equazione Lineare**: $\Phi w = y$

### La Soluzione Ottimale
La soluzione ai minimi quadrati si ottiene tramite la **Pseudo-inversa di Moore-Penrose** ($\Phi^\dagger$):
$$w = \Phi^\dagger y$$
Questa soluzione rappresenta la proiezione del vettore dei target $y$ nello spazio delle colonne di $\Phi$, minimizzando la norma dell'errore $\| \Phi w - y \|_2^2$.

---

## 2. Implementazione Pratica in Python (NumPy)

Il docente illustra come generare dati sintetici e risolvere il sistema numericamente.

### Generazione dei Dati
Si scelgono i parametri reali (es. $m=2.0, q=3.0$) e si aggiunge del rumore gaussiano:
```python
import numpy as np
import matplotlib.pyplot as plt

N = 100
m, q = 2.0, 3.0
noise_magnitude = 2.0

# Generazione x e y reale
x = np.random.randn(N)
y_real = m * x + q

# Aggiunta di rumore
y = y_real + np.random.randn(N) * noise_magnitude
```

### Costruzione della Matrice $\Phi$
Per includere l'intercetta $q$, si affianca al vettore $x$ una colonna di uni:
```python
# Utilizzo di np.column_stack per creare la matrice di design
Phi = np.column_stack((x, np.ones(N)))
```

### Calcolo dei Pesi ($w$)
Si utilizza la pseudo-inversa calcolata tramite la decomposizione ai valori singolari (SVD):
```python
# w[0] sarà m_stimato, w[1] sarà q_stimato
w = my_pinv_thinSVD(Phi) @ y
```

---

## 3. Analisi della Convergenza e del Rumore

Il video mette in evidenza come l'accuratezza del modello stimato (linea nera) rispetto al modello reale (linea rossa) dipenda da due fattori critici:

1.  **Magnitudo del Rumore ($\sigma$):**
    *   Se il rumore è basso (es. `noise = 0.1`), la linea stimata è quasi identica a quella reale.
    *   Se il rumore è alto (es. `noise = 2.0`), i punti sono dispersi e la stima richiede più dati per essere precisa.
2.  **Numero di Campioni ($N$):**
    *   All'aumentare di $N$ (es. passando da 100 a 10.000 punti), la stima dei parametri $[m, q]$ converge verso i valori reali nonostante la presenza di rumore elevato. Questa è una dimostrazione empirica della legge dei grandi numeri applicata alla regressione.

---

## 4. Metodi Alternativi: Le Equazioni Normali

Oltre alla pseudo-inversa, il docente introduce il metodo delle **Equazioni Normali**, che trasforma il problema in un sistema quadrato:
$$(\Phi^T \Phi) w = \Phi^T y$$

### Risoluzione con `np.linalg.solve`
Invece di calcolare esplicitamente l'inversa (che può essere numericamente instabile), si utilizza un solutore lineare:
```python
# Risoluzione del sistema quadrato
w2 = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
```
**Nota Tecnica:** La differenza tra il risultato ottenuto con la pseudo-inversa e quello con le equazioni normali è dell'ordine di $10^{-16}$ (vicino alla *machine epsilon*), indicando che entrambi i metodi sono validi per sistemi di piccole dimensioni.

---

## 5. Introduzione alla Ridge Regression (Cenni)
Il segmento finale introduce il concetto di **Regolarizzazione**. Quando il problema è mal condizionato o si vuole evitare l'overfitting su modelli complessi, si aggiunge un parametro $\lambda$:
$$(\Phi^T \Phi + \lambda I) w = \Phi^T y$$
*   **$\lambda$ (Parametro di Regolarizzazione):** Penalizza i pesi troppo grandi, forzando il modello a essere più "semplice" o stabile.
*   Questo metodo è fondamentale quando si passa dalla regressione lineare semplice a modelli non lineari (regressione kernel).

---
*Fine degli appunti.*

---

## ⏱️ Min 60-90

Certamente. Ecco degli appunti universitari dettagliati basati sul segmento video fornito, che tratta della regressione Ridge e della sua formulazione "kernel-form".

---

# Appunti di Machine Learning: Regressione Ridge e Introduzione ai Metodi Kernel

## 1. Regressione Ridge (Ridge Regression)

La regressione Ridge è una variante della regressione lineare dei minimi quadrati che introduce un termine di regolarizzazione per prevenire l'overfitting e gestire problemi di malcondizionamento delle matrici.

### 1.1 Definizione e Parametro di Regolarizzazione
L'obiettivo è minimizzare la somma dei quadrati degli scarti aggiungendo una penalità proporzionale al quadrato della norma dei pesi ($L_2$ regularization):
$$\min_{w} ||\Phi w - y||^2 + \lambda ||w||^2$$
Dove:
*   $\Phi$ è la matrice delle feature (design matrix).
*   $w$ è il vettore dei pesi.
*   $y$ è il vettore dei target.
*   $\lambda$ è il **parametro di regolarizzazione** (nel video impostato inizialmente a $1.0$).

### 1.2 Metodi di Risoluzione

Esistono due approcci principali per risolvere le equazioni della regressione Ridge:

#### Metodo 1: Utilizzo delle Equazioni Normali
Si deriva direttamente la funzione di costo rispetto a $w$:
$$(\Phi^T\Phi + \lambda I)w = \Phi^T y$$
Questa forma è preferibile quando il numero di feature $D$ è inferiore al numero di campioni $N$.

#### Metodo 2: Identità di Woodbury (Forma Duale o "Kernel-form")
Sfruttando l'identità di Woodbury, possiamo riscrivere il problema in termini di un vettore di variabili duali $\alpha$:
1.  Si calcola $\alpha$ risolvendo il sistema: $(\Phi\Phi^T + \lambda I)\alpha = y$
2.  Si ricavano i pesi: $w = \Phi^T \alpha$

Questo approccio è particolarmente utile quando $N < D$, poiché la matrice da invertire ha dimensioni $N \times N$ invece di $D \times D$. Inoltre, pone le basi per la regressione kernel.

---

## 2. Implementazione Pratica in Python

Il video mostra la generazione di un modello sintetico per testare l'efficacia della regressione.

### 2.1 Generazione dei Dati
Viene utilizzato un modello non lineare basato sulla tangente iperbolica:
$$y = \tanh(2x - 1)$$
*   **Dati di training:** 100 punti campionati da una distribuzione gaussiana standard.
*   **Rumore sintetico:** Viene aggiunto un rumore gaussiano con media zero e deviazione standard $\sigma = 0.1$.
*   **Dati di test:** 1000 punti distribuiti uniformemente nell'intervallo $[-3, 3]$ per la visualizzazione.

### 2.2 Codice NumPy Rilevante
Per costruire la matrice $\Phi$ per una regressione lineare (con intercetta):
```python
Phi = np.column_stack([x, np.ones_like(x)])
```
Per risolvere il sistema lineare Ridge (Metodo 1):
```python
w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(2), Phi.T @ y)
```

---

## 3. Analisi del Parametro $\lambda$ (Hyperparameter Tuning)

Il comportamento del modello cambia drasticamente al variare di $\lambda$:

*   **$\lambda$ piccolo (es. 1.0):** Il modello è molto vicino ai minimi quadrati standard. Tende a seguire bene i dati ma può essere sensibile al rumore.
*   **$\lambda$ grande (es. 100, 1000):** La penalizzazione sui pesi diventa dominante.
    *   L'inclinazione (slope) e l'intercetta diminuiscono.
    *   Il modello diventa "più semplice" (meno flessibile).
*   **$\lambda \to \infty$:** Il modello collassa su una retta orizzontale passante per lo zero ($w \to 0$), indicando che la penalizzazione ha annullato ogni influenza dei dati.

**Nota tecnica:** Il video avverte che per $\lambda = 0$, l'identità di Woodbury non è direttamente applicabile se le matrici non sono invertibili, mentre le equazioni normali standard possono ancora funzionare (usando la pseudoinversa).

---

## 4. Verso la Regressione Kernel (Kernel Regression)

L'intuizione fondamentale è che il termine $\Phi\Phi^T$ nel Metodo 2 rappresenta la matrice delle somiglianze tra i punti (prodotto scalare).

### 4.1 La Matrice Kernel ($K$)
Possiamo sostituire il prodotto scalare $\Phi\Phi^T$ con una funzione kernel generica $K(z_i, z_j)$:
$$(K + \lambda I)\alpha = y$$
Dove $K_{ij} = K(x_i, x_j)$.

### 4.2 Esempi di Kernel menzionati:
1.  **Kernel Prodotto Scalare (Lineare):** $K(x_i, x_j) = x_i^T x_j + 1$.
2.  **Kernel Polinomiale:** $K(x_i, x_j) = (x_i^T x_j + 1)^q$.
3.  **Kernel Gaussiano (RBF):** $K(x_i, x_j) = \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right)$.

L'uso dei kernel permette di modellare relazioni non lineari complesse nello spazio originale dei dati mappandoli implicitamente in spazi di feature a dimensione infinita.

---

## ⏱️ Min 90-120

Certamente! Ecco degli appunti universitari dettagliati basati sul segmento video fornito, strutturati per uno studio approfondito della **Kernel Regression**.

---

# Appunti di Machine Learning: Kernel Regression e Metodi Kernel

## 1. Introduzione alla Kernel Regression
La **Kernel Regression** è una tecnica di regressione non parametrica che permette di modellare relazioni non lineari tra variabili mantenendo la semplicità computazionale di un problema lineare. L'idea fondamentale è mappare i dati in uno spazio di caratteristiche (*feature space*) a dimensione superiore dove una relazione lineare può essere appresa.

### Formulazione Matematica
Il problema della regressione viene risolto tramite l'equazione:
$$(K + \lambda I)\alpha = y$$

Dove:
*   **$K$**: Matrice Kernel (o matrice di Gram) di dimensioni $N \times N$.
*   **$\lambda$**: Parametro di regolarizzazione (*regularization parameter*).
*   **$I$**: Matrice identità.
*   **$\alpha$**: Vettore dei coefficienti da apprendere.
*   **$y$**: Vettore dei target osservati.

L'elemento generico della matrice è definito come $K_{ij} = \mathcal{K}(z_i, z_j)$, dove $\mathcal{K}$ è la funzione kernel scelta.

---

## 2. Tipologie di Kernel Analizzate
Il professore presenta tre tipologie principali di kernel, ciascuna con diverse proprietà di modellazione:

### 1. Kernel Prodotto Scalare (Scalar Product Kernel)
È la forma più semplice e riporta il problema a una regressione lineare standard.
$$\mathcal{K}(x_i, x_j) = x_i \cdot x_j + 1$$

### 2. Kernel Polinomiale di Ordine Superiore (Higher-order Scalar Product Kernel)
Permette di modellare curve polinomiali aumentando la complessità del modello tramite l'esponente $q > 1$.
$$\mathcal{K}(x_i, x_j) = (x_i \cdot x_j + 1)^q$$

### 3. Kernel Gaussiano (RBF - Radial Basis Function)
È un kernel estremamente flessibile basato sulla distanza euclidea tra i punti.
$$\mathcal{K}(x_i, x_j) = \exp \left( -\frac{\|x_i - x_j\|^2}{2\sigma^2} \right)$$
*   **$\sigma$**: Definisce la "larghezza" (*bandwidth*) del kernel, controllando l'area di influenza di ciascun punto.

---

## 3. Implementazione in Python (NumPy/Matplotlib)

### Definizione delle Funzioni Kernel
```python
import numpy as np

def product_kernel(x1, x2):
    return x1 * x2 + 1

def high_order_kernel(x1, x2, q=4):
    return (x1 * x2 + 1)**q

def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-((x1 - x2) / sigma)**2 / 2)
```

### Costruzione della Matrice Kernel
Per implementare la Kernel Regression, è necessario costruire la matrice $K$ valutando la funzione kernel su tutte le coppie di punti del dataset di training.

**Approccio con cicli for (didattico):**
Il professore suggerisce inizialmente un doppio ciclo `for` per chiarezza:
1. Inizializzare una matrice vuota $N \times N$.
2. Iterare su $i$ e $j$ per calcolare ogni entry $K[i, j]$.

**Ottimizzazione tramite Vettorizzazione:**
Viene sottolineato che i cicli `for` in Python sono inefficienti per dataset grandi. Per il kernel lineare, la matrice può essere calcolata come:
$$K = X X^T + 1$$
utilizzando operazioni matriciali di NumPy che sono significativamente più veloci.

---

## 4. Analisi dei Risultati e Hyperparameter Tuning

### Comportamento del Kernel Polinomiale ($q$)
*   **$q=1$**: Risultato lineare (retta rossa). Non riesce a catturare la forma a "S" dei dati.
*   **$q=4$ o $q=5$**: Fornisce un ottimo fit per dati con andamento sigmoideo o polinomiale complesso.
*   **$q=200$ (Overfitting estremo)**: Con un grado troppo elevato, il modello presenta oscillazioni selvagge tra i punti data per minimizzare l'errore, perdendo completamente la capacità di generalizzazione.

### Comportamento del Kernel Gaussiano ($\sigma$)
Il parametro $\sigma$ (sigma) è critico per il bilanciamento tra bias e varianza:
*   **$\sigma$ piccolo (es. 0.001)**: Il kernel è molto "stretto". Il modello tenta di interpolare ogni singolo punto creando picchi locali (spikes). Si ha un alto rischio di **overfitting**.
*   **$\sigma$ ottimale (es. 0.5)**: Le gaussiane si sovrappongono armoniosamente creando una curva morbida che segue bene il trend dei dati.
*   **$\sigma$ grande (es. 100)**: Il kernel diventa molto "largo". L'informazione viene condivisa globalmente tra tutti i punti, portando il modello a prevedere quasi ovunque il valore medio dei dati (una linea piatta). Si ha **underfitting**.

---

## 5. Considerazioni Finali
*   **Costo Computazionale**: La Kernel Regression risolve un problema lineare nello spazio delle feature, ma richiede l'inversione di una matrice $N \times N$. Questo può essere oneroso per dataset molto grandi.
*   **Flessibilità**: Il Kernel Gaussiano è generalmente più flessibile del polinomiale, ma richiede un'attenta scelta della banda $\sigma$.
*   **Conoscenza del Dominio**: In alcuni casi fisici, se si sa che la relazione è intrinsecamente quadratica o cubica, il kernel polinomiale con l'ordine corretto potrebbe essere preferibile per una migliore interpretazione fisica fuori dal dominio di training.

---

## ⏱️ Min 120-150

Certamente! Ecco degli appunti universitari dettagliati basati sul segmento video riguardante l'algoritmo PageRank:

# Note Accademiche: L'Algoritmo PageRank e le Sue Applicazioni

## 1. Introduzione al PageRank
Il **PageRank** è un algoritmo fondamentale sviluppato originariamente dai fondatori di Google per classificare l'importanza relativa delle pagine web all'interno di un grafo. In questo contesto, il web è visto come un grafo orientato dove:
-   **Nodi:** Rappresentano le pagine web (es. articoli di Wikipedia).
-   **Archi (Edge):** Rappresentano i collegamenti ipertestuali (hyperlink) tra le pagine.

L'idea principale è che una pagina sia tanto più importante quanti più link riceve da altre pagine a loro volta importanti.

## 2. Descrizione del Dataset
L'esercitazione utilizza un dataset derivato dalla categoria di Wikipedia **"Machine Learning"**.
-   **nodes.csv:** Elenco degli articoli (nodi del grafo).
-   **edges.csv:** Coppie sorgente-destinazione che indicano la presenza di un link (archi orientati).
-   **traffic.csv:** Statistiche reali sulle visualizzazioni delle pagine (traffico web), utilizzate per validare l'efficacia teorica dell'algoritmo.

## 3. Costruzione del Grafo e della Matrice di Transizione
Per l'implementazione in Python, viene utilizzata la libreria `NetworkX`.

### 3.1 Grafo Orientato
Si crea un grafo orientato (`nx.DiGraph`) perché la relazione di link non è necessariamente simmetrica: se la pagina A linka B, non è detto che B linki A.

### 3.2 Matrice di Transizione ($M$)
La matrice di transizione $M$ rappresenta la probabilità di spostarsi da una pagina all'altra seguendo un link casuale.
-   Se una pagina $u$ ha $k$ link in uscita, la probabilità di cliccare su uno specifico link è $1/k$.
-   Formalmente, se esiste un arco da $u$ a $v$, allora $M_{vu} = 1 / \text{out\_degree}(u)$.
-   Questa è una **matrice stocastica** per colonna (la somma di ogni colonna è 1, assumendo che non ci siano "dangling nodes" o pozzi senza link in uscita).

## 4. Il Fattore di Smorzamento (Damping Factor)
Per modellare meglio il comportamento di un utente reale, che non segue i link all'infinito ma a volte salta a una pagina casuale, si introduce il **damping factor** ($d$, tipicamente fissato a $0.85$).

Si definisce quindi la **Google Matrix** ($G$):
$$G = dM + \frac{1-d}{N} \mathbf{1}_{N \times N}$$
Dove:
-   $N$ è il numero totale di nodi.
-   $\mathbf{1}_{N \times N}$ è una matrice di soli 1.
-   L'utente segue un link con probabilità $d$ e salta a una pagina a caso con probabilità $1-d$.

## 5. Implementazione del PageRank
Esistono due approcci principali presentati nel video:
1.  **NetworkX built-in:** `nx.pagerank(graph, alpha=0.85)`.
2.  **Power Iteration (Metodo delle potenze):** Un approccio iterativo manuale per trovare l'autovettore principale della matrice $G$.

### Algoritmo Power Iteration:
1.  Inizializzare un vettore di probabilità $p$ uniforme ($1/N$ per ogni nodo).
2.  Iterare finché non si raggiunge la convergenza (la differenza tra due iterazioni successive è sotto una tolleranza $\epsilon$):
    -   $p_{next} = G \cdot p$
    -   Normalizzare $p_{next}$ (opzionale se $G$ è perfettamente stocastica, ma utile per stabilità numerica).
3.  Il vettore risultante $p$ contiene i punteggi PageRank dei nodi.

## 6. Visualizzazione del Grafo
La visualizzazione viene effettuata tramite `matplotlib` e `NetworkX` utilizzando un **layout a molla (spring layout)**.
-   Il layout simula un sistema fisico dove i nodi sono masse collegate da molle (gli archi).
-   I nodi con punteggio PageRank più alto vengono visualizzati con dimensioni maggiori.
-   I nodi più importanti e densamente connessi tendono a raggrupparsi (cluster) al centro del grafo.

## 7. Analisi dei Risultati e Correlazione con il Traffico Reale
L'obiettivo finale è confrontare l'importanza teorica calcolata (PageRank) con l'importanza reale (traffico web di Wikipedia).
-   Utilizzando una scala logaritmica, si osserva una tendenza positiva tra le due variabili.
-   Il calcolo del **coefficiente di correlazione** mostra un valore di circa **0.66 (66%)**.
-   **Conclusione:** Sebbene il traffico web sia influenzato da molti fattori complessi, l'algoritmo PageRank basato sulla sola struttura dei link fornisce una previsione ragionevolmente accurata e significativa dell'importanza delle pagine, superando di gran lunga una distribuzione casuale (che ha correlazione prossima a zero).

---

