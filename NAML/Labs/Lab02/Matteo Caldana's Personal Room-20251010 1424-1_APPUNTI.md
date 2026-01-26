# 📝 Appunti: Matteo Caldana's Personal Room-20251010 1424-1
**Modello:** gemini-3-flash-preview | **Data:** 23/01/2026

---

## ⏱️ Min 0-30

Ecco degli appunti universitari dettagliati basati sulla lezione fornita nel video.

---

# Appunti di Analisi Numerica per il Machine Learning: Principal Component Analysis (PCA)

## 1. Definizioni Fondamentali e Intuizione Geometrica

La **Principal Component Analysis (PCA)** è una tecnica di riduzione della dimensionalità che mira a ruotare i dati verso direzioni che massimizzano la varianza spiegata.

### 1.1 Direzioni Principali (Principal Directions - PD)
Le **Direzioni Principali** sono definite come le direzioni dei **vettori propri (eigenvectors)** della matrice di covarianza $C$.
*   **Importanza:** Queste direzioni rappresentano gli assi lungo i quali i dati mostrano la massima varianza. Identificare queste direzioni è il cuore dell'algoritmo PCA.

### 1.2 Componenti Principali (Principal Components - PC)
Le **Componenti Principali** sono le proiezioni dei dati originali sulle direzioni principali.
*   **Operazione Geometrica:** La proiezione corrisponde al **prodotto scalare (o prodotto interno)** tra i dati e i vettori propri che definiscono le direzioni principali.

---

## 2. Convenzioni Notazionali e Assunzioni

Per procedere con la derivazione matematica, è fondamentale stabilire una convenzione sulla forma della matrice dei dati.

### 2.1 Convenzione Adottata nel Laboratorio
Sia $X \in \mathbb{R}^{m \times n}$ la matrice dei dati, dove:
*   $m$: Numero di **features** (es. il numero di pixel in un'immagine).
*   $n$: Numero di **campioni (samples)**.
*   **Struttura:** Solitamente $n \gg m$ (molti campioni, meno feature, sebbene le feature possano essere comunque numerose). In questa convenzione, ogni colonna di $X$ rappresenta un campione.

### 2.2 Assunzione di Centralità
Si assume che i dati in $X$ siano **centrati**, ovvero che abbiano **media zero**:
$$\mu = \frac{1}{n} \sum_{i=1}^n x_i = 0$$
Se i dati non sono centrati, è necessario calcolare la media e sottrarla da ogni campione prima di applicare la PCA: $X_{centrata} = X - \mu$.

---

## 3. Derivazione Tramite SVD (Singular Value Decomposition)

### 3.1 La Matrice di Covarianza
Data la matrice centrata $X$, la matrice di covarianza $C$ è definita come:
$$C = \frac{XX^T}{n-1}$$

### 3.2 Relazione con la SVD
Applichiamo la decomposizione ai valori singolari (SVD) alla matrice dei dati $X$:
$$X = U\Sigma V^T$$
Sostituendo nella formula della matrice di covarianza:
$$C = \frac{(U\Sigma V^T)(U\Sigma V^T)^T}{n-1} = \frac{U\Sigma V^T V\Sigma^T U^T}{n-1}$$
Poiché $V$ è una matrice ortogonale ($V^T V = I$) e $\Sigma$ è pseudo-diagonale:
$$C = U \left( \frac{\Sigma^2}{n-1} \right) U^T$$
Questa espressione rappresenta la **diagonalizzazione** della matrice di covarianza $C$.

**Conclusione:**
*   Le **colonne di $U$** sono i vettori propri di $C$ e corrispondono alle **Direzioni Principali**.
*   I valori sulla diagonale di $\frac{\Sigma^2}{n-1}$ sono gli autovalori di $C$, che rappresentano la varianza lungo ogni direzione principale.

### 3.3 Calcolo delle Componenti Principali
Le componenti principali si ottengono proiettando $X$ sulle direzioni $U$:
$$PC = U^T X$$
Operativamente, l'elemento $(i, j)$ di questa matrice è il prodotto scalare tra la $i$-esima direzione principale (riga di $U^T$) e il $j$-esimo campione (colonna di $X$).

---

## 4. Convenzione Alternativa (Campioni sulle Righe)

Se la matrice dei dati è definita come $X \in \mathbb{R}^{n \times m}$ (campioni sulle righe, feature sulle colonne):
*   La matrice di covarianza diventa $C = \frac{X^T X}{n-1}$.
*   Tramite SVD ($X = U\Sigma V^T$), si ottiene $C = \frac{1}{n-1} V\Sigma^2 V^T$.
*   In questo caso, le **Direzioni Principali sono le colonne di $V$**.
*   **Nota bene:** Entrambe le convenzioni sono valide; è sufficiente scambiare il ruolo di $U$ e $V$ a seconda della forma della matrice $X$.

---

## 5. Esercitazione Pratica: Interpretazione Geometrica

L'obiettivo dell'esercizio in Python (Google Colab) è simulare dei dati "sporchi" per mostrare come la PCA possa recuperare la struttura originale.

### 5.1 Fasi dell'Esercitazione
1.  **Generazione dati:** Si generano 1000 punti casuali seguendo una distribuzione Gaussiana normale indipendente in 2D (una "nuvola" circolare centrata nell'origine).
2.  **Trasformazione Geometrica:** Si applica una trasformazione nota $x_i = Ar_i + b$, dove:
    *   $A$: Matrice che combina **rotazione** (tramite vettori ortogonali $z_1, z_2$) e **dilatazione** (scaling tramite costanti $\rho_1, \rho_2$).
    *   $b$: Vettore di **traslazione**.
    *   Il risultato è una nuvola di punti di forma ovale, ruotata e spostata dall'origine.
3.  **Applicazione PCA:** Si utilizza la PCA per trovare le direzioni di massima varianza senza conoscere a priori la trasformazione $A$.
4.  **Verifica:** Si dimostra che le direzioni identificate dalla PCA corrispondono alle direzioni di rotazione e dilatazione applicate inizialmente.

### 5.2 Note di Implementazione (Python/NumPy)
*   **Broadcasting:** Per sottrarre il vettore media o aggiungere il vettore traslazione $b$ a una matrice di campioni, NumPy utilizza il broadcasting. Se $b$ è un vettore 1D, potrebbe essere necessario aggiungere una dimensione tramite `None` o `np.newaxis` (es. `b[:, None]`) per renderlo una matrice colonna $2 \times 1$ e permettere l'operazione elemento per elemento sulla matrice $2 \times n$.
*   **Visualizzazione:** L'uso di `plt.axis('equal')` in Matplotlib è fondamentale per visualizzare correttamente le proporzioni geometriche (evitando distorsioni tra gli assi x e y).
*   **SVD in NumPy:** Si utilizza la funzione `np.linalg.svd`.

---

---

## ⏱️ Min 30-60

Certamente! Ecco degli appunti universitari dettagliati basati sul segmento video fornito, riguardante l'implementazione della **Analisi delle Componenti Principali (PCA)** in Python.

---

# Appunti di Data Science: Analisi delle Componenti Principali (PCA)

## 1. Introduzione alla PCA
L'Analisi delle Componenti Principali (PCA) è una tecnica statistica utilizzata per la riduzione della dimensionalità e per identificare le direzioni di massima varianza all'interno di un dataset. In termini geometrici, la PCA ruota i dati in un nuovo sistema di coordinate in cui i nuovi assi (componenti principali) catturano la maggiore dispersione possibile dei dati.

## 2. Formulazione Matematica
Per eseguire la PCA sulla matrice dei dati $X \in \mathbb{R}^{d \times n}$ (dove $d$ è la dimensione e $n$ il numero di campioni), seguiamo i seguenti passaggi:

1.  **Calcolo della media campionaria ($\mu$):**
    $$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$$
2.  **Centratura dei dati ($\bar{X}$):**
    Si sottrae la media da ogni colonna della matrice originale:
    $$\bar{X} = X - \mu [1, \dots, 1]$$
3.  **Decomposizione ai Valori Singolari (SVD):**
    Si decompone la matrice centrata:
    $$\bar{X} = U \Sigma V^T$$
    *   $U$: Matrice dei vettori singolari sinistri (le colonne di $U$ sono le direzioni principali).
    *   $\Sigma$: Matrice diagonale dei valori singolari ($\sigma_i$).
    *   $V^T$: Matrice dei vettori singolari destri.

## 3. Implementazione in Python (NumPy)

### 3.1 Calcolo della Media e Centratura
Il video evidenzia l'importanza di gestire correttamente le dimensioni degli array per sfruttare il **broadcasting** di NumPy.

```python
import numpy as np

# Supponiamo che X sia una matrice (2, 1000)
# Calcoliamo la media lungo le righe (axis=1)
X_mean = np.mean(X, axis=1) # Output shape: (2,)

# Per sottrarre X_mean da X, dobbiamo espandere le dimensioni di X_mean
# in modo che diventi (2, 1) per il broadcasting
X_bar = X - X_mean[:, None] # None aggiunge una dimensione (np.newaxis)
```

### 3.2 Esecuzione della SVD
Si utilizza il sottomodulo `linalg` di NumPy.

```python
# SVD con full_matrices=False per efficienza computazionale (Economy SVD)
U, S, VT = np.linalg.svd(X_bar, full_matrices=False)
```

### 3.3 Estrazione delle direzioni e riscalamento
Le direzioni principali sono date dalle colonne di $U$. Per visualizzarle correttamente in relazione alla varianza dei dati, i vettori vengono spesso riscalati utilizzando i valori singolari.

L'entità della varianza (deviazione standard campionaria) lungo una direzione è legata al valore singolare $s_i$ dalla formula:
$$\text{std\_dev}_i = \frac{s_i}{\sqrt{n-1}}$$

```python
n_points = X.shape[1]
# Calcolo del fattore di riscalamento
r = S / np.sqrt(n_points - 1)

# Estrazione dei primi due vettori principali
u1 = U[:, 0]
u2 = U[:, 1]
```

## 4. Visualizzazione e Interpretazione
Nel video, viene mostrato come plottare i vettori principali sopra lo scatter plot dei dati originali.

*   **Frecce Nere/Rosse:** Rappresentano le componenti principali stimate.
*   **Confronto:** Il video mostra che le direzioni identificate dalla PCA (in rosso) coincidono quasi perfettamente con le direzioni reali della trasformazione geometrica originale (in nero), dimostrando l'efficacia della PCA nel recuperare la struttura sottostante dei dati basandosi esclusivamente sulla loro distribuzione statistica.
*   **Significato:** La direzione più lunga corrisponde alla **prima componente principale**, ovvero l'asse lungo il quale i dati mostrano la massima varianza.

## 5. Termini Tecnici Chiave
*   **SVD (Singular Value Decomposition):** Metodo numerico per decomporre una matrice, fondamentale per la PCA.
*   **Broadcasting:** Funzionalità di NumPy che permette operazioni aritmetiche tra array di forme diverse.
*   **Varianza Spiegata:** La quantità di informazione (dispersione) catturata da ogni componente principale.
*   **Centratura dei dati:** Processo di traslazione del dataset affinché la media sia l'origine (0,0).

---

---

## ⏱️ Min 60-90

Ecco degli appunti universitari dettagliati basati sul segmento video della lezione sull'Analisi delle Componenti Principali (PCA) e il riconoscimento di cifre scritte a mano.

---

# Appunti di Analisi Numerica per il Machine Learning
## Lezione: PCA e Applicazione al Dataset MNIST

### 1. Riepilogo PCA su Dati 2D
L'obiettivo iniziale della sessione pratica è visualizzare i risultati di una PCA calcolata tramite **SVD (Singular Value Decomposition)** su un set di dati bidimensionale sintetico.

*   **Direzioni di Trasformazione**: Vengono confrontate le direzioni reali ($z_1, z_2$) con quelle stimate ($u_1, u_2$).
    *   **Ambiguità di Segno**: È fondamentale notare che la PCA non determina l'orientamento assoluto della direzione. Se si moltiplica un vettore direzione per $-1$, esso rappresenta ancora la stessa direzione di massima varianza. Pertanto, i vettori stimati potrebbero differire da quelli reali solo per il segno.
*   **Calcolo delle Componenti Principali ($\Phi$)**: 
    $$\Phi = U^T \bar{X}$$
    Dove $U^T$ è la trasposta della matrice dei vettori singolari sinistri e $\bar{X}$ è la matrice dei dati a media zero. Ogni colonna di $\Phi$ rappresenta la proiezione dei dati sulle direzioni principali.
*   **Visualizzazione e Riscalamento**:
    *   Proiettando i dati sulle prime due componenti principali, si ottiene una rotazione del dataset originale.
    *   **Riscalamento**: Dividendo le componenti per la deviazione standard stimata ($r$), si ottiene una "nuvola rotonda" di punti, indicativa di una distribuzione quasi normale standardizzata.
    *   *Nota tecnica*: Usare `plt.axis('equal')` in Matplotlib è essenziale per evitare distorsioni visive e mantenere le proporzioni corrette tra gli assi.

---

### 2. Dataset MNIST e Best Practices di Machine Learning
La lezione si sposta sull'applicazione della PCA a un dataset reale: **MNIST (Handwritten Digits)**.

#### Best Practices: Training vs. Testing
*   **Suddivisione dei Dati**: In ogni algoritmo di ML, è imperativo avere almeno due dataset:
    1.  **Training Set**: Utilizzato per addestrare l'algoritmo.
    2.  **Testing Set**: Dati mai visti dall'algoritmo, usati per verificare le prestazioni in scenari reali.
*   **Overfitting**: Se l'algoritmo impara troppo bene le "idiosincrasie" (rumore o dettagli specifici) del training set, non riuscirà a generalizzare sui nuovi dati. Il test set serve a prevenire questo fenomeno.

#### Caratteristiche del Dataset MNIST
*   **Dimensioni**: Il file CSV caricato contiene 20.000 campioni e 785 colonne.
*   **Struttura delle colonne**:
    *   La prima colonna (indice 0) rappresenta le **Labels** (la cifra reale da 0 a 9).
    *   Le restanti 784 colonne rappresentano i **Pixel** dell'immagine.
*   **Formato Immagine**: Ogni immagine è un vettore "appiattito" (flattened) di 784 elementi. Poiché $\sqrt{784} = 28$, l'immagine originale è una griglia di **28x28 pixel**.

---

### 3. Esplorazione e Pre-elaborazione dei Dati
Prima dell'analisi, è necessario manipolare i dati per adattarli alle convenzioni algebriche.

*   **Trasposizione**: Per convenzione matematica nella lezione, ogni colonna della matrice dei dati deve rappresentare un singolo campione.
*   **Visualizzazione**: Per visualizzare i dati come immagini con `plt.imshow`, occorre eseguire un **reshape** del vettore da 784 elementi alla matrice 28x28.
*   **Controllo della Corruzione dei Dati**: La PCA e la SVD sono strumenti molto potenti per identificare dati corrotti o **outlier**. Proiettando i dati sulle direzioni principali, i campioni che non appartengono al cluster principale (dati errati o anomalie) appaiono molto distanti dagli altri.

---

### 4. Tecniche di Filtraggio: Bit-Masking
Per compiti specifici (es. distinguere solo tra le cifre 0 e 9), è necessario filtrare il dataset.

*   **Definizione di Bit-mask**: Si crea un array di valori booleani (`True`/`False`) basato su una condizione logica applicata alle label.
    *   Esempio: `labels_full == 9` genera un array che è `True` solo dove la cifra è un 9.
*   **Utilizzo con NumPy**: Passando questa maschera come indice alle colonne della matrice dei dati, NumPy estrae automaticamente solo i campioni desiderati.
    *   *Vantaggio*: Questo approccio permette computazioni molto veloci ed efficienti rispetto ai cicli `for`.

---

### 5. Task Accademico: Classificazione Binaria (0 e 9)
Il compito assegnato consiste nell'utilizzare la PCA per semplificare il dataset e prepararlo per una classificazione.

**Passaggi richiesti**:
1.  **Filtraggio**: Estrarre solo le immagini corrispondenti ai numeri 0 e 9.
2.  **Media Zero**: Calcolare e sottrarre l'immagine media (centratura dei dati).
3.  **PCA tramite SVD**: Calcolare i valori singolari e i vettori singolari.
4.  **Analisi della Varianza Spiegata**: Graficare la frazione cumulativa dei valori singolari per determinare quante componenti sono necessarie per rappresentare i dati con precisione.
5.  **Visualizzazione degli Assi Principali**: Visualizzare le prime 30 colonne di $U$ (gli "autovoltaggi" o basi principali del dataset).
6.  **Scatter Plot**: Creare un grafico a dispersione delle prime due componenti principali raggruppate per label (0 vs 9) per osservare se i cluster sono separabili.

---

### Glossario Tecnico
*   **SVD (Singular Value Decomposition)**: Tecnica di fattorizzazione di matrici usata per calcolare le componenti della PCA.
*   **Explained Variance (Varianza Spiegata)**: Misura di quanta informazione (variabilità) del dataset originale viene mantenuta riducendo le dimensioni.
*   **Outlier**: Punto di dato che differisce significativamente dalle altre osservazioni; può indicare un errore sperimentale o una rarità.
*   **Reshape**: Operazione che cambia la forma di un array senza modificarne i dati (es. da vettore a matrice).

---

## ⏱️ Min 90-120

Ecco degli appunti universitari dettagliati basati sulla sessione di laboratorio di Leonardo riguardante il riconoscimento di cifre scritte a mano tramite PCA (Principal Component Analysis) e SVD (Singular Value Decomposition).

---

# Appunti di Laboratorio: Analisi e Riconoscimento di Cifre Manoscritte tramite PCA

## 1. Obiettivo dell'Esercitazione
L'obiettivo è analizzare un dataset di cifre manoscritte (presumibilmente MNIST) concentrandosi sulle classi **0** e **9**. Si utilizzerà la decomposizione ai valori singolari (SVD) per estrarre le direzioni principali (PCA) e comprendere come i dati possano essere compressi o ricostruiti tramite un numero ridotto di componenti.

## 2. Pre-elaborazione e Filtraggio dei Dati
Il dataset originale contiene cifre da 0 a 9. Per isolare solo gli '0' e i '9', si utilizza la tecnica del **bit-masking** (mascheramento booleano).

### Codice e Logica:
*   **Definizione dei target:** `digits = [0, 9]`
*   **Creazione della maschera:** Si genera un vettore booleano della stessa lunghezza delle etichette (`labels_full`). La maschera è vera se l'etichetta è 0 OPPURE 9.
    *   Metodo 1: `np.logical_or(labels_full == digits[0], labels_full == digits[1])`
    *   Metodo 2 (Sintassi abbreviata): `(labels_full == digits[0]) | (labels_full == digits[1])`
*   **Applicazione:** La maschera viene usata per indicizzare la matrice dei pixel (`A_full`) e il vettore delle etichette, ottenendo i sottoinsiemi filtrati `A` e `labels`.

## 3. Visualizzazione dei Dati e Media
### Visualizzazione (Plotting)
Le immagini sono memorizzate come vettori colonna di pixel (es. 784 elementi per immagini 28x28). Per visualizzarle con `matplotlib`, è necessario effettuare un **reshape** a (28, 28).
*   Si utilizza `plt.subplots` per creare una griglia (es. 3 righe, 10 colonne).
*   La funzione `imshow` con `cmap='gray'` visualizza l'intensità dei pixel.

### Calcolo dell'Immagine Media
Il primo passo della PCA è il calcolo del valore medio per ogni pixel attraverso tutti i campioni.
*   **Formula:** $\bar{A} = \frac{1}{n} \sum_{i=1}^{n} A_i$
*   **Codice:** `A_mean = np.mean(A, axis=1)`
*   **Interpretazione:** L'immagine media visualizzata mostra una sovrapposizione sfocata: si riconosce la forma circolare dello '0' e un'area più sbiadita in basso a destra che rappresenta il "gambo" del '9'.

## 4. Decomposizione ai Valori Singolari (SVD)
Per eseguire la PCA, i dati devono essere a **media nulla** (zero-mean data).
1.  **Centratura:** `A_bar = A - A_mean[:, np.newaxis]` (si utilizza il broadcasting di NumPy per sottrarre il vettore media da ogni colonna della matrice dei dati).
2.  **SVD:** Si decompone la matrice centrata $A\_bar$ in $U \Sigma V^T$.
    *   `U, s, VT = np.linalg.svd(A_bar, full_matrices=False)`
    *   `U`: Contiene gli assi principali (eigen-digits).
    *   `s`: Contiene i valori singolari $\sigma_i$.

## 5. Analisi delle Componenti e della Varianza
Vengono prodotti tre grafici per analizzare l'importanza delle componenti:
1.  **Valori Singolari ($\sigma_k$):** Visualizzati in scala semi-logaritmica. Si osserva un crollo repentino dei valori verso la fine (attorno a $10^{-11}$, vicino alla precisione di macchina o *machine epsilon*).
2.  **Frazione Cumulata dei Valori Singolari:** $\frac{\sum \sigma_i}{\sum \sigma_{total}}$
3.  **Frazione della Varianza Spiegata:** $\frac{\sum \sigma_i^2}{\sum \sigma_{total}^2}$

**Interpretazione del crollo dei valori singolari:**
Il drastico calo indica che molte direzioni non aggiungono informazioni. Questo accade perché molti pixel (specialmente lungo i bordi delle immagini 28x28) sono sempre neri (valore 0) in tutti i campioni del dataset filtrato. Questi pixel hanno varianza nulla e non contribuiscono alla ricostruzione della forma.

## 6. Visualizzazione degli Assi Principali (Eigen-digits)
Le colonne della matrice $U$ rappresentano le direzioni principali.
*   **Prime componenti:** Catturano le caratteristiche macroscopiche. La prima componente spesso evidenzia la differenza fondamentale tra uno '0' e un '9'.
*   **Componenti intermedie (es. indice 100-130):** Catturano dettagli sempre più fini e variazioni locali.
*   **Ultime componenti (es. indice > 600):** Rappresentano essenzialmente rumore o pixel di sfondo costanti, confermando l'analisi dei valori singolari.

## 7. Calcolo delle Componenti Principali (Proiezione)
Per rappresentare un'immagine nello spazio ridotto della PCA, proiettiamo i dati centrati sugli assi principali.
*   **Logica matematica:** La proiezione $\Phi$ è data dal prodotto scalare tra la trasposta di $U$ e la matrice dei dati centrata: $\Phi = U^T \bar{A}$.
*   **Verifica:** Il docente dimostra che calcolare `np.inner` (prodotto scalare) tra un singolo campione e una singola colonna di $U$ restituisce lo stesso valore presente nella corrispondente cella della matrice di proiezione totale calcolata tramite prodotto matriciale (`@`).

---
**Terminologia Tecnica Chiave:**
*   **SVD (Singular Value Decomposition):** Fattorizzazione di una matrice.
*   **PCA (Principal Component Analysis):** Tecnica di riduzione della dimensionalità.
*   **Eigen-digits:** Le direzioni principali visualizzate come immagini.
*   **Zero-mean data:** Dati a cui è stata sottratta la media.
*   **Machine Epsilon:** Il più piccolo numero che, aggiunto a 1, produce un risultato diverso da 1 (limite della precisione numerica).
*   **Broadcasting:** Capacità di NumPy di eseguire operazioni su array di dimensioni diverse.

---

## ⏱️ Min 120-150

Ecco degli appunti universitari dettagliati basati sul segmento video riguardante il riconoscimento della scrittura a mano tramite PCA.

---

# Appunti di Laboratorio: Riconoscimento di Cifre Manoscritte tramite PCA

## 1. Visualizzazione dei Componenti Principali (PCA)
L'obiettivo iniziale è visualizzare come le immagini delle cifre manoscritte (nello specifico '0' e '9') si distribuiscono nello spazio dei primi due componenti principali.

### Metodologia di Proiezione:
*   **Approccio Naive (Lento):** Utilizzo di un ciclo `for` per calcolare il prodotto interno (proiezione) di ogni singola immagine sui primi due assi principali.
*   **Approccio Ottimizzato (Vettorizzato):** È preferibile utilizzare le operazioni tra matrici di NumPy per proiettare l'intero dataset simultaneamente, migliorando drasticamente le prestazioni.
*   **Scatter Plot:** Si proiettano i dati in 2D usando il primo componente principale (PC1) sull'asse x e il secondo (PC2) sull'asse y. 
    *   Le cifre '0' vengono colorate in un modo (es. giallo) e le cifre '9' in un altro (es. viola).
    *   **Risultato:** Si osserva una netta separazione spaziale tra i due cluster, indicando che la PCA è efficace nel ridurre la dimensionalità mantenendo le caratteristiche discriminanti.

## 2. Definizione di un Classificatore Lineare Semplice
Dato che i due cluster (0 e 9) sono ben separati lungo l'asse del primo componente principale, è possibile implementare un classificatore estremamente semplice basato su una **soglia (threshold)**.

### Logica del Classificatore:
1.  Si osserva visivamente lo scatter plot per identificare un valore di PC1 che separi i due gruppi.
2.  Nel video, viene scelta una soglia pari a **0.0**.
3.  **Regola di Decisione:**
    *   Se il valore del primo componente principale di un campione è inferiore alla soglia, viene classificato come una cifra (es. '0').
    *   Se è superiore, viene classificato come l'altra (es. '9').
4.  Visualizzazione: Viene tracciata una linea verticale (`plt.axvline`) in corrispondenza della soglia per mostrare il confine di decisione.

## 3. Validazione sul Test Dataset
Per verificare la bontà del classificatore, questo deve essere applicato a dati che il modello non ha mai "visto" durante la fase di analisi (Training).

### Passaggi Cruciali per il Test:
*   **Caricamento dati:** Importazione del dataset di test (`mnist_test.csv`).
*   **Filtro:** Estrazione solo dei campioni corrispondenti alle cifre '0' e '9'.
*   **Trasformazione (Zero Data Leakage):** Questo è il punto più critico. Per proiettare i dati di test:
    1.  Si deve utilizzare la **media calcolata sul training set** (`A_mean`).
    2.  Si devono utilizzare le **direzioni dei componenti principali (matrice U)** ottenute dalla SVD del training set.
    *   *Nota tecnica:* Non si effettua una nuova SVD sui dati di test. Si "trasformano" i dati di test nello spazio definito dai dati di training: `Phi_test = U.T @ (A_test - A_mean)`.

## 4. Valutazione delle Performance
Una volta ottenute le predizioni sul test set, si procede all'analisi quantitativa.

### Metriche calcolate:
*   **Accuracy (Accuratezza):** Calcolata come il rapporto tra le previsioni corrette (veri 0 + veri 9) e il numero totale di campioni. Nel video si ottiene un'accuratezza del **95.2%**.
*   **Matrice di Confusione (Confusion Matrix):** Strumento fondamentale per analizzare gli errori.
    *   **Veri Positivi/Negativi:** Campioni classificati correttamente.
    *   **Falsi Positivi/Negativi:** Campioni classificati erroneamente (es. uno '0' scambiato per un '9').
    *   **Bilanciamento:** È importante che gli errori siano distribuiti in modo simile tra le classi. Una matrice di confusione bilanciata indica che il classificatore non è polarizzato verso una cifra specifica.

### Strumenti Python:
*   `numpy.where`: Utilizzato per generare il vettore delle etichette predette in base alla soglia.
*   `sklearn.metrics.ConfusionMatrixDisplay`: Utilizzato per una visualizzazione professionale e normalizzata dei risultati della classificazione.

---
**Conclusione:** Nonostante la semplicità del metodo (una singola soglia lineare su un solo componente principale), l'approccio PCA si dimostra estremamente potente per il riconoscimento di pattern in dataset ad alta dimensionalità come le immagini MNIST.

---

## ⏱️ Min 150-180

Ecco degli appunti universitari dettagliati basati sul segmento video fornito, che si concentra sulla valutazione di un modello di riconoscimento della scrittura a mano (cifre 0 e 9).

---

# Appunti sulla Valutazione del Modello: Riconoscimento Cifre Manoscritte

## 1. Introduzione alla Matrice di Confusione
L'obiettivo principale di questa sezione è calcolare e interpretare una **matrice di confusione** per un classificatore binario (che distingue tra le cifre '0' e '9'). La matrice di confusione è uno strumento fondamentale per valutare le prestazioni di un modello di classificazione, permettendo di visualizzare non solo l'accuratezza globale, ma anche i tipi specifici di errori commessi.

### Componenti della Matrice (nel contesto 0 vs 9):
*   **True 0 (Veri Positivi per 0):** Quanti '0' sono stati correttamente classificati come '0'.
*   **True 9 (Veri Positivi per 9):** Quanti '9' sono stati correttamente classificati come '9'.
*   **False 0 (Falsi Positivi per 0 / Falsi Negativi per 9):** Quanti '9' sono stati erroneamente classificati come '0'.
*   **False 9 (Falsi Positivi per 9 / Falsi Negativi per 0):** Quanti '0' sono stati erroneamente classificati come '9'.

## 2. Implementazione in Python (NumPy)
Il codice mostrato utilizza la libreria NumPy per calcolare manualmente queste metriche confrontando le etichette reali (`labels_test`) con quelle predette dal modello (`labels_predict`).

### Logica di Predizione
Viene utilizzata la funzione `np.where` per assegnare le classi in base a una soglia applicata alla prima componente principale (PC1):
```python
labels_predict = np.where(PC_1 > threshold, digits[0], digits[1])
```
*   **Funzionamento di `np.where`:** Restituisce il valore di `digits[0]` se la condizione (`PC_1 > threshold`) è vera, altrimenti restituisce `digits[1]`.

### Calcolo delle Metriche
```python
true_0 = np.sum((labels_test == 0) & (labels_predict == 0))
true_9 = np.sum((labels_test == 9) & (labels_predict == 9))
false_0 = np.sum((labels_test == 9) & (labels_predict == 0))
false_9 = np.sum((labels_test == 0) & (labels_predict == 9))
```
Dall'output del video, i risultati ottenuti sono:
*   **True 0:** 927
*   **True 9:** 979
*   **False 9:** 53
*   **False 0:** 30

## 3. Calcolo dell'Accuratezza (Accuracy)
L'accuratezza rappresenta la frazione di predizioni corrette rispetto al totale dei campioni.

**Formula:**
$$Accuracy = \frac{True\_0 + True\_9}{True\_0 + True\_9 + False\_0 + False\_9}$$

Nel video, il risultato calcolato è circa **0.958 (95.8%)**.

## 4. Alternative Standard: Scikit-learn
Sebbene il calcolo manuale sia utile a fini didattici, nella pratica professionale si utilizza il modulo `metrics` della libreria **Scikit-learn** (`sklearn`), che offre funzioni ottimizzate e compatte per generare report di classificazione e matrici di confusione:
```python
from sklearn import metrics
```

## 5. Esercitazioni Future e Compiti
Il docente introduce un terzo notebook dedicato a un caso d'uso reale: la **classificazione di dati oncologici (Cancer Dataset)**. 
*   **Obiettivo:** Classificare i pazienti (es. tumore maligno vs benigno) utilizzando logiche simili a quelle applicate per le cifre.
*   **Nota:** Questo esercizio è lasciato come compito a casa per consolidare la comprensione della classificazione su dataset reali e clinicamente rilevanti.

---

---

