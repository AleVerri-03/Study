# 📝 Appunti: Matteo Caldana's Personal Room-20251031 1510-1
**Modello:** gemini-3-flash-preview | **Data:** 23/01/2026

---

## ⏱️ Min 0-30

Ecco dei dettagliati appunti universitari basati sul segmento video fornito:

---

# Appunti del Laboratorio: Completamento di Matrici e Sistemi di Raccomandazione

## 1. Obiettivi della Sessione di Laboratorio
La sessione è divisa in due parti principali:
*   **Parte 1:** Implementazione dell'algoritmo **Singular Value Thresholding (SVT)** applicato al completamento di matrici.
*   **Parte 2:** Introduzione a **JAX**, una libreria moderna e potente per la differenziazione automatica, ampiamente utilizzata nella ricerca sul machine learning.

## 2. Il Problema: Sistemi di Raccomandazione
Il caso di studio principale riguarda la previsione dei rating mancanti in un dataset di film (simile alla celebre *Netflix Prize*).
*   **Contesto:** Grandi aziende come Netflix o Amazon hanno milioni di utenti e migliaia di prodotti.
*   **Dati Sparsi:** La maggior parte degli utenti valuta solo una piccola frazione del contenuto disponibile. La matrice risultante (Utenti x Film) è estremamente "sparsa" (piena di buchi).
*   **Obiettivo:** Prevedere i rating mancanti (punti interrogativi nella matrice) per fornire raccomandazioni migliori.
*   **Impatto economico:** Anche piccoli miglioramenti nella precisione (punti percentuali) si traducono in milioni di dollari di entrate grazie al maggiore coinvolgimento degli utenti.

## 3. Analisi del Dataset (MovieLens)
Il laboratorio utilizza il dataset **MovieLens**, che consiste in:
*   **100.000 rating** (su una scala da 1 a 5).
*   **943 utenti**.
*   **1.682 film**.
*   Ogni utente nel dataset ha valutato almeno 20 film.

### Formato del file (`movielens.csv`):
I dati sono strutturati come una lista separata da tabulazioni con le seguenti colonne:
1.  `user_id`: Identificativo univoco dell'utente.
2.  `item_id`: Identificativo univoco del film.
3.  `rating`: Il punteggio assegnato (1-5).
4.  `timestamp`: Informazione temporale sulla valutazione (non utilizzata in questa implementazione di base).

## 4. Implementazione in Python: Preparazione dei Dati

### Librerie Utilizzate
*   `pandas` (come `pd`): Per la lettura e la manipolazione dei dati tabulari.
*   `numpy` (come `np`): Per il calcolo numerico e la gestione degli array.
*   `matplotlib.pyplot` (come `plt`): Per la visualizzazione.
*   `scipy.sparse`: Per gestire in modo efficiente le matrici sparse (usando il formato `csr_matrix`).
*   `scipy.stats`: Per il calcolo del coefficiente di correlazione di Pearson.

### Caricamento e Ispezione
Il dataset viene caricato in un DataFrame Pandas. Per verificare la consistenza dei dati, vengono calcolati i valori univoci:
```python
n_people = np.unique(dataset.user_id).size
n_movies = np.unique(dataset.item_id).size
n_ratings = len(dataset)
```
*Risultato:* 943 persone, 1682 film, 100.000 rating.

### Shuffle dei Dati e Riproducibilità
È fondamentale mescolare i dati prima di dividerli in set di addestramento (train) e test.
*   **Perché fare lo shuffle?** I dati reali sono spesso ordinati cronologicamente (timestamp). Se prendessimo l'ultimo 20% come test senza mescolare, testeremmo il modello solo su rating "nuovi", perdendo l'entropia necessaria per una valutazione equa dei rating dei singoli utenti presenti in entrambe le fasi.
*   **Riproducibilità:** Si imposta un `random.seed(1)` in modo che i risultati siano identici ad ogni esecuzione.

## 5. Gestione degli Indici Sparsi
Un problema tecnico comune è la presenza di "salti" negli ID degli utenti o dei film (es. un ID mancante tra 1 e 1000). Questo creerebbe righe o colonne vuote non necessarie nella matrice densa.
*   **Soluzione:** Utilizzare `np.unique` con l'opzione `return_inverse=True` per rimappare gli ID originali in indici contigui (0, 1, 2...).

## 6. Divisione Train/Test e Costruzione della Matrice
I dati vengono divisi:
*   **Training Set:** 80% (80.000 rating).
*   **Test Set:** 20% (20.000 rating).

Viene quindi creata la matrice sparsa di addestramento $X$, dove l'entrata $X_{i,j}$ contiene il rating se la coppia (utente $i$, film $j$) è nel training set, altrimenti è $0$.

---
*Nota: Questi appunti coprono la fase di setup e la teoria del problema. La fase successiva riguarda l'implementazione pratica dell'algoritmo SVT e del sistema di raccomandazione basato sulla media (baseline).*

---

## ⏱️ Min 30-60

Ecco dei dettagliati appunti universitari basati sulla lezione video riguardante il completamento di matrici per i sistemi di raccomandazione.

---

# Appunti di Laboratorio: Sistemi di Raccomandazione e Matrix Completion

## 1. Introduzione
Il laboratorio si focalizza sull'implementazione di algoritmi per i sistemi di raccomandazione basati sulla tecnica del **completamento di matrici (Matrix Completion)**. L'obiettivo è predire le valutazioni (*ratings*) mancanti in una matrice utente-film, partendo da un set di dati noto.

## 2. Recommender System "Trivial" (Basale)
Prima di passare a modelli complessi, viene implementato un predittore banale (*trivial predictor*) che funge da baseline. Questo modello ipotizza che la miglior previsione per un utente sia semplicemente la sua valutazione media su tutti i film che ha già votato.

### 2.1 Calcolo della Media per Utente
Esistono due modi principali per calcolare la valutazione media $\bar{r}_i$ per ogni utente $i$ nel set di addestramento:

*   **Metodo tramite Matrice Densa (`X_full`):**
    Si sommano gli elementi di ogni riga e si divide per il numero di elementi diversi da zero (ovvero i film effettivamente votati).
    ```python
    avg_rating[i] = X_full[i, :].sum() / (X_full[i, :] > 0).sum()
    ```
*   **Metodo tramite Masking Booleano (più efficiente):**
    Si utilizzano i vettori sparsi delle valutazioni (`vals_train`) e degli indici degli utenti (`rows_train`).
    ```python
    avg_rating[i] = vals_train[rows_train == i].mean()
    ```
    Questo approccio estrae solo le valutazioni appartenenti all'utente $i$ e ne calcola la media direttamente.

### 2.2 Generazione del Vettore di Predizione
Una volta ottenute le medie per ogni utente, si costruisce il vettore `vals_trivial` per il test set. Per ogni coppia (utente, film) nel set di test, la predizione sarà la media calcolata per quell'utente, indipendentemente dal film.
```python
vals_trivial = avg_rating[rows_test]
```

## 3. Metriche di Valutazione
Le prestazioni dei modelli vengono misurate confrontando le predizioni con i valori reali del test set (`vals_test`).

1.  **RMSE (Root Mean Square Error):** Misura la deviazione standard degli errori di previsione. Un RMSE più basso indica una precisione maggiore.
    $$RMSE = \sqrt{\text{mean}((\hat{r}_{ij} - r_{ij})^2)}$$
2.  **Rho ($\rho$ - Correlazione di Pearson):** Misura la correlazione lineare tra i voti previsti e quelli reali. Un valore vicino a 1 indica un'ottima capacità del modello di seguire l'andamento dei gusti dell'utente.

**Risultati del Trivial Predictor:**
*   **RMSE:** ~1.043
*   **Rho:** ~0.384

## 4. Algoritmo Singular Value Truncation (SVT)
Il metodo SVT è un algoritmo iterativo utilizzato per recuperare matrici a basso rango (*low-rank matrix recovery*), basato sulla tecnica di soglia dei valori singolari.

### 4.1 Logica dell'Algoritmo
Il processo segue questi step iterativi:
1.  **Decomposizione SVD:** Si decompone la matrice corrente $A$ in $U \Sigma V^T$.
2.  **Truncation (Soglia):** Si applica un operatore di soglia (*shrinkage*) ai valori singolari $\sigma_i$ contenuti nella matrice diagonale $\Sigma$. Tutti i valori singolari inferiori a una certa `threshold` vengono impostati a zero.
3.  **Ricostruzione:** Si ricompone la matrice approssimata $\tilde{A} = U \Sigma_{trunc} V^T$.
4.  **Enforcement dei dati noti:** Per mantenere la coerenza con i dati reali, gli elementi di $\tilde{A}$ corrispondenti alle posizioni note del training set vengono sovrascritti con i valori originali.
5.  **Criterio di Arresto:** L'algoritmo termina quando il numero massimo di iterazioni è raggiunto o quando l'incremento (norma di Frobenius della differenza tra la matrice nuova e quella precedente) è inferiore a una tolleranza fissata (`increment_tol`).

### 4.2 Dettagli Implementativi
Il codice implementa un ciclo `for` che aggiorna la matrice $A$ ad ogni iterazione:
```python
for i in range(n_max_iter):
    A_old = A.copy()
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    
    # Troncamento dei valori singolari
    s[s < threshold] = 0.0
    A = (U * s) @ VT
    
    # Reinserimento valori del training set
    A[rows_train, cols_train] = vals_train
    
    # Calcolo dell'incremento per la convergenza
    increment = np.linalg.norm(A - A_old, ord='fro')
    
    # Calcolo metriche per monitoraggio
    vals_pred = A[rows_test, cols_test]
    # ... calcolo RMSE e Rho ...
    
    if increment < increment_tol:
        break
```

### 4.3 Osservazioni sul Tuning
*   **Threshold:** È il parametro critico. Nel laboratorio viene usato un valore di 100, calibrato sperimentalmente. Valori troppo alti portano a una perdita eccessiva di informazioni, valori troppo bassi non riducono a sufficienza il rumore/rango.
*   **Andamento Iterativo:** Monitorando RMSE e Rho ad ogni iterazione, si nota come l'algoritmo tenda a migliorare progressivamente la capacità predittiva rispetto al modello trivial basale.

---
*Note: Questi appunti coprono la parte di implementazione del predittore basale e la struttura fondamentale dell'algoritmo di Matrix Completion basato su SVD troncato.*

---

## ⏱️ Min 60-90

Ecco degli appunti universitari dettagliati basati sul segmento video fornito, organizzati per argomenti tecnici.

---

# Appunti di Calcolo Numerico e Machine Learning

## 1. Completamento di Matrici tramite Singular Value Truncation (SVT)

L'algoritmo **Singular Value Truncation (SVT)** è un metodo iterativo utilizzato per risolvere problemi di completamento di matrici, comune nei sistemi di raccomandazione e nel restauro di immagini.

### Metriche di Valutazione
Per misurare l'efficacia del modello, si utilizzano principalmente due metriche:
*   **RMSE (Root Mean Square Error):** L'obiettivo è minimizzare questo valore (tende a 0). Indica lo scarto tra i valori predetti e quelli reali.
*   **Rho ($\rho$ - Coefficiente di Correlazione di Pearson):** L'obiettivo è massimizzare questo valore (tende a 1). Indica quanto bene le predizioni seguono il trend dei dati reali.

### Il ruolo del "Trivial Predictor" (Baseline)
È fondamentale confrontare modelli complessi come SVT con un **predittore banale** (es. la media globale dei rating). 
*   **Osservazione:** Sebbene l'SVT superi rapidamente il predittore banale in termini di correlazione ($\rho$), il miglioramento dell'RMSE può essere marginale o richiedere molte iterazioni per scendere sotto la soglia della baseline. Ciò evidenzia l'importanza di avere modelli di riferimento semplici per validare la reale utilità di algoritmi avanzati.

### Applicazione: Image Inpainting
Il completamento di matrici può essere applicato alla ricostruzione di immagini dove mancano dei pixel (corruzione dei dati).
*   **Algoritmo SVT Pratico:**
    1.  **Inizializzazione:** Si definisce una matrice di rumore (mask) che simula la perdita di dati (es. 50% dei pixel rimossi).
    2.  **Parametri:** Vengono definiti parametri critici come $\delta$ (step size), $\tau$ (soglia di troncamento) e $C_0$.
    3.  **Iterazione:**
        *   Si esegue la SVD (Singular Value Decomposition) parziale.
        *   **Soft-thresholding:** Si applica una contrazione ai valori singolari, riducendo verso lo zero quelli inferiori alla soglia $\tau$.
        *   Aggiornamento delle variabili duali e della matrice ricostruita.
*   **Performance:** L'SVT eccelle nella ricostruzione di immagini con forti pattern geometrici (es. quadri di Mondrian), ma è efficace anche su immagini naturali (paesaggi), sebbene con un costo computazionale maggiore e necessità di più iterazioni.

---

## 2. Introduzione a JAX

**JAX** è una libreria Python per il calcolo numerico ad alte prestazioni, sviluppata da Google e focalizzata sulla differenziazione automatica e l'ottimizzazione per acceleratori hardware.

### Caratteristiche Principali
*   **Autodiff Library:** Progettata per calcolare gradienti di funzioni complesse in modo efficiente.
*   **API NumPy-like:** La sintassi di `jax.numpy` (spesso importata come `jnp`) è quasi identica a quella di NumPy, rendendo la curva di apprendimento molto dolce.
*   **XLA (Accelerated Linear Algebra):** JAX utilizza il compilatore XLA per ottimizzare ed eseguire il codice su GPU e TPU.

### Differenze Cruciali con NumPy

#### A. Immutabilità
A differenza degli array NumPy, gli **array JAX sono immutabili**. Non è possibile modificare un elemento in-place (es. `x[0] = 10` genererà un errore).
*   **Soluzione Functional:** Per aggiornare un valore, si usa la sintassi funzionale:
    `new_x = x.at[index].set(value)`
    JAX creerà una "copia" ottimizzata con la modifica applicata. Sebbene sembri inefficiente, il compilatore JIT è in grado di ottimizzare queste operazioni.

#### B. Gestione dei Numeri Casuali (PRNG)
JAX non utilizza uno stato globale per la generazione di numeri casuali (a differenza di `np.random`).
*   **Explicit State:** È necessario passare esplicitamente una chiave di stato (`PRNGKey`) a ogni funzione che genera casualità. Questo garantisce la riproducibilità totale anche in ambienti di calcolo parallelo.

#### C. Agnosticismo rispetto all'Hardware
JAX permette di eseguire lo stesso codice su CPU, GPU o TPU senza modifiche sostanziali.
*   **Costo del Trasferimento Dati:** Spostare dati dalla memoria della CPU alla GPU ha un overhead significativo. È ottimale creare i dati direttamente sul device di destinazione o minimizzare i trasferimenti durante i loop di calcolo.

### Ottimizzazione tramite JIT (Just-In-Time)
La funzione `jax.jit` è uno degli strumenti più potenti di JAX.
*   **Compilazione:** `jit` analizza la funzione Python al primo avvio e la compila in un kernel ottimizzato per l'acceleratore hardware.
*   **Performance:** Le funzioni "jittate" possono essere ordini di grandezza più veloci rispetto al codice Python standard o NumPy puro, specialmente per operazioni matriciali massive.

---

**Terminologia Tecnica Chiave:**
*   **SVD:** Singular Value Decomposition.
*   **Soft-thresholding:** Operatore di soglia che riduce i valori singolari.
*   **Inpainting:** Tecnica di restauro per colmare lacune in un'immagine.
*   **Functional Programming:** Paradigma di programmazione basato sull'immutabilità dei dati.
*   **PRNGKey:** Chiave pseudocasuale per la gestione deterministica della casualità.

---

## ⏱️ Min 90-120

Ecco degli appunti universitari dettagliati basati sul contenuto del video tutorial su **JAX**, una libreria per il calcolo numerico ad alte prestazioni e il machine learning.

---

# Appunti del Corso: Programmazione ad Alte Prestazioni con JAX

## 1. JIT Compilation (`jax.jit`)
La compilazione **Just-In-Time (JIT)** è uno dei pilastri fondamentali di JAX. Consente di trasformare funzioni Python in codice ottimizzato per acceleratori (GPU/TPU) utilizzando il compilatore **XLA (Accelerated Linear Algebra)**.

- **Definizione**: `jax.jit` è una funzione di ordine superiore (*higher-order function*) che accetta una funzione come input e ne restituisce una versione compilata e ottimizzata.
- **Sintassi**:
  - Utilizzo come funzione: `selu_jit = jax.jit(selu)`
  - Utilizzo come decoratore: `@jax.jit`
- **Vantaggi Prestazionali**: Nel caso della funzione di attivazione **SELU**, la versione compilata (`jit`) risulta essere circa **10 volte più veloce** della versione standard.
- **Ottimizzazione degli Array**: JAX gestisce gli array come immutabili. Tuttavia, dopo la compilazione JIT, JAX è in grado di rilevare se un array può essere aggiornato **in-place** per evitare copie inutili in memoria, ottimizzando drasticamente l'efficienza.

## 2. Differenziazione Automatica (`jax.grad`, `jacrev`, `jacfwd`)
JAX è progettato per calcolare derivate di funzioni complesse in modo automatico ed efficiente.

### 2.1 Funzioni Scalari (`jax.grad`)
- `jax.grad(f)` restituisce una funzione che calcola il gradiente di `f` rispetto al suo primo argomento.
- È possibile comporre le chiamate per ottenere derivate di ordine superiore:
  - Prima derivata: `dfdx = jax.grad(f)`
  - Seconda derivata: `d2fdx2 = jax.grad(jax.grad(f))`

### 2.2 Input Multipli e Funzioni Vettoriali (Jacobiani)
Quando si hanno più input o output vettoriali, si utilizzano le funzioni Jacobiane:
- **`jax.jacfwd` (Forward-mode AD)**: Più efficiente per matrici Jacobiane "alte" (numero di output molto maggiore del numero di input).
- **`jax.jacrev` (Reverse-mode AD)**: Più efficiente per matrici Jacobiane "larghe" (numero di input molto maggiore del numero di output). È la modalità tipicamente usata nel backpropagation delle reti neurali.
- **Calcolo dell'Hessiana**: Il metodo più efficiente per calcolare la matrice Hessiana è combinare le due modalità: `jax.jacfwd(jax.jacrev(f))`.

### 2.3 Casi Limite
- **Funzioni non differenziabili**: Per funzioni come il valore assoluto ($|x|$), JAX adotta un approccio pragmatico. Nel punto di non differenziabilità ($x=0$), JAX assegna convenzionalmente il valore **1.0** alla derivata per evitare la propagazione di valori indefiniti (NaN) o infiniti nel codice.

## 3. Vettorizzazione Automatica (`jax.vmap`)
`jax.vmap` (**Vectorized Map**) elimina la necessità di scrivere cicli `for` espliciti in Python, che sono notoriamente lenti.

- **Funzionamento**: Trasforma una funzione progettata per operare su singoli campioni in una funzione che opera su interi "batch" di dati.
- **Parametro `in_axes`**: Specifica lungo quali assi degli input deve avvenire la mappatura (es. `0` per le righe).
- **Benchmark di prestazioni (Esempio Prodotto Scalare Custom)**:
  1. **Naive (ciclo for)**: ~459 ms
  2. **Vetorizzato (`vmap`)**: ~1.21 ms (circa 400x più veloce)
  3. **Vettorizzato + JIT (`jit(vmap(f))`)**: **~69 μs**.
- **Conclusione**: La combinazione di `vmap` e `jit` rappresenta il "gold standard" per le prestazioni in JAX, arrivando a essere fino a **5000 volte più veloce** di un'implementazione Python standard.

## 4. Struttura delle API: `jnp` vs `lax`
JAX espone due interfacce principali:
- **`jax.numpy` (jnp)**: Un'interfaccia ad alto livello che ricalca quasi perfettamente NumPy. È flessibile e gestisce implicitamente la promozione dei tipi (es. sommare un `int` e un `float`).
- **`jax.lax`**: Un'interfaccia a basso livello, più potente ma meno "user-friendly".
  - È molto più **rigida**: non permette la promozione implicita dei tipi (sommare `int32` e `float32` genera un errore).
  - Richiede la definizione esplicita di parametri tecnici (es. window strides e padding nelle convoluzioni).
  - `jnp` è essenzialmente un wrapper ottimizzato attorno alle primitive di `lax`.

## 5. Tracing e Limitazioni di JIT
Il funzionamento interno di JIT si basa sul concetto di **Tracing**.

- **Fase di Tracing**: Alla prima chiamata di una funzione `jit`, JAX non usa array reali ma degli **oggetti "Tracer"** per mappare la logica della funzione e creare una rappresentazione intermedia chiamata **jaxpr** (JAX Expression).
- **Effetti Collaterali**: Poiché il tracing avviene solo una volta, gli effetti collaterali di Python (come le istruzioni `print`) vengono eseguiti **solo alla prima chiamata**. Nelle chiamate successive, viene eseguito solo il codice compilato XLA, e la `print` scompare.
- **Limitazioni (Dynamic Shapes)**: JIT non può gestire funzioni in cui la dimensione dell'output dipende dai valori dei dati (es. boolean masking come `x[x < 0]`). JAX richiede forme (shapes) statiche per poter ottimizzare il codice; per gestire forme dinamiche sono necessari approcci specifici o l'evitare JIT in quelle sezioni.

---

## ⏱️ Min 120-150

Certamente! Ecco degli appunti universitari dettagliati basati sul segmento video fornito, che approfondisce le sfumature della compilazione JIT (Just-In-Time) in JAX, la gestione degli stati e altre "insidie" comuni (gotchas).

---

# Appunti del Corso: Programmazione Avanzata con JAX

## 1. JAX JIT e Dipendenza dai Valori
La compilazione JIT (`jax.jit`) in JAX è estremamente efficiente ma impone vincoli rigorosi sulla struttura del codice. Uno dei problemi principali sorge quando il flusso di esecuzione (es. un'istruzione `if`) dipende dal **valore** di una variabile, piuttosto che dalla sua forma o tipo.

### 1.1 Il Problema del Tracer
*   **Comportamento del Tracer:** Durante la prima chiamata a una funzione JIT-tata, JAX esegue la funzione con "tracciatori" (tracers) per catturarne le operazioni.
*   **Errore di Dipendenza dal Valore:** Se una condizione `if` dipende da un valore concreto (come un booleano passato come input), JAX non può creare un grafo statico univoco perché il percorso del codice cambierebbe in base al valore. Questo genera un `Traceback: TracerBoolConversionError`.

### 1.2 Strategie di Risoluzione (Workarounds)
1.  **Tricks Aritmetici:** Sostituire la logica condizionale con espressioni matematiche che sfruttano la conversione implicita di booleani in interi (True=1, False=0).
    *   *Esempio:* `return x * (1.0 - 2.0 * neg)` sostituisce `if neg: return -x else: return x`.
2.  **Argomenti Statici (`static_argnums`):** Indicare a JAX quali argomenti devono essere trattati come costanti durante la compilazione.
    *   *Svantaggio:* Modificare un argomento statico forza una ricompilazione della funzione, che è un'operazione costosa.
    *   *Uso:* `@partial(jax.jit, static_argnums=(1,))` permette di usare l'argomento all'indice 1 in un `if`.

---

## 2. Il Paradigma delle Funzioni Pure
JAX è progettato per operare esclusivamente su **funzioni pure**.

### 2.1 Definizione di Funzione Pura
*   Tutti i dati di input passano attraverso i parametri della funzione.
*   Tutti i risultati sono restituiti tramite il valore di ritorno.
*   A parità di input, la funzione restituisce sempre lo stesso output (assenza di effetti collaterali o stati globali).

### 2.2 Problema delle Variabili Globali (Impurezzza)
*   Se una funzione JIT-tata accede a una variabile globale (es. `g = 0.0`), JAX **congela** il valore di quella variabile al momento della prima compilazione.
*   Anche se la variabile globale viene aggiornata successivamente nel codice Python, la funzione JIT-tata continuerà a usare il valore "cacheato" (congelato), portando a risultati errati.
*   La funzione viene ricompilata (e quindi legge il nuovo valore globale) solo se cambiano il tipo o la forma degli argomenti di input.

---

## 3. Gestione dei Numeri Casuali (PRNG)
La generazione di numeri casuali evidenzia la differenza tra l'approccio stateful di NumPy e quello stateless di JAX.

### 3.1 NumPy vs. JAX
*   **NumPy:** Utilizza un generatore di numeri pseudo-casuali (PRNG) **stateful**. Ogni chiamata a `np.random.random()` aggiorna uno stato globale interno in modo invisibile all'utente. Questo è un "effetto collaterale" che viola la purezza funzionale.
*   **JAX:** Utilizza un PRNG **stateless** ed esplicito. Richiede una "chiave" (`key`) come parametro.

### 3.2 Chiavi e Splitting
Per mantenere la purezza e la riproducibilità, in JAX lo stato del PRNG deve essere gestito manualmente tramite `jax.random.split`.
*   Senza lo "split", chiamare la stessa funzione casuale con la stessa chiave produrrà sempre lo stesso numero.
*   `key, subkey = jax.random.split(key)` genera nuove chiavi deterministiche per le operazioni successive.

### 3.3 Riproducibilità e Parallelismo
L'approccio di JAX è fondamentale in ambienti di calcolo parallelo (multi-GPU/TPU). In NumPy, due processi paralleli che accedono allo stesso stato globale possono generare condizioni di gara (race conditions), rendendo il codice non deterministico. JAX, gestendo le chiavi in modo esplicito per ogni unità di calcolo, garantisce la totale riproducibilità.

---

## 4. Altre Insidie Tecniche (Gotchas)

### 4.1 Indicizzazione Out-of-Bounds
*   A differenza di NumPy o Python puro, JAX non genera un errore se si tenta di accedere a un indice fuori limite.
*   **Comportamento:** Gli aggiornamenti fuori limite vengono ignorati; i recuperi di valori fuori limite vengono "clampati" (fissati) all'ultimo indice disponibile. Questo è dovuto all'approccio agnostico rispetto all'acceleratore (GPU/TPU).

### 4.2 Control Flow e Gradienti
*   `jax.grad` (per il calcolo dei gradienti) gestisce bene i rami `if/else` di Python.
*   Tuttavia, se si combina `jax.grad` con `jax.jit`, si ricade nel problema del tracciamento dei valori.
*   **Soluzione:** Utilizzare `jax.numpy.where(condizione, ramo_true, ramo_false)` per vettorizzare la logica condizionale, garantendo che l'output abbia sempre la stessa forma.

### 4.3 Debugging e Precisione
*   **Debugging NaN:** Usare `jax.config.update("jax_debug_nans", True)` per sollevare un errore non appena compare un valore non numerico (NaN), facilitando il tracciamento in codici complessi.
*   **Precisione a 64-bit:** Di default, JAX utilizza la singola precisione (32-bit) per ottimizzare le performance su GPU. Per la doppia precisione (matematica avanzata), è necessario abilitarla esplicitamente con `jax.config.update("jax_enable_x64", True)`.

---

