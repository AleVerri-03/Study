# 📚 Guida Rapida Esercizi PDE - Selezione per l'Esame

Questa guida ti permette di identificare rapidamente quale esercizio è più simile al problema che ti viene assegnato all'esame.

---

## 🔍 Tabella di Ricerca Rapida

| Tipo di Problema | Cartella | Dimensione | Time-Dependent | Note Speciali |
|------------------|----------|------------|----------------|---------------|
| **Heat/ADR 2D** con convezione | `20240115_01` | 2D | ✅ Sì (θ-method) | Crank-Nicolson, mesh da file |
| **Heat/ADR 1D** diffusione-convezione-reazione | `20240715_01` | 1D | ✅ Sì (θ-method) | Mesh interna |
| **ADR 1D** decomposizione domini | `20240715_03` | 1D | ❌ No | Domain Decomposition |
| **Heat/ADR 1D** diffusione-convezione-reazione | `20240828_01` | 1D | ✅ Sì (θ-method) | Mesh interna |
| **Parabolico 1D** decomposizione domini | `20240902_01` | 1D | ✅ Sì (implicito) | Domain Decomposition |
| **ADR 1D** con SUPG | `20250113_01` | 1D | ✅ Sì | Stabilizzazione SUPG |
| **Stokes 2D** stazionario | `20250113_02` | 2D | ❌ No | Flusso in canale |
| **ADR 1D** convezione variabile | `20250210_01` | 1D | ❌ No | Convezione -x·u' |
| **Stokes 2D** tempo-dipendente | `20250210_02` | 2D | ✅ Sì | Navier-Stokes intstazionario |
| **Poisson 1D** base | `20250604` | 1D | ❌ No | Problema base |
| **Stokes 2D** stazionario | `20250604_02` | 2D | ❌ No | Forcing f = (1+sin²(x), 0) |
| **Stokes 2D** stazionario | `20250708_01` | 2D | ❌ No | Simile a 20250604_02 |
| **Diffusion-Reaction 2D** | `fac_simile_1` | 2D | ❌ No | Neumann BC, μ e σ generici |
| **Stokes 2D** inlet parabolico | `fac_simile_2` | 2D | ❌ No | g = 27/4·y·(1-y)² |
| **Diffusion-Reaction 2D** | `fack_simile_1` | 2D | ❌ No | Dirichlet BC, mesh parametrica |
| **Stokes 2D** inlet parabolico | `fack_simile_2` | 2D | ❌ No | Simile a fac_simile_2 |

---

## 📋 Descrizione Dettagliata per Tipo di Equazione

---

### 🔥 EQUAZIONE DEL CALORE / ADR (Advection-Diffusion-Reaction)

#### **20240115_01** - Heat 2D con Convezione
**Equazione:**
$$\frac{\partial u}{\partial t} - \mu \Delta u + \mathbf{b} \cdot \nabla u = f(x,t)$$

**Caratteristiche:**
- Dimensione: **2D**
- Schema temporale: **θ-method** (Crank-Nicolson con θ=0.5)
- Mesh: letta da **file esterno** (`.msh`)
- Parallelizzazione: **MPI** con Trilinos
- Condizioni al bordo: **Dirichlet omogenee**
- Soluzione esatta: $u_{ex} = \sin(\pi x) \sin(2\pi y) \sin(2\pi t)$

**Usa questo se:** Il problema ha convezione ($\mathbf{b} \cdot \nabla u$), diffusione ($-\mu \Delta u$), è 2D e tempo-dipendente.

---

#### **20240715_01** - Heat 1D con ε, κ, b
**Equazione:**
$$\frac{\partial u}{\partial t} - \varepsilon u_{xx} + \kappa u + b u_x = f(x,t)$$

**Caratteristiche:**
- Dimensione: **1D**
- Schema temporale: **θ-method**
- Mesh: generata internamente con `subdivided_hyper_cube(40, 0, 1)`
- Parallelizzazione: **MPI** con Trilinos
- Parametri: ε (diffusione), κ (reazione), b (convezione scalare)
- Soluzione esatta: $u_{ex} = \sin(\pi x/2) \sin(\pi t/2)$

**Usa questo se:** Problema 1D ADR completo (diffusione + convezione + reazione) tempo-dipendente.

---

#### **20240828_01** - Heat 1D (simile a 20240715_01)
**Equazione:**
$$\frac{\partial u}{\partial t} - \varepsilon u_{xx} - b(x) u_x + k u = f(x,t)$$

**Caratteristiche:**
- Dimensione: **1D**
- Schema temporale: **θ-method**
- Mesh: generata internamente
- Convection: **b(x)** può essere funzione di x
- Soluzione esatta: $u_{ex} = \sin(\pi x/2) \sin(\pi t/2)$

**Usa questo se:** Problema 1D ADR con coefficiente di convezione variabile nello spazio.

---

#### **20250113_01** - ADR 1D con SUPG
**Equazione:**
$$\frac{\partial u}{\partial t} - u_{xx} + \kappa u_x = f$$

**Caratteristiche:**
- Dimensione: **1D**
- Schema temporale: **Implicito**
- **Stabilizzazione SUPG** (Streamline Upwind Petrov-Galerkin)
- Calcolo del parametro τ con numero di Peclet locale
- Condizione iniziale: $u_0(x) = x$
- Condizioni al bordo: Robin/Dirichlet miste

**Usa questo se:** Problema convection-dominated che richiede **stabilizzazione SUPG**.

---

### 🌀 DECOMPOSIZIONE DI DOMINI (Domain Decomposition)

#### **20240715_03** - ADR 1D con Domain Decomposition
**Equazione:**
$$-\varepsilon u_{xx} + b u_x + c u = f$$

**Caratteristiche:**
- Dimensione: **1D**
- Stazionario
- **Domain Decomposition** con due sottodomini
- Interfaccia a x = 0.75
- Metodi: Dirichlet-Neumann, Robin-Robin
- Solver: **GMRES** (matrice non simmetrica)

**Usa questo se:** Problema che richiede decomposizione di domini con condizioni di interfaccia.

---

#### **20240902_01** - Parabolico 1D con Domain Decomposition
**Equazione:**
$$\frac{1}{\Delta t} u - \mu u_{xx} + \kappa u = f + \frac{u^{n-1}}{\Delta t}$$

**Caratteristiche:**
- Dimensione: **1D**
- Tempo-dipendente (schema implicito)
- **Domain Decomposition** con due sottodomini
- Interfaccia a x = 0.5
- Parametri: μ = 1, κ = 1, Δt = 0.1

**Usa questo se:** Problema parabolico con domain decomposition.

---

### 📊 PROBLEMI ELLITTICI (Poisson/Diffusion-Reaction)

#### **20250604** - Poisson 1D Base
**Equazione:**
$$-u_{xx} = f(x)$$

**Caratteristiche:**
- Dimensione: **1D**
- Stazionario
- Problema base senza convezione/reazione
- Condizioni Dirichlet omogenee

**Usa questo se:** Problema ellittico 1D semplice, solo diffusione.

---

#### **20250210_01** - ADR 1D con Convezione Variabile
**Equazione:**
$$-u_{xx} - x u_x + u = f(x)$$

**Caratteristiche:**
- Dimensione: **1D**
- Stazionario
- Convezione: **-x·u'** (coefficiente variabile)
- Reazione: +u
- Condizioni Dirichlet omogenee su entrambi i bordi
- Solver: **GMRES** (matrice non simmetrica)

**Usa questo se:** Problema ADR stazionario 1D con coefficiente di convezione variabile.

---

#### **fac_simile_1** - Diffusion-Reaction 2D
**Equazione:**
$$-\nabla \cdot (\mu \nabla u) + \sigma u = f$$

**Caratteristiche:**
- Dimensione: **2D**
- Stazionario
- Coefficienti: μ(x,y) diffusione, σ(x,y) reazione (funzioni generiche)
- Mesh: letta da file con boundary tag
- Condizioni miste: **Dirichlet + Neumann**
- Include quadratura sul bordo per integrali di Neumann

**Usa questo se:** Problema di diffusione-reazione 2D con condizioni Neumann.

---

#### **fack_simile_1** - Diffusion-Reaction 2D
**Equazione:**
$$-\nabla \cdot (\mu \nabla u) + \sigma u = f$$

**Caratteristiche:**
- Dimensione: **2D**
- Stazionario
- Simile a fac_simile_1 ma con mesh parametrica
- Nome file mesh generato da parametro h
- Condizioni Dirichlet
- Soluzione esatta: $u_{ex} = \sin(\pi x/2) \cdot y$

**Usa questo se:** Problema diffusione-reazione 2D con studio di convergenza.

---

### 💧 EQUAZIONI DI STOKES

#### **20250113_02** - Stokes 2D Stazionario (Flusso in Canale)
**Equazione:**
$$-\nu \Delta \mathbf{u} + \nabla p = \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0$$

**Caratteristiche:**
- Dimensione: **2D**
- Stazionario
- ν = 1 (viscosità)
- Forcing: $\mathbf{f} = 0$
- **Inflow condition**: $u_x = 1/4 - y^2$ (profilo parabolico)
- Precondizionatori: Block-Diagonal e Block-Triangular
- FE: Taylor-Hood ($P_k - P_{k-1}$)
- Penalizzazione Nitsche: α = 50

**Usa questo se:** Problema Stokes stazionario con profilo di ingresso parabolico.

---

#### **20250210_02** - Stokes 2D Tempo-Dipendente
**Equazione:**
$$\frac{\partial \mathbf{u}}{\partial t} - \nu \Delta \mathbf{u} + \nabla p = \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0$$

**Caratteristiche:**
- Dimensione: **2D**
- **Tempo-dipendente** con loop temporale
- T_final = 1.0, Δt = 0.1
- Precondizionatore Block-Triangular
- FE: Taylor-Hood

**Usa questo se:** Problema Stokes/Navier-Stokes intstazionario.

---

#### **20250604_02** - Stokes 2D con Forcing Non-Nullo
**Equazione:**
$$-\nu \Delta \mathbf{u} + \nabla p = \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0$$

**Caratteristiche:**
- Dimensione: **2D**
- Stazionario
- ν = 2 (viscosità)
- Forcing: $\mathbf{f} = (1 + \sin^2(x), 0)^T$
- Precondizionatori Block-Diagonal e Block-Triangular

**Usa questo se:** Problema Stokes con termine forzante non banale.

---

#### **20250708_01** - Stokes 2D
**Caratteristiche:** Simile a 20250604_02

---

#### **fac_simile_2** / **fack_simile_2** - Stokes 2D Inlet Parabolico
**Equazione:**
$$-\nu \Delta \mathbf{u} + \nabla p = \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0$$

**Caratteristiche:**
- Dimensione: **2D**
- Stazionario
- **Inlet velocity**: $\mathbf{g} = (27/4 \cdot y \cdot (1-y)^2, 0)^T$
- Precondizionatori: Identity, Block-Diagonal, Block-Triangular

**Usa questo se:** Problema Stokes con inlet parabolico non simmetrico.

---

## 🎯 Checklist per Identificare l'Esercizio Giusto

1. **È tempo-dipendente?**
   - Sì → Heat/Parabolic (`20240115_01`, `20240715_01`, `20240828_01`, `20250113_01`, `20240902_01`, `20250210_02`)
   - No → Ellittico/Stokes stazionario

2. **Quante dimensioni?**
   - 1D → `20240715_01`, `20240828_01`, `20250113_01`, `20250210_01`, `20250604`, `20240715_03`, `20240902_01`
   - 2D → `20240115_01`, `fac_simile_1`, `fack_simile_1`, tutti gli Stokes

3. **È un sistema (Stokes)?**
   - Sì → `20250113_02`, `20250210_02`, `20250604_02`, `20250708_01`, `fac_simile_2`, `fack_simile_2`
   - No → Equazione scalare (Heat, Poisson, ADR)

4. **C'è convezione ($\mathbf{b} \cdot \nabla u$)?**
   - Sì → ADR (`20240115_01`, `20240715_01`, `20240828_01`, `20240715_03`, `20250210_01`)
   - Con stabilizzazione → `20250113_01` (SUPG)

5. **Domain Decomposition?**
   - Sì → `20240715_03`, `20240902_01`

6. **Condizioni al bordo?**
   - Solo Dirichlet → la maggior parte
   - Dirichlet + Neumann → `fac_simile_1`
   - Inlet parabolico (Stokes) → `20250113_02`, `fac_simile_2`

---

## ⚡ Riferimento Rapido Formule Deboli

### ADR Stazionario
$$a(u,v) = \int_\Omega \mu \nabla u \cdot \nabla v + \int_\Omega (\mathbf{b} \cdot \nabla u) v + \int_\Omega \sigma u v$$

### ADR Tempo-Dipendente (θ-method)
$$\left(\frac{u^{n+1} - u^n}{\Delta t}, v\right) + \theta \cdot a(u^{n+1}, v) + (1-\theta) \cdot a(u^n, v) = \theta (f^{n+1}, v) + (1-\theta)(f^n, v)$$

### Stokes
$$\int_\Omega \nu \nabla \mathbf{u} : \nabla \mathbf{v} - \int_\Omega p \nabla \cdot \mathbf{v} = \int_\Omega \mathbf{f} \cdot \mathbf{v}$$
$$\int_\Omega q \nabla \cdot \mathbf{u} = 0$$

---

## 📁 Struttura File Tipica

```
esercizio/
├── src/
│   ├── NomeClasse.hpp     # Dichiarazione classe
│   ├── NomeClasse.cpp     # Implementazione
│   └── exercise-01.cpp    # Main con parametri
├── mesh/
│   └── mesh-*.msh         # File mesh (se serve)
├── scripts/
│   └── run.sh             # Script di esecuzione
└── text/
    └── lab-XX.pdf         # Testo dell'esercizio
```

---

*Ultimo aggiornamento: Gennaio 2026*
