# Guide Architecture & Spécifications de l'Agent RL (AGENTS.md)

Ce document décrit en détail le concept, les spécifications mathématiques et la structure technique du système de patrouille maritime optimisée par **Apprentissage par Renforcement (Reinforcement Learning - RL)**.

---

## 1. Vue d'Ensemble du Concept

Le projet **BaysianPatrol** modélise un problème de Recherche et Sauvetage Maritime (SAR) où un hélicoptère embarqué (décollant d'une frégate) doit intercepter une cible mobile stochastique (navire désemparé, naufragé) localisée initialement avec une incertitude spatio-temporelle (**Datum**).

L'objectif de l'**Agent RL** est d'apprendre une politique de navigation optimale $\pi(a_t | o_t)$ pour maximer le taux d'interception de la cible tout en minimisant le temps de recherche et en respectant les contraintes strictes de sécurité de carburant (**Bingo Fuel**).

---

## 2. Spécification de l'Agent RL (Gymnasium & PPO)

L'agent évolue dans l'environnement Gymnasium `BaysianPatrolEnv` (dossier `/python`).

### A. Espace d'Observations ($O_t$)
Le vecteur d'observation reçu par l'agent à chaque pas $t$ réunit la croyance spatiale bayésienne et l'état cinématique de l'aéronef :

1. **Branche Grille 2D Convolutive (`grid`)** : Matrice $2 \times 32 \times 32$
   - **Canal 0** : Carte de probabilité de présence bayésienne évolutive $P(x,y,t)$ normalisée ($\sum P = 1.0$).
   - **Canal 1** : Masque binaire des zones déjà prospectées par le radar ($1.0 = \text{scanné}, 0.0 = \text{non scanné}$).
2. **Branche Scalaire MLP (`vector`)** : Vecteur normalisé à 10 dimensions
   - $[x_{\text{norm}}, y_{\text{norm}}, \sin(\theta), \cos(\theta), V_{\text{norm}}, \text{Fuel}_{\text{ratio}}, d_{\text{frégate\_norm}}, v_{w,x}, v_{w,y}, R_{0,\text{norm}}]$

### B. Espace d'Actions ($A_t$)
Espace continu $A_t \in [-1.0, 1.0]^2$ :
- **Action[0] ($\Delta \theta$)** : Variation relative de cap, projetée sur le taux de virage maximal admis par la physique ($\omega_{\max} = 3^\circ/\text{s}$).
- **Action[1] ($\Delta V$)** : Variation relative de vitesse, projetée sur l'accélération maximale admissible ($V \in [60, V_{\max}]\text{ kts}$).

### C. Fonction de Récompense ($R_t$)
La fonction de récompense globale au pas de temps $t$ équilibre la vitesse d'interception, l'exploration bayésienne et la sécurité de l'équipage :

$$R_t = R_{\text{détection}} + R_{\text{gain\_prob}} + R_{\text{temps}} + R_{\text{bingo}} + R_{\text{limite}}$$

- **Détection de Cible** : $R_{\text{détection}} = +500.0$ si la cible est détectée par le radar ($P_{\text{det}} > 0.05$ ou proximité $< 0.8\text{ NM}$).
- **Gain d'Information** : $R_{\text{gain\_prob}} = +10.0 \times \Delta P_{\text{scanned}}$ (récompense le balayage de zones à forte probabilité $P(x,y,t)$).
- **Pénalité Temporelle / Carburant** : $R_{\text{temps}} = -0.1 \times dt$ (minimise la durée de patrouille).
- **Pénalité Bingo Fuel** : $R_{\text{bingo}} = -1000.0$ si l'hélicoptère franchit la limite Bingo Fuel sans autonomie suffisante pour rallier la frégate.
- **Sortie de Zone** : $R_{\text{limite}} = -50.0$ si l'aéronef sort des limites de la zone de recherche.

### D. Architecture du Réseau de Neurones
- **Feature Extractor** : Réseau neuronal convolutif (NatureCNN 2D) pour extraire les caractéristiques spatiales de la grille $32 \times 32$, fusionné avec un Perceptron (MLP) à 64 neurones pour les vecteurs cinématiques.
- **Tête de Politique (Actor-Critic PPO)** : Génère la distribution gaussienne des actions $[\Delta \theta, \Delta V]$.
- **Export ONNX** : Politique exportée au format ONNX (`baysian_patrol_policy.onnx`) pour une exécution client-side sans serveur backend.

---

## 3. Banc de Test Monte-Carlo (Comparatif 3-Voies)

Pour évaluer objectivement les performances, le simulateur Monte-Carlo (`/src`) compare l'agent RL à deux autres stratégies sur $N$ tirages stochastiques identiques :

| Stratégie | Description / Principe | Type d'Algorithme |
| :--- | :--- | :--- |
| **Option 1 : Naïve (IAMSAR)** | Râteau de recherche géométrique parallèle basé sur les fiches théoriques SAR. | Heuristique déterministe fixe |
| **Option 2 : Bayésienne (AMI / POMDP)** | Solveur algorithmique recalculant à chaque pas de temps la passe transversale maximisant le gradient de croyance. | Solveur d'optimisation numérique |
| **Option 3 : Modèle RL SIGMA (PPO)** | Réseau de neurones (Agent RL / SIGMA) inférant directement le cap et la vitesse optimaux selon l'état de croyance et les incertitudes. | Réseau de Neurones / ONNX Runtime |

---

## 4. Organisation de la Structure du Projet

```
BaysianPatrol/
├── AGENTS.md                  # Ce fichier de spécification globale de l'agent RL
├── python/                    # MODULE D'ENTRAÎNEMENT PYTHON
│   ├── baysian_patrol_env.py  # Environnement Gymnasium (Grille 2D, Cinématique, Rewards)
│   ├── train.py               # Script d'entraînement PPO (Stable-Baselines3 + TensorBoard)
│   ├── export_onnx.py         # Script d'export vers public/models/baysian_patrol_policy.onnx
│   ├── requirements.txt       # Dépendances Python (gymnasium, torch, onnx, etc.)
│   └── tests/
│       └── test_env.py        # Tests unitaires de l'environnement Gymnasium
├── public/
│   └── models/
│       └── baysian_patrol_policy.onnx # Modèle ONNX pré-compilé pour inférence navigateur
├── src/                       # APPLICATION WEB MONTE-CARLO (REACT / TS)
│   ├── components/
│   │   ├── ControlPanel.tsx   # Panneau de configuration stochastique & choix de stratégie
│   │   ├── StatsDashboard.tsx # Dashboard de comparatif 3-voies (KPIs, Win Rates)
│   │   └── TacticalCanvas.tsx # Visualisateur WebGL / Canvas 2D des trajectoires
│   ├── engine/
│   │   ├── bayesianGrid.ts    # Moteur de mise à jour de la densité bayésienne
│   │   ├── rlAlgorithm.ts     # Inférence du Modèle RL (TypeScript / ONNX)
│   │   ├── simulator.ts       # Moteur d'exécution des runs Monte-Carlo
│   │   └── mcWorker.ts        # Web Worker pour calculs distribués
│   └── types/
│       └── simulation.ts      # Définition des types TypeScript (TRIO, RL_MODEL, Stats)
```

---

## 5. Commandes de Référence pour un Agent ou Développeur

### Entraîner le Modèle RL (Python) :
```bash
python python/train.py
```

### Visualiser l'Entraînement (TensorBoard) :
```bash
tensorboard --logdir python/tensorboard_logs
```

### Lancer les Tests Unitaires Python :
```bash
python python/tests/test_env.py
```

### Lancer l'Application Web & le Comparateur (React) :
```bash
npm run dev
```

### Valider le Build Production Web :
```bash
npm run build
```
