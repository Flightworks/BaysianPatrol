# 🚁 BaysianPatrol - Recherche Maritime Optimisée par RL & Analyse Bayésienne

**BaysianPatrol** est un simulateur de Recherche et Sauvetage Maritime (SAR) permettant d'évaluer et de comparer 3 stratégies de patrouille pour hélicoptère embarqué face à une cible stochastique sous incertitude spatio-temporelle (**Datum**).

---

## 📖 Mode d'Emploi Simple

### 1️⃣ Lancer l'Application Web (Visualisation & Dashboard)

#### Étape A : Installer les dépendances Node.js (première fois uniquement)
```bash
npm install
```

#### Étape B : Démarrer le serveur web local
```bash
npm run dev
```
👉 Ouvrez votre navigateur sur **`http://localhost:5173`**

#### Étape C : Lancer le Comparatif 3-Voies
1. Dans le panneau latéral gauche (**Configuration de la Simulation Stochastique**), défilez jusqu'à **Contrôles Monte-Carlo**.
2. Dans le menu déroulant **Comparaison Stratégie**, choisissez :
   `🏆 COMPARATIF TRIO (Naïve vs AMI vs RL PPO)`
3. Cliquez sur le bouton bleu **"Lancer Simulation Monte-Carlo"** dans la barre supérieure.
4. Cliquez sur l'onglet **"Tableau de Bord Statistique"** (en haut à droite de la carte) pour comparer les taux de réussite, le temps moyen et le carburant consommé.

---

### 2️⃣ Entraîner le Modèle RL (Apprentissage par Renforcement Python)

#### Étape A : Installer les dépendances Python
```bash
pip install -r python/requirements.txt
```

#### Étape B : Lancer l'entraînement PPO
```bash
python python/train.py
```
*Le modèle entraîné est automatiquement exporté au format ONNX dans `public/models/baysian_patrol_policy.onnx` pour être utilisable directement dans l'interface Web.*

#### Étape C : Suivre les courbes d'apprentissage en direct (TensorBoard)
```bash
tensorboard --logdir python/tensorboard_logs
```
👉 Ouvrez **`http://localhost:6006`** dans votre navigateur pour suivre les courbes de récompense et d'interception.

---

## 📊 Les 3 Stratégies Comparées

1. **📐 Option 1 : Naïve (IAMSAR)** — Râteau de recherche géométrique classique à balayage parallèle.
2. **⚡ Option 2 : Bayésienne AMI (POMDP)** — Solveur algorithmique maximisant le gradient de croyance bayésienne $P(x,y,t)$ à chaque pas de temps.
3. **🤖 Option 3 : Modèle RL SIGMA (PPO)** — Réseau de neurones hybride (CNN 2D + Perceptron) inférant le cap et la vitesse optimaux avec masquage de sécurité **Bingo Fuel**.

---

## 🛠️ Commandes Utiles

- **Valider le build de production Web** :
  ```bash
  npm run build
  ```
- **Exécuter les tests unitaires de l'environnement RL** :
  ```bash
  python python/tests/test_env.py
  ```
- **Re-générer le modèle ONNX de base** :
  ```bash
  python python/export_onnx.py
  ```
- **Spécification technique complète pour agent AI** : Voir [`AGENTS.md`](AGENTS.md)
