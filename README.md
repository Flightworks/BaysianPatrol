# Baysian Patrol 2.4

Simulateur maritime de recherche et d’interception permettant de comparer trois stratégies sur exactement les mêmes situations Monte-Carlo.

## Les trois stratégies

1. **Stratégie hybride 2027** — le modèle choisit une direction tactique ; l’exécuteur conserve ce choix sur une branche minimale cohérente avec la portée radar.
2. **Recherche bayésienne** — choisit un corridor à forte probabilité encore peu couvert, puis termine une branche avant de recalculer.
3. **Balayage parallèle inspiré IAMSAR** — propage le datum pendant le transit et oriente les branches perpendiculairement à la route estimée du mobile, avec un espacement lié à la portée radar.

Chaque comparaison utilise la même cible, la même météo, la même position de frégate et les mêmes tirages de détection pour les trois stratégies.

Le datum est une référence observée, jamais la position vraie certaine. Pour chaque réalisation, la vérité terrain est tirée séparément à partir des incertitudes spatiale et temporelle, puis progresse dès la première minute selon sa route, sa vitesse et la dérive.

## Workflow de démonstration

```bash
npm install
npm run dev
```

Ouvrir `http://localhost:5173`, puis :

1. choisir un scénario, le nombre de tirages et la seed ;
2. lancer la campagne depuis l’onglet **Comparaison** ;
3. comparer détection, temps d’interception, carburant, retours sûrs et violations Bingo ;
4. rejouer un tirage dans **Carte tactique** ;
5. retrouver les vingt derniers résumés dans **Historique**.

Le modèle actif est toujours la stratégie hybride qualifiée en 2027. L’interface métier ne propose ni sélection de modèle, ni entraînement, ni réglages PPO.

## Comment le modèle a été entraîné

Le modèle n’a pas appris seul en prenant des risques dans le monde réel. Il a appris dans un simulateur, en trois temps :

1. imiter un expert qui choisit la zone la plus probable ;
2. essayer d’améliorer ses décisions avec PPO ;
3. conserver la version la plus sûre après comparaison sur 500 situations inédites.

Les trois entraînements indépendants ont retenu la version par imitation (Behavior Cloning), avec 500 interceptions sur 500, zéro Bingo et zéro sortie de zone. La stratégie active de l’application est cette version qualifiée avec la seed 2027.

L’explication complète, destinée aux lecteurs non spécialistes, se trouve dans [`docs/rl-explique-aux-profanes.md`](docs/rl-explique-aux-profanes.md).

## Carte tactique

La carte superpose les trois trajectoires sur un fond noir. Ses couches probabilistes sont :

- **Prévision de route** : position probable issue du datum, de la route estimée et de la dérive, sans effet des balayages radar ;
- **Posterior radar** : prévision corrigée après les non-détections ; une zone balayée perd de la probabilité ;
- **Posterior tactique** : posterior radar ajusté selon l’efficacité du capteur et l’angle d’approche.

La vérité terrain est désactivée par défaut. Lorsqu’elle est volontairement affichée, la carte distingue le datum fixe de la cible vraie mobile. Le rejeu utilise une trajectoire cible commune aux trois stratégies et continue jusqu’à la fin de la dernière stratégie active, même si une autre a déjà intercepté.

## Historique local

Les vingt dernières campagnes sont stockées dans le navigateur avec leurs paramètres et leurs statistiques compactes. Les trajectoires détaillées ne sont pas conservées afin de limiter le volume de stockage.

## Architecture de sécurité

Python/Gymnasium reste la référence canonique. Le modèle ne pilote pas directement le cap ou la vitesse :

```text
modèle hybride → waypoint relatif → pilote automatique → geofence → superviseur carburant/RTB
```

La vitesse commandée est une vitesse air. Le pilote automatique calcule la vitesse sol et l’angle de crabe à partir du vent : vent de face plus lent, vent arrière plus rapide, vent traversier compensé. Le superviseur de retour utilise la même vitesse sol pour éviter un déclenchement Bingo trop tardif.

Un retour normal est publié comme `SAFE_RTB`. Les violations Bingo, pannes carburant et limites de temps restent des résultats distincts.

## Vérification

```bash
# Contrat Python
python -m unittest discover -s python/tests -v

# Contrats TypeScript, historique et râteau IAMSAR
node --experimental-strip-types --test tests-ts/*.test.ts

# Production web
npm run build
```

La procédure complète d’entraînement, d’export ONNX et de qualification est décrite dans [`docs/rl-explique-aux-profanes.md`](docs/rl-explique-aux-profanes.md). Les détails de travail et les contrats internes restent dans [`AGENTS.md`](AGENTS.md).
