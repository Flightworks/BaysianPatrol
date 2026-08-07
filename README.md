# Baysian Patrol 2.4

Simulateur maritime de recherche et d’interception permettant de comparer trois stratégies sur exactement les mêmes situations Monte-Carlo.

## Les trois stratégies

1. **Stratégie hybride 2027** — modèle qualifié choisissant un waypoint relatif, avec pilote automatique, geofence et retour carburant déterministes.
2. **Recherche bayésienne** — planification à partir d’une croyance probabiliste mise à jour pendant la mission.
3. **Balayage parallèle IAMSAR** — recherche en râteau déterministe : départ à un demi-espacement de piste du bord, branches parallèles au grand côté et espacement lié à la portée radar.

Chaque comparaison utilise la même cible, la même météo, la même position de frégate et les mêmes tirages de détection pour les trois stratégies.

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

La carte superpose les trois trajectoires sur un fond noir. Elle permet d’afficher :

- le datum initial ;
- la propagation de l’incertitude ;
- l’estimation tactique active ;
- la vérité terrain, désactivée par défaut ;
- l’ensemble des trajectoires d’une campagne.

## Historique local

Les vingt dernières campagnes sont stockées dans le navigateur avec leurs paramètres et leurs statistiques compactes. Les trajectoires détaillées ne sont pas conservées afin de limiter le volume de stockage.

## Architecture de sécurité

Python/Gymnasium reste la référence canonique. Le modèle ne pilote pas directement le cap ou la vitesse :

```text
modèle hybride → waypoint relatif → pilote automatique → geofence → superviseur carburant/RTB
```

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
