# Baysian Patrol 2.4

Simulateur maritime de recherche et d’interception permettant de comparer trois stratégies sur exactement les mêmes situations Monte-Carlo.

## Les trois stratégies

1. **Stratégie hybride 2.4.4** — le modèle choisit une direction tactique ; l’exécuteur conserve ce choix sur une branche minimale cohérente avec la portée radar.
2. **Recherche bayésienne** — POMDP approché à un pas : propage le posterior précédent par le modèle de mouvement cible, applique les non-détections, compare les actions selon leur probabilité de détection, leur gain d’information attendu et leur coût de transit, puis replanifie toutes les quatre minutes. Ce n’est pas un solveur POMDP exact à horizon long.
3. **Balayage parallèle inspiré IAMSAR** — propage le datum pendant le transit et oriente les branches perpendiculairement à la route estimée du mobile, avec un espacement lié à la portée radar.

Chaque comparaison utilise la même cible, la même météo, la même position de frégate et les mêmes tirages de détection pour les trois stratégies.

La recherche bayésienne n’est pas un solveur POMDP exact, qui serait exponentiel sur cette grille. Elle emploie une approximation déterministe et calculable dans le navigateur : observation binaire (détection/non-détection), mise à jour de Bayes, entropie attendue et ensemble fini d’actions candidates. Contrairement au râteau, ses actions ne sont pas contraintes à rester perpendiculaires à la route estimée.

Le datum est une référence observée, jamais la position vraie certaine. Pour chaque réalisation, la vérité terrain est tirée séparément à partir des incertitudes spatiale et temporelle, puis progresse dès la première minute selon sa route, sa vitesse et la dérive.

## Scénario réaliste 2.4.4

Le scénario standard représente la recherche d’une embarcation rapide avec une information initiale très incertaine. Ses valeurs par défaut sont une portée radar de 4 NM, un écart-type spatial de 20 NM sur chaque axe et un écart-type temporel de 60 minutes. L’interface permet de modifier ces trois valeurs.

Les valeurs spatiales et temporelles sont les écarts-types de lois gaussiennes. Elles ne constituent pas des limites absolues. La simulation conditionne toutefois la vérité initiale à la zone de recherche, comme la grille de croyance normalisée. Une réalisation peut donc commencer à plus de 20 NM du datum ou à plus de 60 minutes de l’heure estimée, mais pas hors de la zone à l’instant initial.

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

Le modèle actif de la branche 2.4.4 est un Behavior Cloning réentraîné sur le scénario `4 NM / 20 NM / 60 min`. L’interface métier ne propose ni sélection de modèle, ni entraînement, ni réglages PPO.

## Comment le modèle a été entraîné

Le modèle n’a pas appris seul en prenant des risques dans le monde réel. Il a appris dans un simulateur, en trois temps :

1. imiter un expert qui choisit la zone la plus probable ;
2. essayer d’améliorer ses décisions avec PPO ;
3. conserver la version la plus sûre après comparaison sur 500 situations inédites.

Le modèle historique, entraîné avec la seed 2027 sur l'ancien scénario, atteignait 198 interceptions sur 500 dans le nouvel environnement Python. Le Behavior Cloning 2.4.4, entraîné avec la seed 2026, en obtient 275 sur 500 dans sa version ONNX, sans Bingo ni sortie de zone. Dans la campagne navigateur corrigée et appariée de 250 situations, il obtient 96 interceptions, contre 89 pour le POMDP approché à un pas et 32 pour le râteau. L'écart entre l'hybride et le POMDP n'est pas présenté comme statistiquement significatif. Ces résultats ne justifient ni PPO ni AutoResearch.

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
