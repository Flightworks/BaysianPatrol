# Stratégie hybride et campagnes Monte-Carlo

## À quoi sert le simulateur ?

Baysian Patrol compare trois façons de rechercher une cible maritime :

- la stratégie hybride issue de l'apprentissage ;
- la recherche bayésienne ;
- le balayage parallèle en râteau inspiré IAMSAR.

L'objectif n'est pas de produire une jolie trajectoire. Il faut mesurer, dans les mêmes conditions, le taux de détection, le temps d'interception, la consommation de carburant et les retours sûrs.

## Monte-Carlo, en pratique

Une simulation unique ne suffit pas. Le résultat dépend toujours de paramètres qui varient d'une mission à l'autre :

- position initiale du datum et de la frégate ;
- position réelle de la cible ;
- vent et dérive ;
- vitesse et cap de la cible ;
- tirages de détection radar.

Une campagne Monte-Carlo répète donc la mission un grand nombre de fois. À chaque répétition, le simulateur tire une nouvelle situation plausible, puis rejoue les trois stratégies.

```text
seed de campagne
      ↓
250 situations reproductibles
      ↓
chaque situation est jouée par les 3 stratégies
      ↓
statistiques : détection, temps, carburant, sécurité
```

### Pourquoi une seed ?

La seed est le point de départ du générateur aléatoire. Avec la même seed, on retrouve les mêmes situations. Cela permet de comparer les stratégies équitablement : elles rencontrent la même météo, la même cible et les mêmes tirages radar.

La campagne n'est donc pas « au hasard » au sens d'un résultat impossible à refaire. Elle est aléatoire dans la construction des cas, mais reproductible pour l'analyse.

### Comment lire une campagne ?

- **Taux de détection** : proportion des missions où la cible est interceptée.
- **Temps moyen** : durée moyenne des interceptions réussies.
- **Retour sûr** : l'hélicoptère retourne à la frégate avant d'atteindre le Bingo carburant.
- **Violation Bingo** : la marge nécessaire au retour n'était plus disponible.
- **Sortie de zone** : l'hélicoptère a quitté la zone de recherche autorisée.
- **Borne inférieure de Wilson** : estimation prudente du taux de réussite, utile quand le nombre de cas reste limité.

Un bon résultat ne se résume donc pas au taux de détection. Une stratégie légèrement plus lente mais sans Bingo peut être préférable à une stratégie rapide qui met l'aéronef en difficulté.

## Comment le modèle a appris

Le modèle actif n'a pas appris seul dans le monde réel. Il a été entraîné dans l'environnement canonique Python/Gymnasium, qui simule la cible, l'incertitude, le radar, le carburant et le retour vers la frégate.

### 1. Un expert fournit les exemples

L'expert est une règle déterministe : il se dirige vers la zone où la probabilité de présence est la plus forte, tout en respectant les contraintes du simulateur.

Le programme collecte 500 épisodes d'exemple en augmentant progressivement la difficulté : cible immobile, cible mobile, incertitude plus large et mouvements plus difficiles.

### 2. Le modèle apprend par imitation

Cette étape s'appelle Behavior Cloning. Le modèle reçoit :

- une carte de probabilité et de zones déjà observées ;
- la position, le cap, la vitesse et le carburant ;
- la position de la frégate et le niveau d'incertitude.

Il apprend à produire une décision proche de celle de l'expert. La décision est un **waypoint relatif**, c'est-à-dire un point vers lequel chercher, et non une commande directe de moteur ou de gouverne.

### 3. PPO affine le comportement

PPO est une méthode d'apprentissage par renforcement. Le modèle essaie des décisions dans le simulateur et reçoit une note :

- détection : récompense importante ;
- réduction de l'incertitude : récompense utile ;
- temps consommé : petite pénalité ;
- Bingo ou sortie de zone : forte pénalité.

L'apprentissage se fait sur quatre niveaux de difficulté, avec 50 000 étapes par niveau. PPO produit un candidat, mais ce candidat n'est pas accepté automatiquement.

### 4. La sécurité décide

Le modèle Behavior Cloning et le candidat PPO sont rejoués sur les mêmes 500 seeds fixes. Le choix suit cet ordre :

1. zéro Bingo ;
2. zéro sortie de zone ;
3. meilleur taux de réussite prudent, calculé avec la borne de Wilson à 95 % ;
4. meilleur temps moyen.

Les trois entraînements indépendants, seeds 2026, 2027 et 2028, ont tous retenu Behavior Cloning :

| seed | résultat | détection | Bingo | sorties de zone | temps moyen |
|---:|---|---:|---:|---:|---:|
| 2026 | Behavior Cloning | 500/500 | 0 | 0 | 9,610 min |
| 2027 | Behavior Cloning | 500/500 | 0 | 0 | 9,608 min |
| 2028 | Behavior Cloning | 500/500 | 0 | 0 | 9,596 min |

La stratégie active de l'application est donc la version Behavior Cloning seed 2027. Le choix n'est pas laissé à l'utilisateur pendant une campagne.

## Ce que le réseau ne peut pas faire

Le réseau propose une direction de recherche, mais il ne pilote pas directement l'aéronef. Le moteur conserve les protections suivantes :

```text
modèle → waypoint relatif → pilote automatique → geofence → contrôle carburant/RTB
```

- le waypoint reste dans la zone autorisée ;
- le pilote automatique effectue le déplacement ;
- la distance de retour est recalculée ;
- la réserve carburant est conservée ;
- le retour sûr est déclenché avant le Bingo.

Les résultats `SAFE_RTB`, `BINGO_VIOLATION`, `OUT_OF_BOUNDS` et `TIME_LIMIT` restent séparés. Un retour sûr n'est pas compté comme un échec de sécurité.

## Export vers le navigateur

Python/Gymnasium reste la référence. Le modèle retenu est exporté en ONNX pour le navigateur, puis comparé à sa version Python sur les mêmes observations.

Le seuil d'écart accepté est `0,0001`. Les écarts mesurés sont de l'ordre de `3 × 10⁻⁷`. La session ONNX est chargée une seule fois par campagne et réutilisée pour les tirages suivants.

## Limites des résultats

Les 1 500 épisodes qualifiés montrent une bonne stabilité sur le protocole utilisé. Ils ne constituent pas une garantie pour toutes les missions.

Il faut encore évaluer séparément les cas difficiles : tempête, très forte incertitude du datum, cible rapide ou manœuvrante, zone inhabituelle et changements de cap de la frégate.

Le simulateur aide à comparer les stratégies. Il ne remplace ni la doctrine opérationnelle, ni l'équipage, ni une certification réglementaire.

## Références dans le dépôt

- Environnement : [`python/baysian_patrol_env.py`](../python/baysian_patrol_env.py)
- Entraînement : [`python/hybrid_train.py`](../python/hybrid_train.py)
- Qualification : [`python/hybrid_v231_qualification.json`](../python/hybrid_v231_qualification.json)
- Export web : [`python/export_onnx.py`](../python/export_onnx.py)

Vérification locale :

```bash
python -m unittest discover -s python/tests -v
node --experimental-strip-types --test tests-ts/*.test.ts
npm run build
```
