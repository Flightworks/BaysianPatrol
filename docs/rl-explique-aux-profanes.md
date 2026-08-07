# Comment la stratégie hybride a été entraînée

## En une phrase

Le modèle a appris à rechercher une cible maritime en observant les décisions d'un pilote automatique expert dans un simulateur, puis il a été testé dans des situations nouvelles. Il n'a jamais eu le droit de sacrifier la sécurité pour gagner du temps.

Le terme "RL" signifie *Reinforcement Learning*, ou apprentissage par renforcement. Dans ce projet, le modèle n'a pas appris seul depuis zéro. La méthode utilisée est hybride :

1. on lui montre d'abord de bons exemples ;
2. il essaie ensuite de faire mieux dans le simulateur ;
3. on ne conserve sa version améliorée que si elle reste au moins aussi sûre.

---

## L'analogie du pilote élève

Imaginez un élève pilote dans un simulateur de vol.

- Le simulateur lui donne une situation : position de l'hélicoptère, météo, carburant et zone où la cible a le plus de chances de se trouver.
- Un pilote expert indique une décision raisonnable.
- L'élève essaie de reproduire cette décision.
- Ensuite, l'élève s'entraîne seul et reçoit une bonne note lorsqu'il détecte la cible, mais une très mauvaise note s'il met l'hélicoptère en danger.
- Un instructeur compare enfin l'élève avec la version précédente sur une série de situations inconnues.

Baysian Patrol suit exactement cette logique, avec une différence importante : le modèle ne commande pas directement le cap et la vitesse. Il indique un **point vers lequel chercher**. Un pilote automatique déterministe effectue ensuite le déplacement.

```text
situation observée
        ↓
modèle hybride : "cherche vers ce point"
        ↓
waypoint relatif
        ↓
pilote automatique
        ↓
limites de zone + surveillance carburant + retour sûr
```

Cette séparation évite qu'une sortie aberrante du modèle commande directement une manœuvre dangereuse.

---

## Ce que le modèle voit

À chaque étape, le modèle reçoit deux types d'information :

- une carte de probabilité : les endroits où la cible est probablement présente, ainsi que la mémoire des zones déjà balayées ;
- quelques informations de situation : position de l'hélicoptère, direction, vitesse, carburant disponible, position de la frégate, position de la zone probable et niveau d'incertitude.

La carte n'est pas figée. Si le temps passe sans détection, la zone probable se déplace et s'élargit. Le modèle doit donc chercher dans une incertitude vivante, pas dans un simple cercle dessiné une fois pour toutes.

### Ce que le modèle produit

Le modèle produit deux nombres compris entre -1 et 1. Ils représentent une direction relative dans la zone de recherche. Ces deux nombres deviennent un waypoint.

Il ne produit pas directement :

- un ordre de virage moteur par moteur ;
- une vitesse dangereuse ;
- une autorisation d'entrer dans une zone interdite ;
- une permission de dépasser la réserve de retour.

Ces protections restent hors du réseau et sont donc prévisibles.

---

## Étape 1 : construire un expert

L'expert n'est pas un autre réseau neuronal. C'est une règle de décision calculée dans l'environnement Gymnasium : il regarde le sommet de la probabilité actuelle et propose de se diriger vers cette zone.

L'environnement applique ensuite les règles de mission :

- évolution de la cible ;
- déplacement et élargissement de la croyance ;
- détection radar ;
- consommation de carburant ;
- géofence ;
- retour vers la frégate lorsque la marge devient insuffisante.

Cet expert fournit des exemples reproductibles. Le programme collecte **500 épisodes de démonstration**, en faisant varier progressivement la difficulté : cible immobile, cible mobile, incertitude plus forte et mouvements plus difficiles.

Chaque exemple contient :

```text
ce que l'hélicoptère voyait → le waypoint choisi par l'expert
```

---

## Étape 2 : Behavior Cloning, ou imitation

Le *Behavior Cloning* est une forme d'apprentissage supervisé. Le modèle reçoit les exemples de l'expert et ajuste ses paramètres pour produire des waypoints proches de ceux-ci.

C'est comparable à un élève qui étudie 500 vols annotés avant de prendre les commandes.

Dans le pipeline utilisé :

- le modèle commence avec une architecture capable de lire la carte et les informations de situation ;
- il apprend pendant **20 passages** sur les exemples ;
- l'erreur mesurée compare le waypoint produit avec celui de l'expert ;
- les grandes corrections sont limitées pour éviter des mises à jour instables.

Cette étape donne le modèle **BC** (*Behavior Cloning*). Elle sert aussi de référence de sécurité : si l'étape suivante fait moins bien, on peut conserver cette version.

---

## Étape 3 : affinement avec PPO

PPO signifie *Proximal Policy Optimization*. Le modèle essaie des décisions dans le simulateur et reçoit une récompense :

- détecter la cible est fortement récompensé ;
- réduire l'incertitude est utile ;
- perdre du temps coûte un peu ;
- un Bingo carburant est fortement pénalisé ;
- une sortie de zone est également pénalisée.

Le mot "proximal" est important : PPO limite l'ampleur des changements entre deux mises à jour. Le modèle ne doit pas oublier brutalement ce qu'il savait déjà.

L'affinement suit quatre niveaux de difficulté. Pour chaque niveau, le modèle est entraîné avec **50 000 étapes par niveau**, sur quatre environnements parallèles. Il apprend donc progressivement au lieu de commencer directement par les situations les plus difficiles.

PPO produit un candidat. Il ne devient pas automatiquement le modèle utilisé par l'application.

---

## Étape 4 : choisir entre BC et PPO

La sélection ne se fait pas sur la récompense moyenne seule. Une politique qui gagne du temps mais met un appareil en danger est rejetée.

L'ordre de décision est le suivant :

1. moins de violations Bingo ;
2. moins de sorties de zone ;
3. meilleur taux de réussite avec une marge statistique prudente ;
4. meilleur temps moyen d'interception.

La marge statistique utilisée est la borne inférieure de Wilson à 95 %. Elle répond à une question simple : "Même en tenant compte de l'incertitude du nombre d'essais, quel taux de réussite minimal peut-on raisonnablement garantir ?"

Le candidat est comparé à l'expert et à la copie BC sur **500 seeds fixes**, de `200000` à `200499`. Les mêmes seeds permettent de comparer les candidats sur les mêmes situations.

Le garde-fou d'acceptation exige notamment :

- zéro Bingo ;
- zéro sortie de zone ;
- une borne de réussite qui ne soit pas inférieure de plus de deux points à celle de l'expert.

Sur les trois entraînements indépendants réalisés avec les seeds 2026, 2027 et 2028 :

| entraînement | version retenue | interceptions | Bingo | sorties de zone | temps moyen |
|---|---:|---:|---:|---:|---:|
| seed 2026 | BC | 500/500 | 0 | 0 | 9,610 min |
| seed 2027 | BC | 500/500 | 0 | 0 | 9,608 min |
| seed 2028 | BC | 500/500 | 0 | 0 | 9,596 min |

Le résultat est clair : PPO n'a pas apporté un gain suffisant pour remplacer BC. La stratégie active retenue pour l'application est donc la version **Behavior Cloning issue de l'entraînement seed 2027**.

Cela explique pourquoi l'interface ne propose pas de choisir entre PPO et BC : ce choix a été fait pendant la qualification, pas pendant une mission opérationnelle.

---

## Étape 5 : rendre le modèle utilisable dans le navigateur

L'entraînement est fait avec Python et Gymnasium, qui restent la référence. Pour l'application web, le modèle retenu est exporté au format ONNX.

L'export n'est pas accepté sur simple présence du fichier. Une vérification compare les sorties Python et ONNX sur les mêmes observations. Le seuil de différence est `0,0001`.

Les écarts mesurés pour les trois entraînements sont de l'ordre de `3 × 10⁻⁷`, donc largement sous le seuil. Le navigateur utilise ainsi le même comportement que le modèle qualifié en Python.

La session ONNX est chargée une seule fois par campagne et réutilisée. Elle n'est pas recréée à chaque tirage Monte-Carlo.

---

## Ce qui protège encore la mission après le modèle

Même un modèle correctement entraîné peut rencontrer une situation inhabituelle. Le réseau ne possède donc pas le dernier mot.

Le moteur conserve des protections déterministes :

- le waypoint est borné à la zone de recherche ;
- le pilote automatique contrôle le déplacement ;
- la distance de retour est calculée à chaque étape ;
- la réserve carburant est conservée ;
- le retour sûr vers la frégate est déclenché avant le Bingo ;
- les résultats `SAFE_RTB`, `BINGO_VIOLATION`, `OUT_OF_BOUNDS` et `TIME_LIMIT` restent distincts.

Le réseau propose une direction de recherche. Il ne peut pas désactiver ces règles.

---

## Ce que ces résultats veulent dire, et ce qu'ils ne veulent pas dire

Les 1 500 épisodes qualifiés montrent que le modèle est stable sur le protocole de qualification utilisé. Ils ne prouvent pas qu'il réussira toutes les missions réelles.

Il reste nécessaire de tester séparément :

- tempête et fort clapot ;
- très forte incertitude sur le datum ;
- cible rapide ou très manœuvrante ;
- zones de recherche inhabituelles ;
- changements de cap de la frégate ;
- limites du capteur radar.

Le simulateur mesure le comportement. Il ne remplace ni la doctrine opérationnelle, ni la décision de l'équipage, ni une certification réglementaire.

---

## Où retrouver les éléments techniques

- Environnement canonique : [`python/baysian_patrol_env.py`](../python/baysian_patrol_env.py)
- Pipeline expert → imitation → PPO : [`python/hybrid_train.py`](../python/hybrid_train.py)
- Qualification multi-seed : [`python/hybrid_v231_qualification.json`](../python/hybrid_v231_qualification.json)
- Export web : [`python/export_onnx.py`](../python/export_onnx.py)
- Tests de contrat Python : [`python/tests`](../python/tests)

Commandes de vérification :

```bash
python -m unittest discover -s python/tests -v
node --experimental-strip-types --test tests-ts/*.test.ts
npm run build
```
