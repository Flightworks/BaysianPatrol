# AGENTS.md — BaysianPatrol v2.4.0 hybride / Monte-Carlo

Ce fichier est le contrat de travail pour tout agent intervenant sur le dépôt. **Python/Gymnasium est la référence canonique du contrat RL** ; TypeScript/ONNX doit reproduire exactement ses observations, actions, règles de sécurité et critères terminaux.

## 1. Mission et règle de sécurité

BaysianPatrol compare trois stratégies de recherche SAR maritime : râteau naïf, SIGMA bayésien et politique RL. Le critère principal est le taux d'interception sur des scénarios identiques et reproductibles.

Le Bingo carburant et les limites de zone sont des **contraintes dures**, pas des comportements que le réseau doit apprendre par pénalité :

- le superviseur calcule `fuel_margin = fuel_remaining - return_time - bingo_buffer` ;
- quand la marge approche zéro, il ignore l'action RL et force le retour frégate ;
- le geofence borne tout waypoint à la zone de recherche ;
- un retour réussi se termine par `safe_rtb` / `SAFE_RTB` ;
- un retour normal ne doit jamais être compté comme violation Bingo.

Résultats opérationnels communs :

- `intercepted` / `INTERCEPTED` ;
- `safe_rtb` / `SAFE_RTB` ;
- `bingo` / `BINGO_VIOLATION` ;
- `out_of_bounds` / `OUT_OF_BOUNDS` ;
- `OUT_OF_FUEL` ;
- `time_limit` / `TIME_LIMIT`.

## 2. Contrat RL v2.3.1

### Observation `grid` — `(2, 32, 32)`

- Canal 0 : posterior bayésien mis à l'échelle par son pic pour l'entrée CNN (`P / max(P)`, donc plage 0–1).
- Canal 1 : mémoire continue de balayage radar, plage 0–1 avec décroissance temporelle.

La distribution probabiliste interne reste séparément normalisée avec `sum(P) = 1`. Ne jamais remplacer la distribution interne par sa version peak-scaled.

Après une non-détection, appliquer réellement :

```text
posterior = prior × (1 - Pdet)
posterior /= sum(posterior)
```

Le filtre de croyance ne doit jamais utiliser le véritable cap caché de la cible.

### Observation `vector` — 10 valeurs

Ordre immuable, identique dans Python et TypeScript :

```text
0  sin(cap hélicoptère)
1  cos(cap hélicoptère)
2  vitesse / vitesse_max
3  marge_carburant / endurance
4  dx frégate relatif
5  dy frégate relatif
6  dx pic de croyance relatif
7  dy pic de croyance relatif
8  entropie normalisée de la croyance
9  temps écoulé / endurance
```

N'ajouter aucune variable non causale. Le vent n'a pas besoin d'être exposé si son effet est déjà intégré au filtre de croyance.

### Action — waypoint relatif

`Box([-1,-1], [1,1])` :

```text
action[0] = décalage X relatif du waypoint
 action[1] = décalage Y relatif du waypoint
```

Le réseau ne pilote plus directement virage et accélération. Un autopilote déterministe vole vers le waypoint à la vitesse de recherche. Le safety shield et le geofence s'appliquent ensuite. Toute modification de cette sémantique doit être faite simultanément dans :

- `python/baysian_patrol_env.py` ;
- `src/engine/missionContract.ts` ;
- `src/engine/rlAlgorithm.ts` ;
- l'export ONNX et ses tests.

## 3. Entraînement recommandé

Pipeline principal : `python/hybrid_train.py`.

1. Générer des démonstrations avec `env.expert_action()`.
2. Initialiser la politique par Behavior Cloning.
3. Affiner avec PPO selon le curriculum 1 → 4.
4. Évaluer expert, BC et PPO sur le **même jeu fixe de seeds**.
5. Exporter sous `PPO_CANDIDATE_*.onnx`.
6. Ne promouvoir vers `public/models/baysian_patrol_policy.onnx` qu'après passage des gates.

Curriculum :

1. cible fixe ;
2. cible mobile simple ;
3. dérive stochastique ;
4. dynamique complète.

L'ancienne `autoresearch_v3.py` n'est pas la voie principale pour v2.3.1. Elle ne doit être réutilisée qu'après obtention d'une baseline hybride saine, pour des ablations limitées autour de cette baseline. Ne jamais relancer une recherche large d'hyperparamètres pour masquer un défaut de contrat.

## 4. Gates de promotion d'un modèle

Un modèle candidat ne remplace le modèle actif que si :

- évaluation finale sur au moins 500 seeds fixes et séparées de l'entraînement ;
- `bingo_fail_rate == 0` ;
- `out_of_bounds_rate == 0` ;
- taux d'interception au moins équivalent à la meilleure baseline sur les mêmes seeds ;
- borne inférieure de Wilson 95 % publiée ;
- équivalence Python/ONNX, erreur absolue maximale `< 1e-4` sur au moins 100 observations ;
- build TypeScript réussi ;
- test de trajectoires de référence navigateur/Python réussi.

Les modèles `PPO_CANDIDATE_*` et les rapports de smoke test ne sont pas automatiquement opérationnels. Le candidat court seed 77 produit pendant le développement était volontairement insuffisant et ne doit pas être activé.

## 5. Monte-Carlo

Les comparaisons doivent utiliser des réalisations appariées et reproductibles :

- `ScenarioConfig.monteCarloSeed` contrôle la suite ; défaut 2026 ;
- `SeededRandom` remplace `Math.random()` pour cible et détection ;
- chaque stratégie reçoit la même réalisation, la même trajectoire cible et des tirages radar communs ;
- `bingoRate` mesure uniquement une violation de réserve, pas le déclenchement d'un retour normal ;
- `safeReturnRate` est publié séparément.

La grille Monte-Carlo conserve trois cartes (`classical`, `bayesianStandard`, `bayesianEvolved`) et applique une mise à jour négative après balayage. Les règles physiques partagées avec Python doivent rester cohérentes : vitesse, endurance, réserve, portée, déplacement cible, limites et critères terminaux.

La stratégie de référence `NAIVE` est un **Parallel Sweep IAMSAR** déterministe :

- point de départ dans le coin le plus proche, à un demi-espacement de piste des deux bords ;
- branches parallèles au grand côté de la zone ;
- espacement constant dérivé de la largeur de balayage radar ;
- retour frégate prioritaire dès le seuil carburant.

L'interface métier ne doit exposer ni entraînement, ni hyperparamètres PPO, ni sélection de fichier ONNX. Le modèle qualifié seed 2027 est la stratégie hybride active. Le workflow est limité à `Comparaison`, `Carte tactique` et `Historique`; les vingt derniers résumés de campagne sont conservés localement sans leurs trajectoires.

## 6. Fichiers importants

```text
python/baysian_patrol_env.py        environnement canonique
python/hybrid_train.py              expert → BC → curriculum PPO
python/export_onnx.py               export acteur SB3 réel
python/tests/test_env_v231.py       contrat RL et sécurité
python/tests/test_hybrid_train.py   seeds, évaluation et démonstrations
src/engine/missionContract.ts       contrat navigateur partagé
src/engine/rlAlgorithm.ts           observation ONNX + autopilote + RTB
src/engine/bayesianGrid.ts          posterior Monte-Carlo
src/engine/random.ts                PRNG seedé
src/engine/targetGenerator.ts       réalisation et trajectoire cible
src/engine/simulator.ts             Monte-Carlo apparié et outcomes
src/engine/iamsarPattern.ts         plan de balayage parallèle IAMSAR
src/engine/runHistory.ts            historique compact local
tests-ts/missionContract.test.ts    tests du contrat TypeScript
tests-ts/naiveIamsar.test.ts        géométrie du râteau IAMSAR
tests-ts/runHistory.test.ts         persistance des campagnes
```

## 7. Commandes de vérification

Depuis le dépôt :

```bash
# Python
python -m unittest discover -s python/tests -v

# Contrat TypeScript pur (Node >=22)
node --experimental-strip-types --test tests-ts/*.test.ts

# Frontend
npm install
npm run build

# Smoke test hybride sans promotion automatique
python python/hybrid_train.py \
  --demo-episodes 40 --bc-epochs 5 \
  --ppo-steps-per-level 1024 --n-envs 2 \
  --eval-episodes 30 --seed 77
```

Sur HP (`192.168.1.238`), privilégier quatre environnements CPU. La GTX 950M `sm_50` n'est pas une cible PyTorch moderne recommandée.

## 8. Discipline de modification

- TDD : écrire et exécuter le test rouge avant tout changement de comportement.
- Seeds d'entraînement et d'évaluation séparées.
- Comparer tous les candidats sur les mêmes seeds.
- La récompense est diagnostique ; le taux d'interception et la sécurité déterminent le classement.
- Ne jamais écraser le modèle actif pendant une campagne.
- Ne jamais publier un résultat provenant d'un seul seed comme preuve de convergence.
