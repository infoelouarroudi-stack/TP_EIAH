# TP Gamification adaptative — Rendu (M2 EIAH)

---

## 1. Contexte et objectif

Ce TP porte sur un environnement d'apprentissage gamifié où un **élément de jeu** (avatar, badges, score, timer, progression, classement) est attribué à chaque élève.

L'objectif est de construire une recommandation d'éléments de jeu à partir du modèle utilisateur (profil joueur HEXAD et motivations initiales) afin **d'augmenter** la motivation intrinsèque et extrinsèque et de **réduire** l'amotivation.

---

## 2. Données et fichiers fournis

### 2.1. Données élèves (`userStats.csv`)

Chaque ligne correspond à un élève. Le fichier contient notamment :

- `User` (identifiant) et `GameElement` (élément attribué)
- Profil HEXAD : `achiever`, `player`, `socialiser`, `freeSpirit`, `disruptor`, `philanthropropist`
- Motivations initiales détaillées : `micoI`, `miacI`, `mistI`, `meidI`, `meinI`, `mereI`, `amotI` (ainsi que versions finales et variations)

### 2.2. Matrices PLS (24 fichiers au total)

Le TP fournit des matrices issues d'analyses PLS (coefficients + p-values) pour relier :

- **HEXAD → variations de motivation** (6 éléments × (coeff + p-values))
- **Motivations initiales → variations de motivation** (6 éléments × (coeff + p-values))

Les p-values permettent de filtrer les effets (significatifs si p < 0.05, ou p < 0.1 selon la tolérance).

---

## 3. Question 1 — Commentaire de 2 matrices PLS (seuil p < 0.1)

### 3.1. Règle d'analyse adoptée

- On commente uniquement les coefficients dont la p-value est **< 0.1** (approche tolérante)
- On interprète les signes par rapport à l'objectif du TP : augmenter MIVar et MEVar, diminuer amotVar

---

### 3.2. Matrice 1 : Avatar (dossier Motivation)

#### Description

- Fichier coefficients : `avatarPathCoefs.csv` (lignes : `MIVar`, `MEVar`, `amotVar` ; colonnes : `MI`, `ME`, `amotI`)
- Fichier p-values : `avatarpVals.csv`

#### Effets significatifs (p < 0.1)

- **amotI → MIVar** : coef = **0.4231**, p = **0.00467** ⇒ effet positif significatif
- **MI → MEVar** : coef = **0.4079**, p = **0.0381** ⇒ effet positif significatif

Les autres coefficients ne sont pas retenus à ce seuil (p ≥ 0.1).

#### Interprétation (Avatar)

- L'effet **amotI → MIVar** positif suggère que, pour des élèves **initialement plus amotivés** (amotI élevé), l'élément *Avatar* est associé à une hausse de motivation intrinsèque (MIVar) dans ce modèle PLS.
- L'effet **MI → MEVar** positif suggère que plus la motivation intrinsèque initiale est élevée, plus *Avatar* est associé à une hausse de motivation extrinsèque (MEVar).
- Aucun effet retenu (p < 0.1) n'indique une baisse claire de l'amotivation (amotVar), donc on évite de conclure sur cet aspect pour *Avatar*.

---

### 3.3. Matrice 2 : Ranking / Classement (dossier Motivation)

#### Description

- Fichier coefficients : `rankingPathCoefs.csv` (lignes : `MIVar`, `MEVar`, `amotVar` ; colonnes : `MI`, `ME`, `amotI`)
- Fichier p-values : `rankingpVals.csv`

#### Effets significatifs (p < 0.1)

- **amotI → MIVar** : coef = **0.6501**, p = **9.15e-05** ⇒ effet très significatif et positif
- **amotI → MEVar** : coef = **-0.3679**, p = **0.0499** ⇒ effet significatif mais négatif
- Aucun coefficient lié à `MI` ou `ME` n'est significatif à p < 0.1
- Aucun effet sur `AMOTVar` n'est significatif à p < 0.1

#### Interprétation (Ranking)

- L'effet **amotI → MIVar** positif indique que, pour des élèves **initialement plus amotivés**, *Ranking* est associé à une hausse de motivation intrinsèque (MIVar) dans le modèle PLS.
- En revanche, l'effet **amotI → MEVar** négatif va contre l'objectif "augmenter ME", car il suggère une baisse de MEVar quand amotI est élevé.
- Comme pour Avatar, l'absence d'effets significatifs sur `amotVar` (au seuil p < 0.1) empêche de conclure sur la réduction d'amotivation via `amotVar`.

---

## 4. Étape 2 — Code Python (vecteurs d'affinité + recommandation)

### 4.1. Principe de calcul

Pour chaque élément de jeu, on prédit un vecteur de variations `[MIVar, MEVar, amotVar]` à partir :

- du profil HEXAD (vecteur de 6 dimensions) via les matrices du dossier `Hexad/`
- des motivations initiales (vecteur `[MI, ME, amotI]`) via les matrices du dossier `Motivation/`

On applique un filtrage des coefficients par p-value (ici **alpha = 0.1**) avant de calculer la prédiction.

Ensuite, on convertit la prédiction en un score unique (objectif TP) :

**score = MIVar + MEVar − amotVar**

On obtient deux classements (2 vecteurs d'affinité), puis une recommandation finale par combinaison (normalisation + moyenne pondérée).

---

## 5. Résultats d'exécution (élève `elevebf01`, alpha = 0.1)

### 5.1. Commande exécutée

```bash
python main.py --user elevebf01 --alpha 0.1
```

---

### 5.2. Vecteur d'affinité HEXAD

| Rang | Élément  | Score     | pred_MIVar | pred_MEVar | pred_amotVar |
|-----:|----------|----------:|-----------:|-----------:|-------------:|
|    1 | avatar   |  2.994505 |   0.000000 |   1.750452 |    -1.244053 |
|    2 | score    |  1.941858 |   2.939885 |  -0.998027 |     0.000000 |
|    3 | timer    |  1.118727 |   0.000000 |   0.000000 |    -1.118727 |
|    4 | ranking  |  0.000000 |   0.000000 |   0.000000 |     0.000000 |
|    5 | progress | -0.081618 |  -0.530964 |   0.449346 |     0.000000 |
|    6 | badges   | -0.110957 |   0.000000 |  -2.209982 |    -2.099025 |

---

### 5.3. Vecteur d'affinité MOTIVATION

| Rang | Élément  | Score      | pred_MIVar | pred_MEVar | pred_amotVar |
|-----:|----------|------------|------------|------------|--------------|
|    1 | avatar   |  8.446508  |  4.231411  |  4.215096  |  0.0         |
|    2 | timer    |  6.166849  |  6.166849  |  0.000000  |  0.0         |
|    3 | ranking  |  2.822368  |  6.501379  | -3.679011  |  0.0         |
|    4 | score    |  1.474547  |  6.236665  | -4.762118  |  0.0         |
|    5 | badges   | -1.706427  |  4.590641  | -6.297068  |  0.0         |
|    6 | progress | -4.620518  |  4.873957  | -9.494475  |  0.0         |

---

### 5.4. Combinaison finale (recommandation)

| Rang | Élément  | score_final |
|-----:|----------|------------:|
|    1 | avatar   |    1.000000 |
|    2 | timer    |    0.610758 |
|    3 | score    |    0.563740 |
|    4 | ranking  |    0.302661 |
|    5 | badges   |    0.111506 |
|    6 | progress |    0.004724 |

**Recommandation finale : avatar**

---

## 6. Interprétation des résultats (elevebf01)

1. La recommandation finale **avatar** est cohérente car *Avatar* est classé **1er** à la fois dans le vecteur HEXAD et dans le vecteur Motivation, donc la combinaison le conserve naturellement en tête.

2. Côté **HEXAD**, le score élevé d'Avatar vient du fait que certains chemins vers `MEVar` (et/ou `amotVar`) passent le filtre p < 0.1, ce qui permet d'obtenir une prédiction non nulle après masquage par p-values.

3. Côté **Motivation**, les matrices commentées en Question 1 montrent déjà que *Avatar* possède des effets significatifs à p < 0.1 (ex : `amotI → MIVar` et `MI → MEVar`), ce qui explique qu'il ressorte fortement dans le classement Motivation.

4. Le fait que `pred_amotVar = 0` apparaisse souvent (dans la partie Motivation) est compatible avec une situation où aucun coefficient vers `amotVar` n'est significatif au seuil choisi, donc ils ont été mis à 0 par le filtrage p-values.

---

## 7. Comment exécuter (reproductibilité)

### 7.1. Structure du projet

Placer les fichiers dans `data/` selon la structure attendue.

### 7.2. Lancement

```bash
python main.py --user elevebf01 --alpha 0.1
```

### 7.3. Sorties

Les CSV de sortie sont générés dans `outputs/` (vecteur HEXAD, vecteur Motivation, combinaison).

---

## 8. Livrables à rendre

- **Texte Question 1** (commentaire de 2 matrices PLS) : section 3 de ce README
- **Code Python** : `main.py` + `src/recommender.py` + dossier `data/` (ou au minimum les scripts + instructions)
- **Résultats d'exécution** : tables de la section 5 (ou CSV dans `outputs/`)
