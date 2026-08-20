# ACT-209 — Automatisation de la sélection médicale à la souscription

**Impact de la suppression du questionnaire de santé sur la classification du risque**

Projet d'évaluation — cours IA & Assurance, 2025-2026.

---

## Question

La loi Lemoine du 28 février 2022 a supprimé le questionnaire médical pour les prêts immobiliers inférieurs à 200 000 € remboursés avant le 60ᵉ anniversaire de l'emprunteur. Ce travail mesure l'information dont l'assureur se trouve privé de ce fait, par une étude d'ablation : un même modèle de classification du risque est entraîné avec, puis sans, le déclaratif de santé.

## Données

Jeu **Prudential Life Insurance Assessment** (Kaggle) : 59 381 dossiers de souscription, 126 variables prédictives anonymisées, cible ordinale de risque à huit niveaux issue de la décision réelle du souscripteur.

Le fichier `train.csv` n'est pas versionné. Téléchargez-le depuis la page de la compétition et placez-le dans `data/`.

La cible est fortement déséquilibrée : le niveau 8 rassemble 32,8 % des dossiers contre 1,7 % pour le niveau 3, soit un rapport de 19 pour 1.

## Les trois configurations

| Configuration      | Variables | Contenu                                                  |
|--------------------|-----------|----------------------------------------------------------|
| `complete`         | 126       | Toutes les variables. Régime antérieur à la loi.         |
| `sans_antecedents` | 32        | Antécédents médicaux et familiaux retirés, IMC conservé. |
| `lemoine_stricte`  | 29        | Tout le déclaratif de santé retiré, IMC compris.         |

La configuration intermédiaire n'est pas décorative. Taille et poids sont recueillis **par** le questionnaire de santé : sous Lemoine, l'assureur les perd aussi. Beaucoup les considèrent pourtant comme des données non médicales. L'écart entre `sans_antecedents` et `lemoine_stricte` isole précisément ce que vaut ce bloc morphologique.

## Résultats

Kappa quadratique pondéré sur le jeu de test (25 %, découpage stratifié, graine 42) :

| Modèle                  | Complète   | Sans antécédents | Lemoine stricte |
|-------------------------|------------|------------------|-----------------|
| Baseline majoritaire    | 0,0000     | 0,0000           | 0,0000          |
| Ridge + seuils          | 0,6112     | 0,4732           | 0,3540          |
| **LightGBM + seuils**   | **0,6567** | **0,5201**       | **0,4249**      |
| LightGBM classification | 0,5660     | 0,3938           | 0,3057          |

**Résultat central : la suppression du questionnaire de santé fait perdre 35,3 % du pouvoir discriminant du modèle.**

Trois observations complètent ce chiffre.

**Le bloc morphologique pèse démesurément.** Taille, poids et IMC représentent 0,095 de kappa, soit 41 % de la perte totale — pour trois variables, face aux quatre-vingt-quatorze du bloc médical.

**L'exploitation de l'ordre de la cible rapporte plus que le choix de l'algorithme.** L'approche par régression à seuils optimisés domine systématiquement la classification multi-classes, de 9 points de kappa en configuration complète et de 12 en configuration dégradée. Le modèle le plus sophistiqué perd parce qu'il ignore que les niveaux sont ordonnés.

**Ce sont les bons risques qui deviennent invisibles.** Le niveau 8, celui des meilleurs profils, subit la plus forte dégradation de l'échelle : son écart absolu moyen passe de 0,51 à 1,29. En configuration complète, 3 328 dossiers de niveau 8 sur 4 873 sont correctement classés ; sous Lemoine, 2 242 seulement, dont 843 basculent en niveau 6. Privé du questionnaire, l'assureur perd d'abord sa capacité à reconnaître un bon risque — le mécanisme même de l'antisélection.

Le niveau 4 fait exception, avec une dégradation légèrement négative de −0,025. L'effectif concerné est faible (357 dossiers) et l'écart tient au bruit d'échantillonnage.

## Structure

```
config.py                   Chemins, graine, blocs de variables, configurations d'ablation
src/data.py                 Chargement, découpage stratifié
src/features.py             Manquants, encodage, variables dérivées
src/metrics.py              Kappa quadratique, écart absolu, erreurs graves
src/models.py               Baseline, Ridge, LightGBM, optimiseur de seuils
scripts/01_inventaire.py    État des lieux des données
scripts/02_exploration.py   Figures de l'analyse exploratoire
scripts/03_ablation.py      Expérience d'ablation
outputs/                    Tableaux CSV et figures
```

## Exécution

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ~/Downloads/train.csv data/

python3 scripts/01_inventaire.py
python3 scripts/02_exploration.py
python3 scripts/03_ablation.py            # environ 4 minutes
python3 scripts/03_ablation.py --rapide   # version allégée, pour vérifier l'installation
```

## Choix méthodologiques

**Le découpage est fixé en amont de la boucle d'ablation.** Les trois configurations partagent exactement les mêmes individus en apprentissage et en test. Sans cette précaution, l'écart mesuré mélangerait l'effet de l'ablation et celui de l'échantillonnage.

**Les statistiques d'imputation sont apprises sur le seul jeu d'apprentissage.** Toute quantité estimée à partir du jeu de test rendrait l'évaluation optimiste et donc inutilisable.

**Les variables dérivées ne survivent pas à l'ablation de leurs sources.** Construire un indicateur à partir d'une variable censée être retirée réintroduirait l'information par la bande. Rien ne signalerait cette erreur à l'exécution : l'écart mesuré serait simplement faux.

**Les colonnes vides à plus de 95 % sont abandonnées, mais un indicateur de manquant est conservé.** À ce niveau — 99,1 % pour `Medical_History_10` —, l'absence traduit une question non posée plutôt qu'une donnée perdue : c'est en soi une information.

**Le découpage est stratifié.** Une vérification sur trente graines montre que le tirage simple fait varier l'effectif des classes rares avec un coefficient de variation de 4,7 % sur le niveau 3, contre 0,87 % sur le niveau 8. La stratification annule cette variabilité sans contrepartie.

## Limites

Les données proviennent d'un assureur américain exerçant en assurance vie individuelle : les résultats donnent un ordre de grandeur, non un chiffrage transposable au marché français.

Les variables ayant été anonymisées et normalisées, les conclusions portent sur des blocs d'information et non sur des pathologies identifiables.

La cible est la décision d'un souscripteur humain, non une sinistralité observée. Le modèle reproduit donc une pratique de tarification, avec les biais qu'elle peut comporter, plutôt qu'un risque réel.

Le rapprochement avec la loi Lemoine est un apport propre à ce travail : le jeu de données avait été publié pour une question d'efficacité du parcours de souscription.