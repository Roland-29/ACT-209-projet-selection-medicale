"""Chargement des données et découpage apprentissage / test."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Rend config.py importable depuis src/, quel que soit le dossier d'exécution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def charger(chemin=None) -> pd.DataFrame:
    """Charge le fichier de données et vérifie sa conformité."""
    chemin = Path(chemin) if chemin else config.CHEMIN_TRAIN

    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}\n"
            "Les données ne sont pas versionnées dans ce dépôt. Téléchargez "
            "train.csv depuis la page Kaggle de la compétition 'Prudential Life "
            "Insurance Assessment' (un compte et l'acceptation des règles sont "
            "nécessaires), puis placez-le dans le dossier data/."
        )

    df = pd.read_csv(chemin)

    if config.CIBLE not in df.columns:
        raise ValueError(
            f"Colonne cible '{config.CIBLE}' absente du fichier {chemin.name}.\n"
            "Cause probable : le fichier test.csv de la compétition a été utilisé "
            "à la place de train.csv. Le fichier de test ne porte pas la cible, "
            "celle-ci n'étant connue que de l'organisateur. Seul train.csv est "
            "exploitable pour ce projet."
        )

    return df


def decouper(df: pd.DataFrame, seed=None, test_size=None):
    """Découpage stratifié sur la cible.

    Chaque ligne est un dossier de souscription indépendant : le jeu n'a ni
    structure temporelle ni répétition d'individus. Un découpage aléatoire
    est donc légitime, contrairement à ce qui vaudrait sur des données de panel.

    La stratification garantit que les huit niveaux de risque conservent les
    mêmes proportions en apprentissage et en test. Une vérification sur trente
    graines montre que le tirage simple fait varier sensiblement l'effectif des
    classes rares : sur le niveau 3, l'écart-type atteint 12,1 dossiers pour un
    effectif attendu de 253, soit un coefficient de variation de 4,7 %, contre
    0,87 % seulement pour le niveau 8, le plus fréquent. La stratification annule
    entièrement cette variabilité, sans contrepartie.
    """
    seed = seed if seed is not None else config.SEED
    test_size = test_size if test_size is not None else config.TEST_SIZE

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[config.CIBLE],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def separer_xy(df: pd.DataFrame, colonnes: list):
    """Isole la matrice des prédicteurs et le vecteur cible.

    `colonnes` filtre les prédicteurs selon la configuration d'ablation.
    Les colonnes absentes du fichier sont ignorées et signalées, ce qui rend
    le code robuste aux variantes de version du jeu de données.
    """
    presentes = [c for c in colonnes if c in df.columns]
    manquantes = sorted(set(colonnes) - set(presentes))
    X = df[presentes].copy()
    y = df[config.CIBLE].copy()
    return X, y, manquantes