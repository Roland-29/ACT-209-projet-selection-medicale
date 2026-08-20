"""Métriques adaptées à une cible ordinale.

La cible compte huit niveaux de risque ordonnés. Une erreur de 1 vers 2 n'a pas
la même portée métier qu'une erreur de 1 vers 8 : dans le premier cas la
tarification est légèrement décalée, dans le second elle est absurde. Les
métriques de classification usuelles — exactitude, F1 — traitent ces deux cas
de façon identique et sont donc inadaptées.

La métrique de référence retenue est le kappa quadratique pondéré, qui pénalise
chaque erreur proportionnellement au carré de sa distance sur l'échelle. C'était
également la métrique officielle de la compétition Prudential, ce qui permet de
situer nos résultats par rapport à ceux publiés.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def kappa_quadratique(y_vrai, y_pred) -> float:
    """Kappa de Cohen à pondération quadratique.

    Vaut 1 pour un accord parfait, 0 pour un accord équivalent au hasard,
    et peut devenir négatif si la prédiction fait pire que le hasard.
    """
    return cohen_kappa_score(y_vrai, y_pred, weights="quadratic")


def ecart_absolu_moyen(y_vrai, y_pred) -> float:
    """Écart moyen exprimé en nombre de niveaux de risque.

    Métrique de lecture directe : une valeur de 0,8 signifie que le modèle se
    trompe en moyenne de moins d'un niveau.
    """
    return float(np.mean(np.abs(np.asarray(y_vrai) - np.asarray(y_pred))))


def taux_erreur_grave(y_vrai, y_pred, seuil=2) -> float:
    """Part des dossiers mal classés de plus de `seuil` niveaux.

    Lecture métier : ce sont les erreurs susceptibles de conduire à une
    tarification franchement inadaptée, voire à un refus injustifié.
    """
    ecarts = np.abs(np.asarray(y_vrai) - np.asarray(y_pred))
    return float(np.mean(ecarts > seuil))


def evaluer(y_vrai, y_pred) -> dict:
    """Jeu complet de métriques pour une prédiction."""
    return {
        "kappa_quadratique": kappa_quadratique(y_vrai, y_pred),
        "exactitude": accuracy_score(y_vrai, y_pred),
        "ecart_absolu_moyen": ecart_absolu_moyen(y_vrai, y_pred),
        "taux_erreur_grave": taux_erreur_grave(y_vrai, y_pred),
    }


def matrice_confusion(y_vrai, y_pred, n_classes=8) -> pd.DataFrame:
    """Matrice de confusion lisible, indexée par niveau de risque.

    Les lignes portent le niveau réel, les colonnes le niveau prédit. Une
    concentration hors diagonale révèle les confusions systématiques du modèle.
    """
    labels = list(range(1, n_classes + 1))
    m = confusion_matrix(y_vrai, y_pred, labels=labels)
    return pd.DataFrame(
        m,
        index=[f"reel_{i}" for i in labels],
        columns=[f"pred_{i}" for i in labels],
    )


def comparer(resultats: dict) -> pd.DataFrame:
    """Tableau comparatif des configurations, avec écart à la référence.

    `resultats` est un dictionnaire dont les clés sont les noms de
    configuration et les valeurs les dictionnaires produits par `evaluer`.
    L'écart est calculé par rapport à la configuration complète, qui sert de
    référence haute : c'est cet écart qui chiffre le coût informationnel.
    """
    tab = pd.DataFrame(resultats).T
    if "complete" in tab.index:
        ref = tab.loc["complete", "kappa_quadratique"]
        tab["ecart_kappa"] = tab["kappa_quadratique"] - ref
        tab["perte_relative_pct"] = (tab["ecart_kappa"] / ref * 100).round(1)
    return tab.round(4)