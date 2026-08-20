"""Modèles comparés.

Trois niveaux, en escalier :

1. Baseline naïve       -- prédit toujours la classe majoritaire. Plancher.
2. Régression Ridge     -- avec seuils de découpe optimisés.
3. LightGBM             -- en régression à seuils, et en classification directe.

L'approche par régression à seuils mérite un mot. Plutôt que de traiter les huit
niveaux comme des catégories indépendantes, on prédit une valeur continue puis
on la découpe en huit intervalles dont les bornes sont optimisées pour maximiser
le kappa quadratique. Cette approche exploite l'ordre de la cible, ce que la
classification multi-classes ignore. C'était la stratégie des meilleures
solutions de la compétition Prudential, et elle domine nettement ici.
"""
import numpy as np
import pandas as pd
from scipy.optimize import fmin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from metrics import kappa_quadratique

try:
    import lightgbm as lgb
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    lgb = None
    LIGHTGBM_DISPONIBLE = False


class DecoupeurOptimise:
    """Transforme une prédiction continue en niveau de risque entier.

    Les sept bornes séparant les huit classes sont ajustées par optimisation
    directe du kappa quadratique sur le jeu d'apprentissage. L'optimisation
    utilise Nelder-Mead, méthode sans gradient : la fonction objectif est en
    escalier, donc non dérivable, ce qui exclut toute descente de gradient.
    """

    def __init__(self, n_classes=8):
        self.n_classes = n_classes
        self.bornes_ = None

    @staticmethod
    def _appliquer(y_continu, bornes):
        return np.digitize(y_continu, np.sort(np.asarray(bornes))) + 1

    def fit(self, y_continu, y_vrai):
        # Point de départ : bornes placées aux quantiles observés de la cible.
        quantiles = [
            np.quantile(y_continu, (y_vrai <= k).mean())
            for k in range(1, self.n_classes)
        ]
        depart = np.asarray(quantiles, dtype=float)

        def objectif(bornes) -> float:
            return float(-kappa_quadratique(y_vrai, self._appliquer(y_continu, bornes)))

        # fmin applique l'algorithme de Nelder-Mead, méthode sans gradient.
        bornes_optimales = fmin(
            objectif,
            depart,
            maxiter=2000,
            xtol=1e-3,
            ftol=1e-5,
            disp=False,
        )
        self.bornes_ = np.sort(np.asarray(bornes_optimales))
        return self

    def transform(self, y_continu):
        if self.bornes_ is None:
            raise RuntimeError("Appeler fit() avant transform().")
        pred = self._appliquer(y_continu, self.bornes_)
        return np.clip(pred, 1, self.n_classes).astype(int)


class ModeleBaseline:
    """Prédit systématiquement la classe majoritaire. Référence plancher."""

    nom = "baseline_majoritaire"

    def __init__(self, seed=42):
        self.modele = DummyClassifier(strategy="most_frequent", random_state=seed)

    def fit(self, X, y):
        self.modele.fit(X, y)
        return self

    def predict(self, X):
        return self.modele.predict(X).astype(int)


class ModeleRidgeSeuils:
    """Régression linéaire régularisée, suivie d'un découpage optimisé.

    La standardisation est indispensable ici : Ridge pénalise la norme des
    coefficients, donc des variables d'échelles différentes seraient pénalisées
    inégalement.
    """

    nom = "ridge_seuils"

    def __init__(self, alpha=1.0, n_classes=8, seed=42):
        self.scaler = StandardScaler()
        self.modele = Ridge(alpha=alpha, random_state=seed)
        self.decoupeur = DecoupeurOptimise(n_classes)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.modele.fit(Xs, y)
        self.decoupeur.fit(self.modele.predict(Xs), np.asarray(y))
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.decoupeur.transform(self.modele.predict(Xs))


class ModeleLGBMSeuils:
    """Gradient boosting en régression, suivi d'un découpage optimisé.

    C'est le modèle principal de l'étude. Le boosting capte les interactions
    entre variables sans qu'il faille les spécifier, et la standardisation
    est inutile pour un modèle à base d'arbres.
    """

    nom = "lgbm_seuils"

    def __init__(self, n_classes=8, seed=42, **params):
        if not LIGHTGBM_DISPONIBLE:
            raise ImportError("LightGBM absent : pip install lightgbm")
        defauts = dict(
            objective="regression",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        defauts.update(params)
        self.modele = lgb.LGBMRegressor(**defauts)
        self.decoupeur = DecoupeurOptimise(n_classes)

    def fit(self, X, y):
        self.modele.fit(X, y)
        self.decoupeur.fit(self.modele.predict(X), np.asarray(y))
        return self

    def predict(self, X):
        return self.decoupeur.transform(self.modele.predict(X))

    def importances(self, noms_colonnes):
        """Classement des variables par usage dans les arbres du modèle.

        Complément facultatif à l'étude d'ablation : celle-ci mesure combien
        l'assureur perd, les importances indiquent quelles variables portaient
        l'information disparue.
        """
        return (
            pd.Series(self.modele.feature_importances_, index=noms_colonnes)
            .sort_values(ascending=False)
        )


class ModeleLGBMClassification:
    """Gradient boosting en classification multi-classes.

    Sert de témoin. Ce modèle ignore l'ordre de la cible : il traite les huit
    niveaux comme des catégories sans relation. La comparaison avec la version
    à seuils mesure donc exactement ce que rapporte l'exploitation de l'ordre.
    """

    nom = "lgbm_classification"

    def __init__(self, seed=42, **params):
        if not LIGHTGBM_DISPONIBLE:
            raise ImportError("LightGBM absent : pip install lightgbm")
        defauts = dict(
            objective="multiclass",
            num_class=8,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            colsample_bytree=0.7,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        defauts.update(params)
        self.modele = lgb.LGBMClassifier(**defauts)

    def fit(self, X, y):
        self.modele.fit(X, y)
        return self

    def predict(self, X):
        return self.modele.predict(X).astype(int)


def catalogue(seed=42, rapide=False) -> list:
    """Modèles à comparer. `rapide=True` allège pour un premier essai."""
    modeles: list = [ModeleBaseline(seed), ModeleRidgeSeuils(seed=seed)]
    if LIGHTGBM_DISPONIBLE:
        n = 200 if rapide else 600
        modeles.append(ModeleLGBMSeuils(seed=seed, n_estimators=n))
        if not rapide:
            modeles.append(ModeleLGBMClassification(seed=seed))
    return modeles