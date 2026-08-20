"""Préparation des variables.

Deux principes guident ce module.

1. Aucun ajustement n'est appris sur le jeu de test. Les statistiques
   d'imputation et les modalités d'encodage sont calculées sur le seul jeu
   d'apprentissage, puis appliquées au test.

2. Le fait qu'une valeur soit manquante est en soi une information. Certaines
   colonnes d'antécédents sont vides à plus de 90 % : leur absence traduit
   vraisemblablement une question non posée plutôt qu'une donnée perdue.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

SEUIL_ABANDON = 0.95   # au-delà, on ne garde que l'indicateur de manquant


class Preparateur:
    """Prépare les variables. Interface proche de scikit-learn."""

    def __init__(self, seuil_abandon=SEUIL_ABANDON, indicateurs_manquants=True):
        self.seuil_abandon = seuil_abandon
        self.indicateurs_manquants = indicateurs_manquants
        self.medianes_ = {}
        self.modalites_ = {}
        self.colonnes_abandonnees_ = []
        self.colonnes_avec_manquants_ = []
        self.colonnes_finales_ = None

    def fit(self, X: pd.DataFrame):
        """Apprend les paramètres sur le jeu d'APPRENTISSAGE uniquement."""
        X = X.copy()
        taux_na = X.isna().mean()

        self.colonnes_abandonnees_ = taux_na[taux_na > self.seuil_abandon].index.tolist()
        self.colonnes_avec_manquants_ = taux_na[taux_na > 0].index.tolist()

        # Modalités des catégorielles textuelles, apprises sur le train seul.
        for col in config.CATEGORIELLES_TEXTE:
            if col in X.columns:
                self.modalites_[col] = sorted(X[col].dropna().unique().tolist())

        numeriques = [
            c for c in X.columns
            if c not in self.colonnes_abandonnees_
            and c not in config.CATEGORIELLES_TEXTE
        ]
        for col in numeriques:
            self.medianes_[col] = X[col].median()

        self.colonnes_finales_ = self._transformer(X).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._transformer(X.copy())
        if self.colonnes_finales_ is not None:
            out = out.reindex(columns=self.colonnes_finales_, fill_value=0)
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # -- interne --------------------------------------------------------------
    def _transformer(self, X: pd.DataFrame) -> pd.DataFrame:
        # 1. Indicateurs de manquant, AVANT toute imputation.
        if self.indicateurs_manquants:
            for col in self.colonnes_avec_manquants_:
                if col in X.columns:
                    X[f"{col}_manquant"] = X[col].isna().astype(np.int8)

        # 2. Abandon des colonnes quasi vides.
        X = X.drop(columns=[c for c in self.colonnes_abandonnees_ if c in X.columns])

        # 3. Encodage des catégorielles textuelles.
        for col, modalites in self.modalites_.items():
            if col not in X.columns:
                continue
            X[f"{col}_lettre"] = (
                X[col].astype(str).str[0].astype("category").cat.codes.astype(np.int16)
            )
            codes = {m: i for i, m in enumerate(modalites)}
            X[col] = X[col].map(codes).fillna(-1).astype(np.int16)

        # 4. Imputation par la médiane apprise sur le train.
        for col, mediane in self.medianes_.items():
            if col in X.columns:
                X[col] = X[col].fillna(mediane)

        X = self._ajouter_derivees(X.copy())
        return X.fillna(0)

    @staticmethod
    def _ajouter_derivees(X: pd.DataFrame) -> pd.DataFrame:
        """Variables construites à partir des variables existantes.

        Chaque dérivée est conditionnée à la présence effective de ses sources
        dans X. Cette précaution est indispensable à l'intégrité de l'étude
        d'ablation : construire une variable à partir d'une colonne censée être
        retirée réintroduirait par la bande l'information que la configuration
        prétend supprimer. En configuration Lemoine stricte, aucune variable
        morphologique n'est disponible, donc aucune dérivée correspondante
        n'est créée. Rien ne signalerait cette erreur à l'exécution : l'écart
        mesuré entre configurations serait simplement faux.
        """
        if "BMI" in X.columns and "Ins_Age" in X.columns:
            X["BMI_x_Age"] = X["BMI"] * X["Ins_Age"]

        if "Ht" in X.columns and "Wt" in X.columns:
            X["Wt_sur_Ht"] = X["Wt"] / X["Ht"].replace(0, np.nan)

        mots_cles = [c for c in X.columns if c.startswith("Medical_Keyword_")]
        if mots_cles:
            X["nb_mots_cles"] = X[mots_cles].sum(axis=1)

        return X