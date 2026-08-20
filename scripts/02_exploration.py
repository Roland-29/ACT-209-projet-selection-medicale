"""Étape 2 - Analyse exploratoire.

Produit les figures de la partie « analyse » du rapport.

Usage :  python scripts/02_exploration.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "src"))

import config
import data

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLEU, ROUGE = "#2E75B6", "#C00000"


def sauver(fig, nom):
    chemin = config.FIGURES / nom
    fig.savefig(chemin, bbox_inches="tight")
    plt.close(fig)
    print(f"  écrit : {chemin.relative_to(RACINE)}")


def fig_cible(df):
    vc = df[config.CIBLE].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(vc.index, vc.values, color=BLEU)
    ax.set_xlabel("Niveau de risque attribué")
    ax.set_ylabel("Nombre de dossiers")
    ax.set_title("Distribution de la cible")
    for x, v in zip(vc.index, vc.values):
        ax.text(x, v, f"{v / len(df) * 100:.1f}%", ha="center", va="bottom", fontsize=7)
    sauver(fig, "01_distribution_cible.png")


def fig_manquants(df):
    na = df.isna().mean()
    na = (na[na > 0].sort_values() * 100)
    fig, ax = plt.subplots(figsize=(6, 4))
    couleurs = [ROUGE if v > 90 else BLEU for v in na.values]
    ax.barh(range(len(na)), na.values, color=couleurs)
    ax.set_yticks(range(len(na)))
    ax.set_yticklabels(na.index, fontsize=7)
    ax.set_xlabel("Taux de valeurs manquantes (%)")
    ax.set_title("Colonnes incomplètes")
    ax.axvline(90, color=ROUGE, ls="--", lw=0.8)
    sauver(fig, "02_manquants.png")


def fig_risque_par_variable(df, variable, n_bacs=10):
    """Niveau de risque moyen par décile d'une variable continue."""
    if variable not in df.columns:
        return
    d = df[[variable, config.CIBLE]].dropna()
    d["bac"] = pd.qcut(d[variable], n_bacs, duplicates="drop", labels=False)
    moy = d.groupby("bac")[config.CIBLE].mean()
    centres = d.groupby("bac")[variable].mean()

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(centres.values, moy.values, "o-", color=BLEU)
    ax.set_xlabel(f"{variable} (valeur normalisée)")
    ax.set_ylabel("Niveau de risque moyen")
    ax.set_title(f"Risque moyen selon {variable}")
    sauver(fig, f"03_risque_par_{variable}.png")


def fig_contribution_blocs(df):
    """Corrélation absolue moyenne à la cible, par bloc de variables."""
    blocs = {
        "Produit": config.BLOC_PRODUIT,
        "Âge": config.BLOC_DEMOGRAPHIQUE,
        "Morphologie": config.BLOC_MORPHOLOGIQUE,
        "Emploi": config.BLOC_EMPLOI,
        "Assuré": config.BLOC_ASSURE,
        "Hist. assurance": config.BLOC_HISTORIQUE_ASSURANCE,
        "Antéc. familiaux": config.BLOC_ANTECEDENTS_FAMILIAUX,
        "Antéc. médicaux": config.BLOC_ANTECEDENTS_MEDICAUX,
        "Mots-clés méd.": config.BLOC_MOTS_CLES_MEDICAUX,
    }
    resultats = {}
    for nom, cols in blocs.items():
        cols = [c for c in cols if c in df.columns
                and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            continue
        corrs = df[cols].corrwith(df[config.CIBLE]).abs().dropna()
        if len(corrs):
            resultats[nom] = corrs.mean()

    s = pd.Series(resultats).sort_values()
    medicaux = {"Antéc. familiaux", "Antéc. médicaux", "Mots-clés méd.", "Morphologie"}
    couleurs = [ROUGE if n in medicaux else BLEU for n in s.index]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(range(len(s)), s.values, color=couleurs)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel("Corrélation absolue moyenne avec la cible")
    ax.set_title("Lien brut à la cible, par bloc\n(rouge : blocs issus du questionnaire de santé)")
    sauver(fig, "04_contribution_blocs.png")
    return s


def fig_nb_mots_cles(df):
    cols = [c for c in config.BLOC_MOTS_CLES_MEDICAUX if c in df.columns]
    if not cols:
        return
    d = df[[config.CIBLE]].copy()
    d["nb"] = df[cols].sum(axis=1)
    d["nb_groupe"] = np.minimum(d["nb"], 6)
    moy = d.groupby("nb_groupe")[config.CIBLE].mean()

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(moy.index, moy.values, color=BLEU)
    ax.set_xlabel("Nombre de mots-clés médicaux actifs (plafonné à 6)")
    ax.set_ylabel("Niveau de risque moyen")
    ax.set_title("Fardeau pathologique déclaré et risque attribué")
    sauver(fig, "05_mots_cles.png")


def main():
    df = data.charger()
    print(f"Données chargées : {df.shape}\nGénération des figures :")

    fig_cible(df)
    fig_manquants(df)
    for v in ["Ins_Age", "BMI"]:
        fig_risque_par_variable(df, v)
    s = fig_contribution_blocs(df)
    fig_nb_mots_cles(df)

    if s is not None:
        chemin = config.OUTPUTS / "correlation_par_bloc.csv"
        s.sort_values(ascending=False).to_csv(chemin, header=["correlation_moyenne"])
        print(f"\nTableau écrit : {chemin.relative_to(RACINE)}")
        print("\nLien brut à la cible, par bloc :")
        print(s.sort_values(ascending=False).round(4).to_string())
        print("\nAttention : une corrélation linéaire moyenne sous-estime les blocs")
        print("composés de nombreux indicateurs binaires rares, dont l'effet est")
        print("individuellement faible mais collectivement important. C'est l'ablation")
        print("qui tranchera, pas ce graphique.")


if __name__ == "__main__":
    main()