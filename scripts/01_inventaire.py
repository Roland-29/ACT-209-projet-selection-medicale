"""Étape 1 - Inventaire du jeu de données.

Produit un état des lieux exploitable directement dans la partie « données »
du rapport : volumétrie, structure par bloc, manquants, distribution de la
cible, et vérification de la couverture des configurations d'ablation.

Usage :  python scripts/01_inventaire.py
"""
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "src"))

import config
import data


def titre(texte):
    print(f"\n{'=' * 70}\n{texte}\n{'=' * 70}")


def main():
    df = data.charger()

    titre("VOLUMÉTRIE")
    print(f"Observations : {len(df):,}".replace(",", " "))
    print(f"Colonnes     : {df.shape[1]}")
    print(f"Mémoire      : {df.memory_usage(deep=True).sum() / 1e6:.1f} Mo")

    titre("DISTRIBUTION DE LA CIBLE")
    vc = df[config.CIBLE].value_counts().sort_index()
    tab = pd.DataFrame({"effectif": vc, "part_pct": (vc / len(df) * 100).round(2)})
    print(tab.to_string())
    print(f"\nRapport classe majoritaire / minoritaire : {vc.max() / vc.min():.1f}")
    print("Le déséquilibre est marqué. La stratification du découpage est")
    print("indispensable, et l'exactitude brute sera une métrique trompeuse.")

    titre("STRUCTURE PAR BLOC DE VARIABLES")
    blocs = {
        "Produit": config.BLOC_PRODUIT,
        "Démographique": config.BLOC_DEMOGRAPHIQUE,
        "Morphologique": config.BLOC_MORPHOLOGIQUE,
        "Emploi": config.BLOC_EMPLOI,
        "Assuré": config.BLOC_ASSURE,
        "Historique assurance": config.BLOC_HISTORIQUE_ASSURANCE,
        "Antécédents familiaux": config.BLOC_ANTECEDENTS_FAMILIAUX,
        "Antécédents médicaux": config.BLOC_ANTECEDENTS_MEDICAUX,
        "Mots-clés médicaux": config.BLOC_MOTS_CLES_MEDICAUX,
    }
    lignes = []
    for nom, cols in blocs.items():
        presentes = [c for c in cols if c in df.columns]
        absentes = sorted(set(cols) - set(presentes))
        taux_na = df[presentes].isna().mean().mean() * 100 if presentes else 0
        lignes.append({
            "bloc": nom,
            "attendues": len(cols),
            "presentes": len(presentes),
            "manquants_moyen_pct": round(taux_na, 1),
            "absentes": ", ".join(absentes) if absentes else "-",
        })
    print(pd.DataFrame(lignes).to_string(index=False))

    titre("COLONNES À FORT TAUX DE MANQUANTS")
    na = df.isna().mean()
    na = (na[na > 0].sort_values(ascending=False) * 100).round(1)
    print(na.to_string())
    print("\nLecture : au-delà de 90 %, l'absence traduit vraisemblablement une")
    print("question non posée plutôt qu'une donnée perdue. Le prétraitement")
    print("conserve un indicateur de manquant et abandonne la colonne elle-même.")

    titre("COUVERTURE DES CONFIGURATIONS D'ABLATION")
    lignes = []
    for cle in config.ORDRE_CONFIGURATIONS:
        cfg = config.CONFIGURATIONS[cle]
        presentes = [c for c in cfg["colonnes"] if c in df.columns]
        lignes.append({
            "configuration": cle,
            "libelle": cfg["libelle"],
            "variables": len(presentes),
        })
    tab = pd.DataFrame(lignes)
    ref = tab.loc[tab.configuration == "complete", "variables"].iloc[0]
    tab["part_de_la_reference_pct"] = (tab["variables"] / ref * 100).round(1)
    print(tab.to_string(index=False))

    titre("CONTRÔLES DE COHÉRENCE")
    print(f"Doublons sur Id            : {df['Id'].duplicated().sum()}")
    print(f"Lignes intégralement vides : {df.isna().all(axis=1).sum()}")
    hors = ~df[config.CIBLE].between(1, config.N_CLASSES)
    print(f"Cibles hors [1, 8]         : {hors.sum()}")
    txt = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    print(f"Colonnes textuelles        : {txt}")

    chemin = config.OUTPUTS / "inventaire_manquants.csv"
    na.to_csv(chemin, header=["taux_manquants_pct"])
    print(f"\nDétail des manquants écrit dans {chemin.relative_to(RACINE)}")


if __name__ == "__main__":
    main()