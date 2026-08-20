"""Étape 3 - Expérience d'ablation.

Cœur du projet. Pour chaque configuration d'information (complète, sans
antécédents, Lemoine stricte), on entraîne les mêmes modèles sur le même
découpage, puis on compare leurs performances sur le même jeu de test.

Point de méthode : le découpage apprentissage / test est fixé une fois pour
toutes, en amont de la boucle. Les configurations ne diffèrent donc que par
les variables disponibles, jamais par les individus observés. Sans cette
précaution, l'écart mesuré mélangerait l'effet de l'ablation et celui de
l'échantillonnage.

Usage :
    python scripts/03_ablation.py            # expérience complète
    python scripts/03_ablation.py --rapide   # version allégée, pour un essai
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "src"))

import config
import data
import metrics
from features import Preparateur
from models import catalogue


def titre(texte):
    print(f"\n{'=' * 70}\n{texte}\n{'=' * 70}")


def lancer_configuration(cle, train, test, rapide=False, verbeux=True):
    """Entraîne et évalue tous les modèles pour une configuration donnée."""
    cfg = config.CONFIGURATIONS[cle]

    X_tr, y_tr, absentes = data.separer_xy(train, cfg["colonnes"])
    X_te, y_te, _ = data.separer_xy(test, cfg["colonnes"])

    if verbeux:
        print(f"\n--- {cfg['libelle']} ---")
        print(f"Variables brutes : {X_tr.shape[1]}")
        if absentes:
            print(f"Attendues mais absentes du fichier : {absentes}")

    prep = Preparateur()
    X_tr_p = prep.fit_transform(X_tr)
    X_te_p = prep.transform(X_te)

    if verbeux:
        print(f"Variables après préparation : {X_tr_p.shape[1]}")
        if prep.colonnes_abandonnees_:
            print(f"Abandonnées (>95% vides) : {prep.colonnes_abandonnees_}")

    resultats, predictions = {}, {}
    for modele in catalogue(seed=config.SEED, rapide=rapide):
        t0 = time.time()
        modele.fit(X_tr_p, y_tr)
        y_pred = modele.predict(X_te_p)
        scores = metrics.evaluer(y_te, y_pred)
        scores["duree_s"] = round(time.time() - t0, 1)
        resultats[modele.nom] = scores
        predictions[modele.nom] = y_pred
        if verbeux:
            print(f"  {modele.nom:24s} kappa={scores['kappa_quadratique']:.4f}  "
                  f"EAM={scores['ecart_absolu_moyen']:.3f}  "
                  f"({scores['duree_s']}s)")

    return resultats, predictions, y_te, X_tr_p.columns.tolist()


def main():
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--rapide", action="store_true",
                         help="version allégée, pour vérifier que tout tourne")
    args = parseur.parse_args()

    titre("EXPÉRIENCE D'ABLATION")
    df = data.charger()
    train, test = data.decouper(df)
    print(f"Apprentissage : {len(train):,}".replace(",", " "))
    print(f"Test          : {len(test):,}".replace(",", " "))
    print(f"Graine        : {config.SEED}")
    if args.rapide:
        print("\nMode rapide : résultats indicatifs, non exploitables pour le rapport.")

    tous, toutes_preds = {}, {}
    y_te = None
    for cle in config.ORDRE_CONFIGURATIONS:
        res, preds, y_te, colonnes = lancer_configuration(
            cle, train, test, rapide=args.rapide
        )
        tous[cle] = res
        toutes_preds[cle] = preds

    # --- Synthèse par modèle -------------------------------------------------
    titre("RÉSULTATS : KAPPA QUADRATIQUE PAR CONFIGURATION")
    noms_modeles = list(next(iter(tous.values())).keys())
    tableau = pd.DataFrame(
        {cle: {m: tous[cle][m]["kappa_quadratique"] for m in noms_modeles}
         for cle in config.ORDRE_CONFIGURATIONS}
    ).round(4)
    print(tableau.to_string())

    # --- Le résultat central -------------------------------------------------
    meilleur = tableau["complete"].idxmax()
    titre(f"COÛT INFORMATIONNEL (modèle {meilleur})")
    ligne = tableau.loc[meilleur]
    ref = ligne["complete"]
    synthese = pd.DataFrame({
        "kappa": ligne,
        "ecart_absolu": (ligne - ref).round(4),
        "perte_relative_pct": ((ligne - ref) / ref * 100).round(1),
    })
    synthese.index = [config.CONFIGURATIONS[c]["libelle"] for c in synthese.index]
    print(synthese.to_string())

    perte_lemoine = (ligne["lemoine_stricte"] - ref) / ref * 100
    apport_imc = ligne["sans_antecedents"] - ligne["lemoine_stricte"]
    print("\nLecture :")
    print(f"  Perte totale sous régime Lemoine : {perte_lemoine:.1f} % du kappa.")
    print(f"  Dont apport propre de la morphologie : {apport_imc:.4f} de kappa,")
    print("  soit l'écart entre « sans antécédents » et « Lemoine stricte ».")
    print("  Cet écart mesure ce que coûte le retrait de la taille et du poids,")
    print("  que la loi supprime aussi puisqu'ils passent par le questionnaire.")

    # --- Matrices de confusion ----------------------------------------------
    titre("MATRICES DE CONFUSION")
    for cle in ["complete", "lemoine_stricte"]:
        print(f"\n{config.CONFIGURATIONS[cle]['libelle']} :")
        m = metrics.matrice_confusion(y_te, toutes_preds[cle][meilleur])
        print(m.to_string())
        m.to_csv(config.OUTPUTS / f"confusion_{cle}.csv")

    # --- Où se dégrade la prédiction ----------------------------------------
    titre("DÉGRADATION PAR NIVEAU DE RISQUE RÉEL")
    y_te = np.asarray(y_te)
    lignes = []
    for niveau in range(1, config.N_CLASSES + 1):
        masque = y_te == niveau
        if masque.sum() == 0:
            continue
        ligne_n = {"niveau": niveau, "effectif": int(masque.sum())}
        for cle in ["complete", "lemoine_stricte"]:
            pred = np.asarray(toutes_preds[cle][meilleur])[masque]
            ligne_n[f"EAM_{cle}"] = round(float(np.mean(np.abs(pred - niveau))), 3)
        ligne_n["degradation"] = round(
            ligne_n["EAM_lemoine_stricte"] - ligne_n["EAM_complete"], 3
        )
        lignes.append(ligne_n)
    degradation = pd.DataFrame(lignes)
    print(degradation.to_string(index=False))
    print("\nLes niveaux dont la dégradation est la plus forte désignent les profils")
    print("que l'assureur ne parvient plus à situer sans le questionnaire.")

    # --- Écriture ------------------------------------------------------------
    tableau.to_csv(config.OUTPUTS / "ablation_kappa.csv")
    degradation.to_csv(config.OUTPUTS / "ablation_degradation.csv", index=False)
    detail = pd.concat(
        {cle: pd.DataFrame(tous[cle]).T for cle in config.ORDRE_CONFIGURATIONS}
    )
    detail.to_csv(config.OUTPUTS / "ablation_detail.csv")
    print(f"\nRésultats écrits dans {config.OUTPUTS.relative_to(RACINE)}/")


if __name__ == "__main__":
    main()