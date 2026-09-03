"""Démonstration - sélection médicale automatisée sur un dossier individuel.

Illustre concrètement ce que fait le modèle : il prend un dossier de
souscription et attribue un niveau de risque, comme le ferait un souscripteur.

Le même dossier est soumis à deux modèles entraînés séparément :
l'un dispose du questionnaire de santé, l'autre non. La comparaison montre,
sur un cas particulier, ce que la loi Lemoine change pour l'assureur.

Usage :
    python3 scripts/04_demo.py              # trois dossiers contrastés
    python3 scripts/04_demo.py --id 12345   # un dossier précis
    python3 scripts/04_demo.py --n 5        # cinq dossiers au hasard
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Sortie lisible en démonstration : les avertissements de fragmentation
# de pandas n'ont aucune incidence sur les résultats.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "src"))

import config
import data
from features import Preparateur
from models import ModeleLGBMSeuils

LARGEUR = 68


def titre(texte):
    print(f"\n{'=' * LARGEUR}\n{texte}\n{'=' * LARGEUR}")


def entrainer(train, test, cle, verbeux=True):
    """Entraîne le modèle de référence dans une configuration donnée."""
    cfg = config.CONFIGURATIONS[cle]
    X_tr, y_tr, _ = data.separer_xy(train, cfg["colonnes"])
    X_te, _, _ = data.separer_xy(test, cfg["colonnes"])

    prep = Preparateur()
    X_tr_p = prep.fit_transform(X_tr)
    X_te_p = prep.transform(X_te)

    modele = ModeleLGBMSeuils(seed=config.SEED)
    modele.fit(X_tr_p, y_tr)

    if verbeux:
        print(f"  {cfg['libelle']:34s} {X_tr.shape[1]:3d} variables")
    return modele, X_te_p


def profil(ligne):
    """Résume un dossier en quelques caractéristiques lisibles."""
    elements = []
    for col, libelle in [("Ins_Age", "Âge"), ("BMI", "IMC"),
                         ("Ht", "Taille"), ("Wt", "Poids")]:
        if col in ligne.index and pd.notna(ligne[col]):
            elements.append(f"{libelle} {ligne[col]:.3f}")

    mots = [c for c in config.BLOC_MOTS_CLES_MEDICAUX if c in ligne.index]
    if mots:
        elements.append(f"{int(ligne[mots].sum())} mot(s)-clé(s) médical(aux)")

    antecedents = [c for c in config.BLOC_ANTECEDENTS_MEDICAUX if c in ligne.index]
    if antecedents:
        renseignes = int(ligne[antecedents].notna().sum())
        elements.append(f"{renseignes}/{len(antecedents)} antécédents renseignés")

    return "   ".join(elements)


def afficher(dossier_id, reel, pred_complete, pred_lemoine, resume):
    print(f"\n  Dossier n° {dossier_id}")
    print(f"  {resume}")
    print(f"  {'-' * (LARGEUR - 4)}")
    print(f"  Niveau réel (décision du souscripteur) : {reel}")
    print(f"  Prédiction avec questionnaire de santé : {pred_complete}"
          f"   {'exact' if pred_complete == reel else f'écart de {abs(pred_complete - reel)}'}")
    print(f"  Prédiction sous régime Lemoine         : {pred_lemoine}"
          f"   {'exact' if pred_lemoine == reel else f'écart de {abs(pred_lemoine - reel)}'}")

    ecart = abs(pred_lemoine - reel) - abs(pred_complete - reel)
    if ecart > 0:
        print(f"  → La suppression du questionnaire éloigne la prédiction "
              f"de {ecart} niveau(x).")
    elif ecart < 0:
        print(f"  → La prédiction se rapproche de {abs(ecart)} niveau(x). "
              f"Cas minoritaire, sans portée générale.")
    else:
        print("  → Aucun changement sur ce dossier.")


def main():
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--id", type=int, help="identifiant d'un dossier précis")
    parseur.add_argument("--n", type=int, default=0,
                         help="nombre de dossiers tirés au hasard")
    args = parseur.parse_args()

    titre("SÉLECTION MÉDICALE AUTOMATISÉE — DÉMONSTRATION")
    print("Le modèle attribue un niveau de risque entre 1 et 8, comme le ferait")
    print("un souscripteur. Deux modèles sont entraînés : l'un avec le")
    print("questionnaire de santé, l'autre sans.\n")

    df = data.charger()
    train, test = data.decouper(df)

    print("Entraînement des deux modèles :")
    modele_complet, X_complet = entrainer(train, test, "complete")
    modele_lemoine, X_lemoine = entrainer(train, test, "lemoine_stricte")

    pred_complete = modele_complet.predict(X_complet)
    pred_lemoine = modele_lemoine.predict(X_lemoine)
    reels = test[config.CIBLE].to_numpy()

    # --- Choix des dossiers à présenter -------------------------------------
    if args.id is not None:
        positions = test.index[test["Id"] == args.id].tolist()
        if not positions:
            print(f"\nDossier n° {args.id} introuvable dans le jeu de test.")
            print("Les identifiants disponibles vont de "
                  f"{test['Id'].min()} à {test['Id'].max()}.")
            return
        selection = positions
        titre("DOSSIER DEMANDÉ")
    elif args.n > 0:
        rng = np.random.default_rng(config.SEED)
        selection = rng.choice(len(test), size=min(args.n, len(test)),
                               replace=False).tolist()
        titre(f"{len(selection)} DOSSIERS TIRÉS AU HASARD")
    else:
        # Trois cas contrastés, choisis pour illustrer le propos.
        selection = []
        # 1. Un bon risque que le retrait du questionnaire fait chuter.
        candidats = np.where((reels == 8) & (pred_complete == 8)
                             & (pred_lemoine <= 6))[0]
        if len(candidats):
            selection.append(int(candidats[0]))
        # 2. Un dossier à risque élevé, correctement situé dans les deux cas.
        candidats = np.where((reels <= 3) & (pred_complete <= 3))[0]
        if len(candidats):
            selection.append(int(candidats[0]))
        # 3. Un dossier intermédiaire.
        candidats = np.where((reels == 6) & (pred_complete == 6))[0]
        if len(candidats):
            selection.append(int(candidats[0]))
        titre("TROIS DOSSIERS CONTRASTÉS")

    for pos in selection:
        afficher(
            dossier_id=int(test.loc[pos, "Id"]),
            reel=int(reels[pos]),
            pred_complete=int(pred_complete[pos]),
            pred_lemoine=int(pred_lemoine[pos]),
            resume=profil(test.loc[pos]),
        )

    # --- Mise en perspective -------------------------------------------------
    titre("CE QUE CES CAS ILLUSTRENT")
    exact_c = (pred_complete == reels).mean() * 100
    exact_l = (pred_lemoine == reels).mean() * 100
    eam_c = np.abs(pred_complete - reels).mean()
    eam_l = np.abs(pred_lemoine - reels).mean()

    print(f"Sur l'ensemble des {len(test):,} dossiers de test :".replace(",", " "))
    print(f"  Prédictions exactes      : {exact_c:.1f} %  →  {exact_l:.1f} %")
    print(f"  Écart absolu moyen       : {eam_c:.3f}  →  {eam_l:.3f}")
    print("\nUn cas particulier n'a pas valeur de preuve : c'est l'expérience")
    print("d'ablation, sur l'ensemble du jeu de test, qui fonde les conclusions.")


if __name__ == "__main__":
    main()
