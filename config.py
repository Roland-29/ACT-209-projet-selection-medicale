"""Configuration centrale du projet.

Projet ACT-209 : automatisation de la sélection médicale à la souscription.
Impact de la suppression du questionnaire de santé sur la classification du risque.
"""
from pathlib import Path

# --- Chemins -----------------------------------------------------------------
RACINE = Path(__file__).resolve().parent
DATA = RACINE / "data"
OUTPUTS = RACINE / "outputs"
FIGURES = OUTPUTS / "figures"

CHEMIN_TRAIN = DATA / "train.csv"

for d in (DATA, OUTPUTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# --- Reproductibilité --------------------------------------------------------
SEED = 42
TEST_SIZE = 0.25

# --- Cible -------------------------------------------------------------------
CIBLE = "Response"
N_CLASSES = 8

# --- Blocs de variables ------------------------------------------------------
# Le cœur de noter étude d'ablation se fait ici.

BLOC_PRODUIT = [f"Product_Info_{i}" for i in range(1, 8)]

BLOC_DEMOGRAPHIQUE = ["Ins_Age"]

# En France, les assurés déclarent la taille, le poids et l'IMC dans le questionnaire de santé.
# Ils forment donc un bloc distinct, retirable séparément des antécédents.
BLOC_MORPHOLOGIQUE = ["Ht", "Wt", "BMI"]

BLOC_EMPLOI = [f"Employment_Info_{i}" for i in range(1, 7)]

BLOC_ASSURE = [f"InsuredInfo_{i}" for i in range(1, 8)]

# La numérotation de l'historique d'assurance n'est pas continue : Insurance_History_6 n'existe pas.
BLOC_HISTORIQUE_ASSURANCE = [
    f"Insurance_History_{i}" for i in [1, 2, 3, 4, 5, 7, 8, 9]
]

BLOC_ANTECEDENTS_FAMILIAUX = [f"Family_Hist_{i}" for i in range(1, 6)]

BLOC_ANTECEDENTS_MEDICAUX = [f"Medical_History_{i}" for i in range(1, 42)]

BLOC_MOTS_CLES_MEDICAUX = [f"Medical_Keyword_{i}" for i in range(1, 49)]

# --- Catégorielles -----------------------------------------------------------
# C'est la seule colonne réellement textuelle du fichier.
CATEGORIELLES_TEXTE = ["Product_Info_2"]

# --- Configurations d'ablation ----------------------------------------------
# Chaque configuration est définie par les blocs conservés.

BLOCS_NON_MEDICAUX = (
    BLOC_PRODUIT
    + BLOC_DEMOGRAPHIQUE
    + BLOC_EMPLOI
    + BLOC_ASSURE
    + BLOC_HISTORIQUE_ASSURANCE
)

CONFIGURATIONS = {
    "complete": {
        "libelle": "Complète (référence)",
        "description": (
            "Toutes les variables disponibles, y compris le déclaratif de santé. "
            "Représente l'information dont disposait l'assureur avant la loi Lemoine. "
            "Sert de référence haute : c'est l'écart à cette configuration qui mesure "
            "le coût informationnel de la réforme."
        ),
        "colonnes": (
            BLOCS_NON_MEDICAUX
            + BLOC_MORPHOLOGIQUE
            + BLOC_ANTECEDENTS_FAMILIAUX
            + BLOC_ANTECEDENTS_MEDICAUX
            + BLOC_MOTS_CLES_MEDICAUX
        ),
    },
    "sans_antecedents": {
        "libelle": "Sans antécédents (IMC conservé)",
        "description": (
            "Retrait des antécédents médicaux et familiaux, taille et poids conservés. "
            "Cette configuration ne correspond à aucune situation réglementaire réelle : "
            "en France, la morphologie est elle aussi déclarée par le questionnaire de santé. "
            "Elle est construite pour isoler, par différence avec la configuration stricte, "
            "la contribution propre du bloc morphologique à la classification du risque."
        ),
        "colonnes": BLOCS_NON_MEDICAUX + BLOC_MORPHOLOGIQUE,
    },
    "lemoine_stricte": {
        "libelle": "Lemoine stricte",
        "description": (
            "Retrait de l'intégralité du déclaratif de santé, taille et poids compris, "
            "ces derniers étant eux aussi recueillis par le questionnaire médical. "
            "Représente l'information dont dispose l'assureur sous le régime instauré par "
            "la loi Lemoine, qui supprime le questionnaire de santé pour les prêts inférieurs à "
            "200 000 € remboursés avant 60 ans."
        ),
        "colonnes": list(BLOCS_NON_MEDICAUX),
    },
}

# Ordre d'affichage dans les tableaux de résultats.
ORDRE_CONFIGURATIONS = ["complete", "sans_antecedents", "lemoine_stricte"]
