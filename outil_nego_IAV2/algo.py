"""
Jusqu'où aller dans une négociation : risque de conflit de chaque output recherché
=================================================================================

Ce que l'outil renvoie pour une négo à préparer (pas de résultat global) :
  - pour chaque output recherché (ex. « baisse de prix de 5 % »), dans l'ordre de priorité donné par
    l'utilisateur : la probabilité de consensus (obtenu sans contrepartie), de compromis (obtenu avec
    une contrepartie) et de conflit (pas obtenu) ;
  - la ligne « jusqu'ici » : juste avant le premier output dont le risque de conflit dépasse le seuil
    SEUIL_CONFLIT (reglages.py) ;
  - pour les compromis : les contreparties le plus souvent concédées pour des outputs semblables.
  Ces chiffres viennent des outputs semblables de l'historique, dans des négos au contexte comparable.

Ce que l'historique enregistre, une fois la négo terminée :
  - les outputs dans l'ordre, avec la catégorie choisie par l'utilisateur avant la négo ;
  - le nombre d'outputs obtenus dans l'ordre (resultat = part obtenue) : l'output suivant est celui
    qui a bloqué (conflit), les autres n'ont pas été abordés ;
  - la contrepartie concédée pour chaque output obtenu en échange de quelque chose (compromis).

Principe :
  1. Chaque négo = une liste d'outputs (phrase + catégorie consensus/compromis/conflit) + un résultat.
  2. Chaque phrase -> embedding (vecteur de sens).
  3. Chaque négo -> vecteur à 3 compartiments [CONSENSUS | COMPROMIS | CONFLIT].
  4. Arbre construit par découpages successifs selon la proximité (k-means en 2, récursif).
  5. Nouvelle négo SANS résultat -> pour chaque output, recherche des outputs passés semblables
     (phrase, catégorie saisie, niveau chiffré), dans des négos au contexte comparable -> part de
     conflit, de compromis et de consensus parmi eux -> ligne « jusqu'ici ». L'arbre sert à
     retrouver les négos passées les plus proches, affichées pour comparaison.
  6. Contexte (facultatif) : le rapport de force selon les 5 forces de Porter (notes de 1 à 5,
     5 = favorable au négociateur), la raison de la négo, le secteur de l'entreprise, la catégorie
     d'achat et sa sous-catégorie.
     Dans l'interface, les forces sont obtenues par des questions qui s'enchaînent selon les réponses
     (QUESTIONS_FORCES dans reglages.py) ; chaque chemin de réponses donne une note de 1 à 5.
     - On apprend sur l'historique l'effet moyen de chaque élément sur le risque de conflit.
     - Les outputs semblables sont trouvés surtout sur leur texte ; leur issue est ensuite CORRIGÉE de
       l'écart de contexte avec la négo à préparer. Chaque estimation est ainsi expliquée force par force.
     - En option (ALPHA < 1), le contexte compte aussi dans le choix des voisins et dans l'arbre.

Organisation des fichiers :
  - historique.csv     : LA source de vérité (une ligne par argument). Toute négo ajoutée y est écrite.
  - modele_nego.pkl    : l'arbre et les vecteurs déjà calculés (un "cache" qu'on peut toujours
                         reconstruire à partir du CSV).

Format CSV (séparateur ; ou , détecté automatiquement, une ligne par argument) :
    nego_id ; rang ; argument ; categorie ; resultat
    - categorie : consensus / compromis / conflit
    - resultat  : même valeur sur toutes les lignes d'une négo (part des arguments obtenus,
                  0 à 1 ou 0 à 100). Vide pour une négo à estimer.
    Colonnes facultatives (même valeur sur toutes les lignes d'une négo) :
    force_fournisseur ; force_acheteur ; force_substituts ; force_entrants ; force_rivalite (1 à 5)
    raison (fin de contrat, renouvellement, ...) ; secteur ; categorie_achat ; sous_categorie
    (une ancienne colonne "domaine" est lue comme categorie_achat) ; trimestre (T1 à T4 ; sinon déduit de la
    colonne date) ; fournisseur (ou interlocuteur) ; exclue (oui = négo exceptionnelle, gardée dans
    l'historique mais pas utilisée pour estimer) ; toute autre colonne (date...) est conservée.
    Colonne facultative PAR LIGNE : contrepartie (vide si l'output a été obtenu sans contrepartie ;
    ignorée pour les outputs non obtenus).

Commandes :
    python algo.py init historique.csv      construit l'arbre à partir de l'historique
    python algo.py estimer future.csv       issue probable de chaque output + ligne « jusqu'ici »
    python algo.py ajouter realisees.csv    ajoute des négos terminées (avec résultat)
    python algo.py reconstruire             recalcule tout l'arbre depuis le CSV
    python algo.py evaluer                  teste la précision du risque de conflit annoncé
    python algo.py arbre                    affiche l'arbre
    python algo.py demo                     crée des fichiers d'exemple FICTIFS

Utilisation depuis Python :
    from algo import charger_modele, ajouter_negociation, estimer_outputs
"""
import os
import re
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# 0. Réglages : tous les paramètres modifiables sont dans reglages.py
# ---------------------------------------------------------------------------
from reglages import (  # noqa: E402
    COEF_CATEGORIE, POIDS_COMPARTIMENTS, TAILLE_MIN_FEUILLE, TAILLE_MAX_FEUILLE, PART_AVANT_RECONSTRUCTION,
    K_VOISINS, SEUIL_SIMILARITE, MIN_PROCHES, ALPHA, POIDS_FORCES, POIDS_SITUATION,
    SIMILARITE_AUTRE_SOUS_CATEGORIE, REGULARISATION, MIN_NEGOS_CONTEXTE, CONTROLE_NB_ARGUMENTS,
    CONTROLE_SEUIL_ALERTE, MODELE_EMBEDDING, FORCES, RAISONS, SECTEURS, CATEGORIES_ACHAT,
    SEUIL_CONFLIT, K_ARGUMENTS, POIDS_A_PRIORI, POIDS_NIVEAU, CONTREPARTIES, QUESTIONS_FORCES,
    COEF_CATEGORIE_OUTPUTS, POIDS_MEME_FOURNISSEUR,
)

CATEGORIES = ["consensus", "compromis", "conflit"]          # structure de la méthode : ne pas modifier
COLONNES_FORCES = [f[0] for f in FORCES]
# Éléments de contexte à valeur dans une liste : (nom affiché, clé dans la négo, valeurs possibles)
TRIMESTRES = ["T1", "T2", "T3", "T4"]
GROUPES_CONTEXTE = [("Raison", "raison", RAISONS), ("Secteur", "secteur", SECTEURS),
                    ("Catégorie d'achat", "categorie_achat", list(CATEGORIES_ACHAT)),
                    ("Trimestre", "trimestre", TRIMESTRES)]
CLES_CONTEXTE = ["forces", "raison", "secteur", "categorie_achat", "sous_categorie", "trimestre"]
VERSION_MODELE = 5       # change quand le format du modèle sauvegardé change


def verifier_reglages():
    """Arrête le programme avec un message clair si une valeur de reglages.py est incohérente."""
    erreurs = []
    def entre(nom, v, bas, haut):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not bas <= v <= haut:
            erreurs.append(f"{nom} = {v!r} : doit être entre {bas} et {haut}")
    for nom, tableau in (("COEF_CATEGORIE", COEF_CATEGORIE), ("COEF_CATEGORIE_OUTPUTS", COEF_CATEGORIE_OUTPUTS)):
        if not isinstance(tableau, dict) or set(tableau) != set(CATEGORIES) \
                or any(not isinstance(tableau[c], dict) or set(tableau[c]) != set(CATEGORIES) for c in tableau):
            erreurs.append(f"{nom} doit contenir exactement consensus, compromis, conflit (lignes et colonnes)")
            continue
        for a in CATEGORIES:
            for b in CATEGORIES:
                entre(f"{nom}[{a}][{b}]", tableau[a][b], 0, 1)
                if tableau[a][b] != tableau[b][a]:
                    erreurs.append(f"{nom} n'est pas symétrique ({a}/{b} ≠ {b}/{a})")
    if set(POIDS_COMPARTIMENTS) != set(CATEGORIES):
        erreurs.append("POIDS_COMPARTIMENTS doit contenir exactement consensus, compromis, conflit")
    for c, v in POIDS_COMPARTIMENTS.items():
        entre(f"POIDS_COMPARTIMENTS[{c}]", v, 0, 1e6)
    entre("TAILLE_MIN_FEUILLE", TAILLE_MIN_FEUILLE, 2, 1e9)
    if isinstance(TAILLE_MAX_FEUILLE, (int, float)) and TAILLE_MAX_FEUILLE < 2 * TAILLE_MIN_FEUILLE:
        erreurs.append(f"TAILLE_MAX_FEUILLE ({TAILLE_MAX_FEUILLE}) doit valoir au moins 2 × TAILLE_MIN_FEUILLE")
    entre("PART_AVANT_RECONSTRUCTION", PART_AVANT_RECONSTRUCTION, 0, 1)
    entre("K_VOISINS", K_VOISINS, 1, 1e9)
    entre("SEUIL_SIMILARITE", SEUIL_SIMILARITE, 0, 1)
    entre("MIN_PROCHES", MIN_PROCHES, 1, 1e9)
    entre("ALPHA", ALPHA, 0, 1)
    if set(POIDS_FORCES) != set(COLONNES_FORCES):
        erreurs.append(f"POIDS_FORCES doit avoir les mêmes clés que FORCES : {COLONNES_FORCES}")
    for c, v in POIDS_FORCES.items():
        entre(f"POIDS_FORCES[{c}]", v, 0, 1e6)
    if set(POIDS_SITUATION) != {"raison", "secteur", "categorie_achat", "trimestre"}:
        erreurs.append("POIDS_SITUATION doit contenir exactement raison, secteur, categorie_achat, trimestre")
    for c, v in POIDS_SITUATION.items():
        entre(f"POIDS_SITUATION[{c}]", v, 0, 1e6)
    entre("SIMILARITE_AUTRE_SOUS_CATEGORIE", SIMILARITE_AUTRE_SOUS_CATEGORIE, 0, 1)
    if not isinstance(REGULARISATION, (int, float)) or REGULARISATION <= 0:
        erreurs.append(f"REGULARISATION = {REGULARISATION!r} : doit être strictement positif")
    entre("MIN_NEGOS_CONTEXTE", MIN_NEGOS_CONTEXTE, 1, 1e9)
    entre("CONTROLE_NB_ARGUMENTS", CONTROLE_NB_ARGUMENTS, 1, 1e9)
    entre("CONTROLE_SEUIL_ALERTE", CONTROLE_SEUIL_ALERTE, 0, 1)
    entre("SEUIL_CONFLIT", SEUIL_CONFLIT, 0, 1)
    entre("K_ARGUMENTS", K_ARGUMENTS, 1, 1e9)
    entre("POIDS_A_PRIORI", POIDS_A_PRIORI, 0, 1e6)
    entre("POIDS_NIVEAU", POIDS_NIVEAU, 0, 1e6)
    entre("POIDS_MEME_FOURNISSEUR", POIDS_MEME_FOURNISSEUR, 1, 1e6)
    if not isinstance(CONTREPARTIES, (list, tuple)) or not CONTREPARTIES \
            or not all(isinstance(c, str) and c.strip() for c in CONTREPARTIES):
        erreurs.append("CONTREPARTIES doit être une liste de textes non vides")
    for nom, v in (("TAILLE_MIN_FEUILLE", TAILLE_MIN_FEUILLE), ("TAILLE_MAX_FEUILLE", TAILLE_MAX_FEUILLE),
                   ("K_VOISINS", K_VOISINS), ("MIN_PROCHES", MIN_PROCHES), ("MIN_NEGOS_CONTEXTE", MIN_NEGOS_CONTEXTE),
                   ("CONTROLE_NB_ARGUMENTS", CONTROLE_NB_ARGUMENTS), ("K_ARGUMENTS", K_ARGUMENTS)):
        if isinstance(v, float) and not v.is_integer():
            erreurs.append(f"{nom} = {v} : doit être un nombre entier")
    if not isinstance(QUESTIONS_FORCES, dict) or set(QUESTIONS_FORCES) != set(COLONNES_FORCES):
        erreurs.append(f"QUESTIONS_FORCES doit avoir les mêmes clés que FORCES : {COLONNES_FORCES}")
    else:
        for col, arbre in QUESTIONS_FORCES.items():
            erreurs += _verifier_question(arbre, f"QUESTIONS_FORCES[{col}]")
    if erreurs:
        raise ValueError("Réglage(s) incorrect(s) dans reglages.py :\n  - " + "\n  - ".join(erreurs))


def _verifier_question(noeud, ou):
    """Vérifie une question de QUESTIONS_FORCES et ses sous-questions ; renvoie la liste des erreurs."""
    if not isinstance(noeud, dict) or not isinstance(noeud.get("question"), str) \
            or not isinstance(noeud.get("reponses"), (list, tuple)) or not noeud["reponses"]:
        return [f"{ou} : il faut une \"question\" (texte) et une liste de \"reponses\""]
    erreurs, vues = [], set()
    for rep in noeud["reponses"]:
        if not isinstance(rep, (list, tuple)) or len(rep) != 2 or not isinstance(rep[0], str):
            erreurs.append(f"{ou} : chaque réponse doit être (texte, note ou question)")
            continue
        texte, suite = rep
        if texte in vues:
            erreurs.append(f"{ou} : réponse « {texte} » en double")
        vues.add(texte)
        if isinstance(suite, dict):
            erreurs += _verifier_question(suite, f"{ou} > {texte}")
        elif isinstance(suite, bool) or not isinstance(suite, int) or not 1 <= suite <= 5:
            erreurs.append(f"{ou} > {texte} : la note doit être un entier de 1 à 5")
    return erreurs


def suivre_questions(col, chemin):
    """Suit les réponses choisies pour une force. Renvoie (questions posées, suite) :
    questions posées = [(texte de la question, réponses possibles, réponse choisie ou None)] ;
    suite = la note finale (1 à 5) si le chemin est complet, sinon None.
    Une réponse qui n'existe plus (questions modifiées) arrête le chemin à cet endroit."""
    noeud, posees = QUESTIONS_FORCES[col], []
    for choix in list(chemin) + [None]:
        reponses = dict(noeud["reponses"])
        if choix not in reponses:
            posees.append((noeud["question"], [r for r, _ in noeud["reponses"]], None))
            return posees, None
        posees.append((noeud["question"], [r for r, _ in noeud["reponses"]], choix))
        suite = reponses[choix]
        if not isinstance(suite, dict):
            return posees, suite
        noeud = suite
    return posees, None


try:
    verifier_reglages()
except ValueError as _erreur:
    if __name__ == "__main__":          # en ligne de commande : message clair, sans détail technique
        sys.exit(f"Erreur : {_erreur}")
    raise
COEF_MATRICE = np.array([[COEF_CATEGORIE[a][b] for b in CATEGORIES] for a in CATEGORIES], dtype=float)
COEF_MATRICE_OUTPUTS = np.array([[COEF_CATEGORIE_OUTPUTS[a][b] for b in CATEGORIES] for a in CATEGORIES], dtype=float)


def empreinte_reglages():
    """Réglages qui changent l'arbre ou les effets appris : s'ils changent, il faut reconstruire."""
    return repr((POIDS_COMPARTIMENTS, TAILLE_MIN_FEUILLE, TAILLE_MAX_FEUILLE, ALPHA, REGULARISATION,
                 MIN_NEGOS_CONTEXTE, MODELE_EMBEDDING, FORCES, RAISONS, SECTEURS, CATEGORIES_ACHAT))


FICHIER_MODELE = "modele_nego.pkl"
COLONNES = ["nego_id", "rang", "argument", "categorie", "resultat"]


# ---------------------------------------------------------------------------
# 1. Lecture / écriture du CSV
# ---------------------------------------------------------------------------
def _detecter_format(chemin):
    for enc in ("utf-8-sig", "cp1252"):            # Excel français enregistre souvent en cp1252
        try:
            with open(chemin, encoding=enc) as f:
                entete = f.readline()
            return enc, (";" if entete.count(";") >= entete.count(",") else ",")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Encodage de {chemin} non reconnu")


def _en_nombre(v):
    if pd.isna(v) or str(v).strip() == "":
        return None
    return float(str(v).replace("%", "").replace(",", ".").strip())


def lire_negos_csv(chemin):
    """Lit un CSV (une ligne par argument) et renvoie une liste de négos."""
    enc, sep = _detecter_format(chemin)
    df = pd.read_csv(chemin, sep=sep, encoding=enc, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    manquantes = [c for c in COLONNES if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans {chemin} : {manquantes}")

    df["categorie"] = df["categorie"].str.strip().str.lower()
    inconnues = set(df["categorie"]) - set(CATEGORIES)
    if inconnues:
        raise ValueError(f"Catégories inconnues dans {chemin} : {inconnues}")
    df["rang"] = df["rang"].astype(int)
    df["resultat"] = df["resultat"].map(_en_nombre)

    contexte = COLONNES_FORCES + ["raison", "secteur", "categorie_achat", "sous_categorie", "trimestre",
                                  "fournisseur", "exclue"]
    if "categorie_achat" not in df.columns and "domaine" in df.columns:
        contexte.append("domaine")      # anciens fichiers : la colonne "domaine" sert de catégorie d'achat
    extras = [c for c in df.columns if c not in COLONNES + contexte + ["contrepartie"]]   # ex. date : conservées
    negos = []
    # ordre du fichier conservé (l'historique est chronologique : l'évaluation teste les négos les plus récentes)
    ordre_ids = list(dict.fromkeys(df["nego_id"]))
    groupes = {nid: g for nid, g in df.sort_values(["nego_id", "rang"]).groupby("nego_id", sort=False)}
    for nid in ordre_ids:
        g = groupes[nid]
        r = g["resultat"].dropna()
        res = float(r.iloc[0]) if len(r) else None
        if res is not None and res > 1:            # saisi en % (ex. 60) -> 0.60
            res /= 100
        infos = {c: g[c].iloc[0] for c in extras if pd.notna(g[c].iloc[0])}
        trimestre = _premiere_valeur(g, "trimestre")
        trimestre = trimestre.upper() if trimestre and trimestre.upper() in TRIMESTRES else \
            trimestre_de(infos.get("date"))
        negos.append({"id": str(nid),
                      "arguments": list(zip(g["argument"].str.strip(), g["categorie"])),
                      "contreparties": _lire_contreparties(g),
                      "resultat": res,
                      "forces": _lire_forces(g, nid, chemin),
                      "raison": _premiere_valeur(g, "raison"),
                      "secteur": _premiere_valeur(g, "secteur"),
                      "categorie_achat": _premiere_valeur(g, "categorie_achat") or _premiere_valeur(g, "domaine"),
                      "sous_categorie": _premiere_valeur(g, "sous_categorie"),
                      "trimestre": trimestre,
                      "fournisseur": _premiere_valeur(g, "fournisseur"),
                      "exclue": (_premiere_valeur(g, "exclue") or "").lower() in ("oui", "o", "1", "x", "vrai", "true"),
                      "infos": infos})
    return negos


def trimestre_de(jour):
    """'2026-11-19' -> 'T4' ; None si la date est absente ou illisible."""
    try:
        mois = int(str(jour)[5:7])
    except (TypeError, ValueError):
        return None
    return f"T{(mois - 1) // 3 + 1}" if 1 <= mois <= 12 else None


def meme_fournisseur(a, b):
    """Même fournisseur / interlocuteur (sans tenir compte des majuscules ni des espaces autour)."""
    return bool(a) and bool(b) and a.strip().lower() == b.strip().lower()


def changer_exclusion(chemin, nego_id, exclue):
    """Marque (ou démarque) une négo comme exceptionnelle dans le CSV : colonne exclue = oui / vide."""
    enc, sep = _detecter_format(chemin)
    df = pd.read_csv(chemin, sep=sep, encoding=enc, dtype=str)
    col_id = next(c for c in df.columns if c.strip().lower() == "nego_id")
    if "exclue" not in df.columns:
        df["exclue"] = ""
    lignes = df[col_id].astype(str) == str(nego_id)
    if not lignes.any():
        raise ValueError(f"Négo {nego_id} absente de {chemin}")
    df.loc[lignes, "exclue"] = "oui" if exclue else ""
    df.to_csv(chemin, sep=sep, index=False, encoding=enc)


def _lire_contreparties(g):
    """Contrepartie de chaque output (None = obtenu sans contrepartie, ou colonne absente)."""
    if "contrepartie" not in g.columns:
        return [None] * len(g)
    return [v.strip() if isinstance(v, str) and v.strip() else None for v in g["contrepartie"]]


def _premiere_valeur(g, col):
    if col not in g.columns:
        return None
    v = g[col].dropna()
    v = v[v.str.strip() != ""]
    return v.iloc[0].strip() if len(v) else None


def _lire_forces(g, nid, chemin):
    """Renvoie {colonne: note 1-5} si les 5 forces sont renseignées, sinon None (négo sans contexte)."""
    notes = {}
    for col in COLONNES_FORCES:
        v = _premiere_valeur(g, col)
        if v is None:
            return None
        note = _en_nombre(v)
        if note is None or not 1 <= note <= 5:
            raise ValueError(f"{chemin} : négo {nid}, {col} = {v!r} (attendu : une note de 1 à 5)")
        notes[col] = int(note) if float(note).is_integer() else note
    return notes


def ecrire_negos_csv(negos, chemin, sep=";", encoding="utf-8-sig"):
    """Ajoute des négos à la fin d'un CSV (le crée avec l'en-tête s'il n'existe pas)."""
    lignes = []
    for n in negos:
        contexte = dict(n.get("forces") or {})
        for cle in ("raison", "secteur", "categorie_achat", "sous_categorie", "trimestre", "fournisseur"):
            if n.get(cle):
                contexte[cle] = n[cle]
        if n.get("exclue"):
            contexte["exclue"] = "oui"
        contreparties = n.get("contreparties") or [None] * len(n["arguments"])
        for i, (a, c) in enumerate(n["arguments"]):
            lignes.append({**n.get("infos", {}), "nego_id": n["id"], **contexte, "rang": i + 1,
                           "argument": a, "categorie": c,
                           "resultat": "" if n["resultat"] is None else n["resultat"],
                           "contrepartie": contreparties[i] or ""})
    nouveau = pd.DataFrame(lignes)
    if os.path.exists(chemin):
        encoding, sep = _detecter_format(chemin)
        ancien = pd.read_csv(chemin, sep=sep, encoding=encoding, dtype=str)
        ancien.columns = [c.strip() for c in ancien.columns]
        # colonnes de l'ancien fichier d'abord (même ordre), puis les nouvelles éventuelles (ex. forces)
        colonnes = list(ancien.columns) + [c for c in nouveau.columns if c not in ancien.columns]
        tout = pd.concat([ancien, nouveau.astype(str).replace("nan", "")], ignore_index=True)
    else:
        ordre = (["nego_id"] + [c for c in nouveau.columns if c not in COLONNES + ["contrepartie"]]
                 + COLONNES[1:] + ["contrepartie"])
        colonnes = [c for c in ordre if c in nouveau.columns]
        tout = nouveau
    tout.reindex(columns=colonnes).to_csv(chemin, sep=sep, index=False, encoding=encoding)


# ---------------------------------------------------------------------------
# 2. Embeddings : phrase -> vecteur
# ---------------------------------------------------------------------------
class Embedder:
    NOM_MODELE = MODELE_EMBEDDING

    def __init__(self, phrases):
        self.cache = {}
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.NOM_MODELE)   # téléchargé une fois, puis en local
            self.mode = "sentence-transformers"
        except Exception:
            print("ℹ Modèle d'embedding indisponible (non installé ou pas de connexion) : repli TF-IDF.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(phrases)
            self.mode = "TF-IDF (repli : installer sentence-transformers pour de vrais embeddings)"

    def __getstate__(self):                        # on ne sauvegarde pas le gros modèle sur disque
        etat = self.__dict__.copy()
        etat["model"] = None
        return etat

    def encode(self, phrases):
        manquantes = [p for p in set(phrases) if p not in self.cache]
        if manquantes:
            if self.mode == "sentence-transformers":
                if self.model is None:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(self.NOM_MODELE)
                V = self.model.encode(manquantes)
            else:
                V = self.tfidf.transform(manquantes).toarray()
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
            self.cache.update(zip(manquantes, V))
        return np.array([self.cache[p] for p in phrases])


# ---------------------------------------------------------------------------
# 3. Vecteur d'une négo : 3 compartiments
# ---------------------------------------------------------------------------
def vectoriser_nego(nego, emb):
    E = emb.encode([a for a, _ in nego["arguments"]])
    cats = [c for _, c in nego["arguments"]]
    blocs = []
    for cat in CATEGORIES:
        idx = [i for i, c in enumerate(cats) if c == cat]
        bloc = E[idx].mean(0) if idx else np.zeros(E.shape[1])
        n = np.linalg.norm(bloc)
        blocs.append((bloc / n if n > 0 else bloc) * np.sqrt(POIDS_COMPARTIMENTS[cat]))
    v = np.concatenate(blocs)
    v = v / (np.linalg.norm(v) + 1e-12)
    # bloc contexte : notes ramenées entre 0 et 1 (0,5 = neutre si non renseigné)
    f = vecteur_forces(nego)
    f = np.full(len(FORCES), 0.5) if f is None else f
    return np.concatenate([np.sqrt(ALPHA) * v, np.sqrt(1 - ALPHA) * f / np.sqrt(len(FORCES))])


def vecteur_forces(nego):
    """Notes des forces ramenées entre 0 (défavorable) et 1 (favorable), ou None."""
    if not nego.get("forces"):
        return None
    return np.array([(nego["forces"][c] - 1) / 4 for c in COLONNES_FORCES])


def a_contexte(nego):
    return any(nego.get(cle) for cle in CLES_CONTEXTE)


def vecteur_contexte(nego):
    """Forces (0 à 1 ; 0,5 si non renseignées) + une case par valeur possible de chaque élément de
    GROUPES_CONTEXTE (raison, secteur, catégorie d'achat). None si aucun contexte."""
    if not a_contexte(nego):
        return None
    f = vecteur_forces(nego)
    parties = [np.full(len(FORCES), 0.5) if f is None else f]
    for _, cle, valeurs in GROUPES_CONTEXTE:
        parties.append([float(nego.get(cle) == v) for v in valeurs])
    return np.concatenate(parties)


def blocs_contexte():
    """[(nom, clé, tranche dans le vecteur de contexte)] : les forces une par une, puis chaque groupe."""
    blocs, k = [], 0
    for col, nom, *_ in FORCES:
        blocs.append((nom, col, slice(k, k + 1)))
        k += 1
    for nom, cle, valeurs in GROUPES_CONTEXTE:
        blocs.append((nom, cle, slice(k, k + len(valeurs))))
        k += len(valeurs)
    return blocs


def similarite_contexte(nego_a, nego_b):
    """Ressemblance de situation entre 0 et 1, sur les éléments renseignés dans les DEUX négos.
    Catégorie d'achat : même sous-catégorie (ou sous-catégorie inconnue) = 1, autre sous-catégorie = 0,5."""
    parts = []                                   # (valeur, poids) ; poids réglables dans reglages.py
    fa, fb = vecteur_forces(nego_a), vecteur_forces(nego_b)
    poids_f = np.array([POIDS_FORCES[c] for c in COLONNES_FORCES], dtype=float)
    if fa is not None and fb is not None and poids_f.sum() > 0:
        parts.append((1 - float((np.abs(fa - fb) * poids_f).sum() / poids_f.sum()), float(poids_f.sum())))
    for cle in ("raison", "secteur", "trimestre"):
        if nego_a.get(cle) and nego_b.get(cle):
            parts.append((float(nego_a[cle] == nego_b[cle]), POIDS_SITUATION[cle]))
    if nego_a.get("categorie_achat") and nego_b.get("categorie_achat"):
        if nego_a["categorie_achat"] != nego_b["categorie_achat"]:
            v = 0.0
        else:
            sa, sb = nego_a.get("sous_categorie"), nego_b.get("sous_categorie")
            v = SIMILARITE_AUTRE_SOUS_CATEGORIE if (sa and sb and sa != sb) else 1.0
        parts.append((v, POIDS_SITUATION["categorie_achat"]))
    total = sum(p for _, p in parts)
    if total <= 0:
        return None
    return sum(v * p for v, p in parts) / total


# ---------------------------------------------------------------------------
# 4. Comparaison fine argument par argument (phrases x catégorie)
# ---------------------------------------------------------------------------
def similarite_negos(nego_a, nego_b, emb):
    Ea = emb.encode([a for a, _ in nego_a["arguments"]])
    Eb = emb.encode([a for a, _ in nego_b["arguments"]])
    ca = [CATEGORIES.index(c) for _, c in nego_a["arguments"]]
    cb = [CATEGORIES.index(c) for _, c in nego_b["arguments"]]
    S = np.clip(Ea @ Eb.T, 0, 1) * COEF_MATRICE[np.ix_(ca, cb)]
    s_args = float((S.max(1).mean() + S.max(0).mean()) / 2)
    s_ctx = similarite_contexte(nego_a, nego_b)
    return s_args if s_ctx is None else ALPHA * s_args + (1 - ALPHA) * s_ctx


# ---------------------------------------------------------------------------
# 5. L'arbre
# ---------------------------------------------------------------------------
class Noeud:
    def __init__(self, indices, parent=None, profondeur=0):
        self.indices = list(indices)    # négos contenues dans ce nœud
        self.parent = parent
        self.profondeur = profondeur
        self.centres = None             # 2 centres si le nœud est coupé
        self.enfants = []

    def est_feuille(self):
        return not self.enfants


def construire_arbre(X, indices, parent=None, profondeur=0):
    indices = np.asarray(indices)
    noeud = Noeud(indices, parent, profondeur)
    if len(indices) < 2 * TAILLE_MIN_FEUILLE:
        return noeud
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X[indices])
    gauche, droite = indices[km.labels_ == 0], indices[km.labels_ == 1]
    if min(len(gauche), len(droite)) < TAILLE_MIN_FEUILLE:
        return noeud
    noeud.centres = km.cluster_centers_.copy()
    noeud.enfants = [construire_arbre(X, gauche, noeud, profondeur + 1),
                     construire_arbre(X, droite, noeud, profondeur + 1)]
    return noeud


def chemin_descente(arbre, x):
    """Liste des (nœud, branche choisie) de la racine jusqu'à la feuille."""
    chemin, noeud = [], arbre
    while not noeud.est_feuille():
        j = int(np.argmin(np.linalg.norm(noeud.centres - x, axis=1)))
        chemin.append((noeud, j))
        noeud = noeud.enfants[j]
    return chemin, noeud


def afficher_arbre(noeud, y, prefixe=""):
    r = y[noeud.indices]
    print(f"{prefixe}{'🍃' if noeud.est_feuille() else '●'} {len(r)} négos, "
          f"résultat moyen {r.mean():.0%} (écart-type {r.std():.0%})")
    for e in noeud.enfants:
        afficher_arbre(e, y, prefixe + "    ")


# ---------------------------------------------------------------------------
# 6. Le modèle
# ---------------------------------------------------------------------------
class ModeleNegociation:
    def __init__(self, historique, chemin_csv=None):
        self.chemin_csv = chemin_csv
        # négos exceptionnelles (exclue = oui) : gardées pour l'affichage, jamais utilisées pour estimer
        self.exclues = [n for n in historique if n["resultat"] is not None and n.get("exclue")]
        self.negos = [n for n in historique if n["resultat"] is not None and not n.get("exclue")]
        if len(self.negos) < 2 * TAILLE_MIN_FEUILLE:
            print(f"⚠ Seulement {len(self.negos)} négos avec résultat : l'arbre n'aura qu'une feuille.")
        self.emb = Embedder([a for n in self.negos for a, _ in n["arguments"]])
        self.X = np.stack([vectoriser_nego(n, self.emb) for n in self.negos])
        self.y = np.array([n["resultat"] for n in self.negos])
        self.arbre = construire_arbre(self.X, np.arange(len(self.negos)))
        self.taille_a_la_construction = len(self.negos)
        self.ajouts_depuis_construction = 0
        self.version = VERSION_MODELE
        self.reglages = empreinte_reglages()
        self._table = None
        self.ajuster_effets()

    def __getstate__(self):                        # la table des outputs se recalcule : pas sur disque
        etat = self.__dict__.copy()
        etat["_table"] = None
        return etat

    def ajuster_effets(self):
        """Apprend l'effet moyen de chaque élément de contexte sur le RISQUE DE CONFLIT d'un output
        (régression ridge sur tous les outputs abordés : 1 = conflit, 0 = obtenu).
        self.effets = None tant qu'il n'y a pas assez de négos avec contexte."""
        t = _table_outputs(self)
        nb_negos = int(t["a_ctx"].sum())
        lignes = t["a_ctx"][t["negos_idx"]] if len(t["negos_idx"]) else np.zeros(0, dtype=bool)
        self.effets = None
        if nb_negos < MIN_NEGOS_CONTEXTE or not lignes.any():
            return
        C, y = t["C"][t["negos_idx"][lignes]], t["conflit"][lignes]
        moy = C.mean(0)
        Cc, yc = C - moy, y - y.mean()
        coef = np.linalg.solve(Cc.T @ Cc + REGULARISATION * np.eye(C.shape[1]), Cc.T @ yc)
        self.effets = {"moyenne": moy, "coef": coef, "n": nb_negos}

    def contributions(self, nego):
        """Contribution de chaque case du contexte, par rapport à une négo de contexte moyen.
        Un élément non renseigné compte comme « moyen » : sa contribution est nulle."""
        c = vecteur_contexte(nego)
        if c is None or self.effets is None:
            return None
        ecarts = c - self.effets["moyenne"]
        for nom, cle, tranche in blocs_contexte():
            renseigne = nego.get("forces") if cle in COLONNES_FORCES else nego.get(cle)
            if not renseigne:
                ecarts[tranche] = 0.0
        return self.effets["coef"] * ecarts

    def effet_contexte(self, nego):
        """Écart de risque de conflit dû au contexte, par rapport à une négo de contexte moyen."""
        contrib = self.contributions(nego)
        return 0.0 if contrib is None else float(contrib.sum())

    def ids(self):
        return {n["id"] for n in self.negos} | {n["id"] for n in getattr(self, "exclues", [])}

    def toutes_negos(self):
        """Négos utilisées pour estimer + négos exclues (pour l'affichage de l'historique)."""
        return self.negos + getattr(self, "exclues", [])

    def fournisseurs(self):
        """Fournisseurs / interlocuteurs déjà rencontrés, par ordre alphabétique."""
        return sorted({n["fournisseur"].strip() for n in self.toutes_negos() if n.get("fournisseur")}, key=str.lower)

    def sauvegarder(self, chemin=FICHIER_MODELE):
        with open(chemin, "wb") as f:
            pickle.dump(self, f)


def charger_modele(chemin=FICHIER_MODELE, verifier_version=True):
    if not os.path.exists(chemin):
        sys.exit(f"Pas de modèle ({chemin}). Lance d'abord : python algo.py init historique.csv")
    with open(chemin, "rb") as f:
        m = pickle.load(f)
    if verifier_version and getattr(m, "version", 1) != VERSION_MODELE:
        raise ValueError("Modèle sauvegardé avec une ancienne version de l'outil : "
                         "lance 'reconstruire' (ou 'init') pour le recalculer.")
    if verifier_version and getattr(m, "reglages", None) != empreinte_reglages():
        raise ValueError("Des réglages qui demandent une reconstruction ont changé dans reglages.py : "
                         "lance 'reconstruire' pour recalculer l'arbre.")
    return m


# ---------------------------------------------------------------------------
# 7. AJOUT d'une négociation terminée (mise à jour de l'arbre, sans tout recalculer)
# ---------------------------------------------------------------------------
def ajouter_negociation(modele, nego, ecrire_csv=True):
    """
    Ajoute une négo terminée (avec résultat) :
      1. on calcule son vecteur ;
      2. elle descend l'arbre comme pour une estimation ;
      3. à chaque nœud traversé, le centre de la branche prise est déplacé un peu vers elle
         (moyenne mise à jour : centre += (x - centre) / (n + 1)) ;
      4. elle est rangée dans la feuille d'arrivée ;
      5. si la feuille devient trop grosse, on recoupe CETTE feuille seulement ;
      6. elle est écrite dans le CSV historique (source de vérité).
    """
    if nego["resultat"] is None:
        raise ValueError(f"La négo {nego['id']} n'a pas de résultat : utiliser estimer_outputs.")
    if nego["id"] in modele.ids():
        raise ValueError(f"La négo {nego['id']} existe déjà dans l'historique.")
    if nego.get("exclue"):                        # négo exceptionnelle : enregistrée, mais l'arbre ne change pas
        modele.exclues.append(nego)
        if ecrire_csv and modele.chemin_csv:
            ecrire_negos_csv([nego], modele.chemin_csv)
        return modele

    x = vectoriser_nego(nego, modele.emb)
    i = len(modele.negos)
    modele.negos.append(nego)
    modele.X = np.vstack([modele.X, x])
    modele.y = np.append(modele.y, nego["resultat"])

    chemin, feuille = chemin_descente(modele.arbre, x)
    modele.arbre.indices.append(i)                   # la racine contient toutes les négos
    for noeud, j in chemin:
        enfant = noeud.enfants[j]
        n_avant = len(enfant.indices)
        noeud.centres[j] += (x - noeud.centres[j]) / (n_avant + 1)
        enfant.indices.append(i)

    if len(feuille.indices) > TAILLE_MAX_FEUILLE:
        nouveau = construire_arbre(modele.X, feuille.indices, feuille.parent, feuille.profondeur)
        if feuille.parent is None:
            modele.arbre = nouveau
        else:
            p = feuille.parent
            p.enfants[p.enfants.index(feuille)] = nouveau

    modele.ajouts_depuis_construction += 1
    _table_ajouter(modele, i)                        # la table des outputs est complétée, pas recalculée
    modele.ajuster_effets()
    if ecrire_csv and modele.chemin_csv:
        ecrire_negos_csv([nego], modele.chemin_csv)
    return modele


def reconstruction_conseillee(modele):
    return modele.ajouts_depuis_construction > PART_AVANT_RECONSTRUCTION * modele.taille_a_la_construction


# ---------------------------------------------------------------------------
# 8. ESTIMATION d'une négociation à préparer : issue probable de chaque output + ligne « jusqu'ici »
# ---------------------------------------------------------------------------
def issues_outputs(nego):
    """Issue de chaque output d'une négo terminée, dans l'ordre :
    consensus (obtenu sans contrepartie), compromis (obtenu avec une contrepartie),
    conflit (l'output sur lequel la négo a bloqué) ou None (pas abordé : après le blocage)."""
    n = len(nego["arguments"])
    if nego.get("resultat") is None:
        return [None] * n
    k = int(round(nego["resultat"] * n))          # nombre d'outputs obtenus dans l'ordre
    contreparties = nego.get("contreparties") or [None] * n
    issues = []
    for i in range(n):
        if i < k:
            issues.append("compromis" if contreparties[i] else "consensus")
        elif i == k:
            issues.append("conflit")
        else:
            issues.append(None)
    return issues


_NOMBRE_NIVEAU = re.compile(r"\d+(?:[.,]\d+)?")


def niveau(texte):
    """Niveau chiffré d'un output : le dernier nombre du texte (« de 30 à 60 jours » -> 60), ou None."""
    nombres = _NOMBRE_NIVEAU.findall(texte)
    return float(nombres[-1].replace(",", ".")) if nombres else None


def _lignes_table(modele, indices):
    """Lignes de la table pour les négos d'indices donnés (voir _table_outputs)."""
    L = {k: [] for k in ("textes", "cats", "issues", "contreparties", "negos_idx", "textes_tous", "cats_tous")}
    N = {k: [] for k in ("F", "raison", "secteur", "trimestre", "categorie_achat", "sous_categorie",
                         "fournisseur", "C", "M", "a_ctx")}
    for i in indices:
        n = modele.negos[i]
        cps = n.get("contreparties") or [None] * len(n["arguments"])
        for (texte, cat), issue, cp in zip(n["arguments"], issues_outputs(n), cps):
            L["textes_tous"].append(texte)
            L["cats_tous"].append(CATEGORIES.index(cat))
            if issue is None:
                continue
            L["textes"].append(texte)
            L["cats"].append(CATEGORIES.index(cat))
            L["issues"].append(issue)
            L["contreparties"].append(cp if issue == "compromis" else None)
            L["negos_idx"].append(i)
        f = vecteur_forces(n)
        N["F"].append(np.full(len(FORCES), np.nan) if f is None else f)
        for cle in ("raison", "secteur", "trimestre", "categorie_achat", "sous_categorie"):
            N[cle].append(n.get(cle) or None)
        N["fournisseur"].append((n.get("fournisseur") or "").strip().lower() or None)
        c, masque = _contexte_et_masque(n)
        N["C"].append(c)
        N["M"].append(masque)
        N["a_ctx"].append(a_contexte(n))
    return L, N


def _contexte_et_masque(n):
    """Vecteur de contexte (zéros si aucun) et masque des éléments renseignés (pour l'effet du contexte)."""
    c = vecteur_contexte(n)
    taille = sum(t.stop - t.start for _, _, t in blocs_contexte())
    if c is None:
        return np.zeros(taille), np.zeros(taille)
    masque = np.zeros(taille)
    for _, cle, tranche in blocs_contexte():
        renseigne = n.get("forces") if cle in COLONNES_FORCES else n.get(cle)
        if renseigne:
            masque[tranche] = 1.0
    return c, masque


def _table_outputs(modele):
    """Tous les outputs ABORDÉS de l'historique (obtenus ou bloquants) et, par négo, son contexte.
    Gardée en cache dans le modèle ; complétée (pas recalculée) à chaque négo ajoutée."""
    if getattr(modele, "_table", None) is not None:
        return modele._table
    L, N = _lignes_table(modele, range(len(modele.negos)))
    modele._table = _assembler_table(modele, L, N)
    return modele._table


def _assembler_table(modele, L, N, ancienne=None):
    """Transforme les listes en tableaux (et les ajoute à la table existante si ancienne est donnée)."""
    def tab(liste, dtype=None, forme=None):
        if not liste:
            return np.zeros(forme or (0,), dtype=dtype)
        return np.array(liste, dtype=dtype)
    # embeddings : une ligne par phrase DISTINCTE (U), chaque output pointe vers sa phrase (tid)
    if ancienne is None:
        phrases, U = [], None
    else:
        phrases, U = ancienne["phrases"], ancienne["U"]
    index = {p: k for k, p in enumerate(phrases)}
    nouvelles = [p for p in dict.fromkeys(L["textes_tous"]) if p not in index]
    if nouvelles:
        V = modele.emb.encode(nouvelles)
        U = V if U is None or not len(U) else np.vstack([U, V])
        for p in nouvelles:
            index[p] = len(index)
        phrases = phrases + nouvelles
    if U is None:
        U = np.zeros((0, 1))
    t = {"textes": L["textes"], "cats": tab(L["cats"], int), "issues": tab(L["issues"], object),
         "contreparties": L["contreparties"], "negos_idx": tab(L["negos_idx"], int),
         "niveaux": tab([np.nan if niveau(x) is None else niveau(x) for x in L["textes"]], float),
         "tid": tab([index[x] for x in L["textes"]], int),
         "tid_tous": tab([index[x] for x in L["textes_tous"]], int),
         "cats_tous": tab(L["cats_tous"], int),
         "F": tab(N["F"], float, (0, len(FORCES))), "C": tab(N["C"], float), "M": tab(N["M"], float),
         "a_ctx": tab(N["a_ctx"], bool)}
    for cle in ("raison", "secteur", "trimestre", "categorie_achat", "sous_categorie", "fournisseur"):
        t[cle] = tab(N[cle], object)
    if ancienne is not None:
        for cle, v in t.items():
            if isinstance(v, list):
                t[cle] = ancienne[cle] + v
            elif len(v):
                t[cle] = np.concatenate([ancienne[cle], v]) if len(ancienne[cle]) else v
            else:
                t[cle] = ancienne[cle]
    t["phrases"], t["U"] = phrases, U
    # issues réelles et taux moyens de tout l'historique (point de départ quand il y a peu d'exemples ;
    # volontairement indépendants de la catégorie saisie, qui n'est qu'une prévision)
    issues = t["issues"]
    t["conflit"] = (issues == "conflit").astype(float)
    t["obtenu"] = issues != "conflit"
    t["compromis"] = (issues == "compromis").astype(float)
    t["taux_conflit"] = float(t["conflit"].mean()) if len(issues) else 0.5
    t["part_compromis"] = float(t["compromis"][t["obtenu"]].mean()) if t["obtenu"].any() else 0.5
    return t


def _table_ajouter(modele, i):
    """Ajoute la négo d'indice i à la table en cache (si elle existe déjà)."""
    if getattr(modele, "_table", None) is None:
        return
    L, N = _lignes_table(modele, [i])
    modele._table = _assembler_table(modele, L, N, ancienne=modele._table)


def _effets_negos(modele, t):
    """Effet du contexte de chaque négo passée sur le risque de conflit (même calcul qu'effet_contexte)."""
    if modele.effets is None or not len(t["C"]):
        return np.zeros(len(modele.negos))
    return ((t["C"] - modele.effets["moyenne"]) * t["M"]) @ modele.effets["coef"]


def _similarites_contexte(nego, t):
    """similarite_contexte(nego, n) pour toutes les négos passées d'un coup (NaN = rien à comparer)."""
    nb = len(t["C"])
    total, somme = np.zeros(nb), np.zeros(nb)
    f = vecteur_forces(nego)
    poids_f = np.array([POIDS_FORCES[c] for c in COLONNES_FORCES], dtype=float)
    if f is not None and poids_f.sum() > 0 and nb:
        ok = ~np.isnan(t["F"]).any(axis=1)
        v = 1 - (np.abs(t["F"][ok] - f) * poids_f).sum(axis=1) / poids_f.sum()
        somme[ok] += v * poids_f.sum()
        total[ok] += poids_f.sum()
    for cle in ("raison", "secteur", "trimestre"):
        if nego.get(cle):
            ok = t[cle] != None                      # noqa: E711 (comparaison élément par élément)
            somme[ok] += (t[cle][ok] == nego[cle]) * POIDS_SITUATION[cle]
            total[ok] += POIDS_SITUATION[cle]
    if nego.get("categorie_achat"):
        ok = t["categorie_achat"] != None            # noqa: E711
        meme = ok & (t["categorie_achat"] == nego["categorie_achat"])
        v = meme.astype(float)
        if nego.get("sous_categorie"):
            autre_sous = meme & (t["sous_categorie"] != None) & (t["sous_categorie"] != nego["sous_categorie"])  # noqa
            v[autre_sous] = SIMILARITE_AUTRE_SOUS_CATEGORIE
        somme[ok] += v[ok] * POIDS_SITUATION["categorie_achat"]
        total[ok] += POIDS_SITUATION["categorie_achat"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, somme / np.where(total > 0, total, 1), np.nan)


def _facteur_niveau(v, niveaux):
    """(petit / grand) ** POIDS_NIVEAU quand les deux outputs ont un niveau chiffré, sinon 1."""
    f = np.ones(len(niveaux))
    if v is None or v <= 0 or POIDS_NIVEAU == 0:
        return f
    ok = ~np.isnan(niveaux) & (niveaux > 0)
    f[ok] = (np.minimum(v, niveaux[ok]) / np.maximum(v, niveaux[ok])) ** POIDS_NIVEAU
    return f


def estimer_outputs(modele, nego, seuil=None, complet=True):
    """
    Pour chaque output de la négo (dans l'ordre de priorité de l'utilisateur) : probabilités de
    consensus / compromis / conflit, contreparties habituelles, et position par rapport à la ligne.
    1. on cherche les outputs passés les plus semblables (phrase, catégorie saisie, niveau chiffré),
       en privilégiant les négos au contexte comparable ;
    2. leur issue réelle est d'abord ramenée à un contexte moyen, puis on ajoute l'effet du contexte
       de la négo à préparer (effets appris sur l'historique, comme pour la v3) ;
    3. s'il y a peu d'exemples, l'estimation est tirée vers le taux moyen de conflit de l'historique.
    La catégorie saisie par l'utilisateur est une prévision : par défaut elle ne change pas le risque
    (COEF_CATEGORIE_OUTPUTS), elle sert à signaler un écart entre la prévision et l'historique.
    """
    seuil = SEUIL_CONFLIT if seuil is None else seuil
    t = _table_outputs(modele)
    # ressemblance de contexte entre la négo à préparer et chaque négo passée (0,5 si rien à comparer)
    if a_contexte(nego):
        s_ctx = np.nan_to_num(_similarites_contexte(nego, t), nan=0.5)
    else:
        s_ctx = np.ones(len(modele.negos))
    poids_ctx = ALPHA + (1 - ALPHA) * s_ctx[t["negos_idx"]] if len(t["textes"]) else np.zeros(0)
    # même fournisseur / interlocuteur : ses outputs comptent POIDS_MEME_FOURNISSEUR fois plus
    fourn = nego.get("fournisseur")
    meme = (t["fournisseur"] == fourn.strip().lower()) if (fourn and len(t["fournisseur"])) \
        else np.zeros(len(modele.negos), dtype=bool)
    meme = np.asarray(meme, dtype=bool)
    if len(t["textes"]) and meme.any():
        poids_ctx = poids_ctx * np.where(meme[t["negos_idx"]], POIDS_MEME_FOURNISSEUR, 1.0)
    effet = modele.effet_contexte(nego)
    neutres = t["conflit"] - _effets_negos(modele, t)[t["negos_idx"]] if len(t["textes"]) else np.zeros(0)

    outputs = []
    for rang, (texte, cat) in enumerate(nego["arguments"], 1):
        c = CATEGORIES.index(cat)
        if len(t["textes"]):
            s_phrases = np.clip(t["U"] @ modele.emb.encode([texte])[0], 0, 1)
            s_texte = s_phrases[t["tid"]] * _facteur_niveau(niveau(texte), t["niveaux"])
            s_arg = s_texte * COEF_MATRICE_OUTPUTS[c, t["cats"]]
        else:
            s_texte = s_arg = np.zeros(0)
        proches = np.flatnonzero(s_arg >= SEUIL_SIMILARITE)
        fiable = len(proches) >= MIN_PROCHES
        candidats = proches if fiable else np.flatnonzero(s_arg > 0)   # sinon : les moins éloignés
        w_tous = s_arg * poids_ctx
        sel = candidats[np.argsort(-w_tous[candidats])][:K_ARGUMENTS]
        w = w_tous[sel]
        # risque de conflit, ramené au contexte moyen puis corrigé du contexte de cette négo
        p0 = t["taux_conflit"]
        base = ((w * neutres[sel]).sum() + POIDS_A_PRIORI * p0) / (w.sum() + POIDS_A_PRIORI) \
            if (w.sum() + POIDS_A_PRIORI) > 0 else p0
        p_conflit = float(np.clip(base + effet, 0, 1))
        # parmi les outputs obtenus : part obtenue avec une contrepartie
        ob = t["obtenu"][sel] if len(sel) else np.zeros(0, dtype=bool)
        q0 = t["part_compromis"]
        w_ob = w[ob]
        part = ((w_ob * t["compromis"][sel][ob]).sum() + POIDS_A_PRIORI * q0) / (w_ob.sum() + POIDS_A_PRIORI) \
            if (w_ob.sum() + POIDS_A_PRIORI) > 0 else q0
        p_compromis = (1 - p_conflit) * float(part)
        p_consensus = 1 - p_conflit - p_compromis
        # contreparties habituelles des outputs semblables obtenus en compromis
        compte, total = {}, 0.0
        for j, wj in zip(sel, w):
            cp = t["contreparties"][j]
            if cp:
                compte[cp] = compte.get(cp, 0.0) + wj
                total += wj
        nb_cp = {cp: sum(1 for j in sel if t["contreparties"][j] == cp) for cp in compte}
        contreparties = [(cp, v / total, nb_cp[cp]) for cp, v in sorted(compte.items(), key=lambda kv: -kv[1])][:2]
        probas = {"consensus": p_consensus, "compromis": p_compromis, "conflit": p_conflit}
        outputs.append({"rang": rang, "texte": texte, "categorie": cat, **probas,
                        "issue_probable": max(probas, key=probas.get),
                        "contreparties": contreparties, "nb_semblables": int(len(sel)),
                        "nb_proches": int(len(proches)), "fiable": bool(fiable),
                        "avec_fournisseur": _deja_avec_fournisseur(modele, t, meme, s_texte)})

    ligne = placer_ligne(outputs, seuil)
    if not complet:                       # évaluation : seulement les probabilités et la ligne
        return {"id": nego["id"], "outputs": outputs, "ligne": ligne, "seuil": seuil}
    idx, s, feuille, noeud = _voisins(modele, nego)
    nb_comp = sum(1 for _, c in nego["arguments"] if c == "compromis")
    negos_fourn = [n for n in modele.toutes_negos() if meme_fournisseur(fourn, n.get("fournisseur"))]
    negos_fourn.sort(key=lambda n: n.get("infos", {}).get("date", ""), reverse=True)
    return {
        "id": nego["id"],
        "outputs": outputs,
        "ligne": ligne,
        "seuil": seuil,
        "precedents": [(modele.negos[i]["id"], float(si), float(modele.y[i])) for i, si in zip(idx, s)],
        "remonte_de_niveaux": feuille.profondeur - noeud.profondeur,
        "alertes": controle_coherence(modele, nego, t) + ecarts_prevision(outputs),
        "contexte_utilise": a_contexte(nego),
        "facteurs": facteurs_contexte(modele, nego),
        "effet_contexte": effet,
        "equilibre": {"compromis": nb_comp, "consensus + conflit": len(nego["arguments"]) - nb_comp},
        "fournisseur": fourn,
        "negos_fournisseur": [{"id": n["id"], "date": n.get("infos", {}).get("date", ""),
                               "obtenus": int(round(n["resultat"] * len(n["arguments"]))),
                               "nb": len(n["arguments"]), "exclue": bool(n.get("exclue"))} for n in negos_fourn],
    }


def _deja_avec_fournisseur(modele, t, meme, s_texte):
    """Ce qui s'est passé pour des outputs semblables (quelle que soit leur catégorie) dans les négos
    passées avec le même fournisseur : [(date, issue, contrepartie, texte)], les plus récentes d'abord."""
    if not len(t["textes"]) or not meme.any():
        return []
    idx = np.flatnonzero(meme[t["negos_idx"]] & (s_texte >= SEUIL_SIMILARITE))
    lignes = [(modele.negos[t["negos_idx"][j]].get("infos", {}).get("date", ""), str(t["issues"][j]),
               t["contreparties"][j], t["textes"][j]) for j in idx]
    return sorted(lignes, key=lambda l: l[0], reverse=True)[:3]


def texte_negos_fournisseur(r):
    """« Avec EMB-03 : 3 négos passées — 2026-03-12 : 2/5 obtenus ; ... » (les 5 plus récentes)."""
    liste = r.get("negos_fournisseur") or []
    if not liste:
        return f"Aucune négo passée avec {r['fournisseur']} dans l'historique."
    details = " ; ".join(f"{n['date'] or n['id']} : {n['obtenus']}/{n['nb']} obtenus" + (" (exclue)" if n["exclue"] else "")
                         for n in liste[:5])
    return f"Avec {r['fournisseur']} : {len(liste)} négo(s) passée(s) — {details}" + (" ; …" if len(liste) > 5 else "")


def texte_avec_fournisseur_court(o):
    """Version courte pour le tableau : « bloqué ×2, obtenu ×1 · dernier : bloqué (2026-08-25) »."""
    lignes = o.get("avec_fournisseur") or []
    if not lignes:
        return "—"
    nb_bloque = sum(1 for l in lignes if l[1] == "conflit")
    morceaux = ([f"bloqué ×{nb_bloque}"] if nb_bloque else []) + \
        ([f"obtenu ×{len(lignes) - nb_bloque}"] if len(lignes) > nb_bloque else [])
    jour, issue, cp, _ = lignes[0]
    dernier = "bloqué" if issue == "conflit" else ("obtenu contre " + cp if cp else "obtenu")
    return ", ".join(morceaux) + f" · dernier : {dernier} ({jour or 'date ?'})"


def texte_avec_fournisseur(o):
    """« bloqué (2026-03-12) ; obtenu contre engagement de volume (2025-11-02) » ou « »."""
    morceaux = []
    for jour, issue, cp, texte in o.get("avec_fournisseur", []):
        quoi = {"conflit": "bloqué", "consensus": "obtenu", "compromis": f"obtenu contre {cp}"}[issue]
        morceaux.append(f"{quoi} ({jour or 'date ?'}, « {texte} »)")
    return " ; ".join(morceaux)


def placer_ligne(outputs, seuil):
    """Place la ligne « jusqu'ici » juste avant le premier output (ordre de l'utilisateur) dont le risque
    de conflit dépasse le seuil ; renseigne la position de chaque output. Renvoie l'indice de la ligne
    (= nombre d'outputs à pousser)."""
    ligne = next((i for i, o in enumerate(outputs) if o["conflit"] > seuil), len(outputs))
    for i, o in enumerate(outputs):
        if i < ligne:
            o["position"] = "à pousser"
        elif i == ligne:
            o["position"] = "point bloquant probable"
        elif o["conflit"] > seuil:
            o["position"] = "risque élevé"
        else:
            o["position"] = "faible risque, après le blocage"
    return ligne


def _voisins(modele, nego):
    """Négos passées les plus proches (arbre + comparaison fine), comme dans la v3."""
    x = vectoriser_nego(nego, modele.emb)
    _, feuille = chemin_descente(modele.arbre, x)
    noeud = feuille
    while True:                                     # remonte si la feuille est trop pauvre
        idx_noeud = np.array(noeud.indices)
        sims = np.array([similarite_negos(nego, modele.negos[i], modele.emb) for i in idx_noeud])
        assez = len(idx_noeud) >= K_VOISINS and (sims >= SEUIL_SIMILARITE).sum() >= MIN_PROCHES
        if assez or noeud.parent is None:
            break
        noeud = noeud.parent
    ordre = np.argsort(-sims)[:K_VOISINS]
    return idx_noeud[ordre], sims[ordre], feuille, noeud


def facteurs_contexte(modele, nego):
    """Contribution de chaque élément de contexte renseigné au RISQUE DE CONFLIT (en points), par rapport
    à une négo de contexte moyen ; positif = plus de risque. La même pour tous les outputs de la négo.
    Appris sur l'historique : c'est une corrélation, pas une preuve de cause à effet."""
    contrib = modele.contributions(nego)
    if contrib is None:
        return []
    lignes = []
    for nom, cle, tranche in blocs_contexte():
        if cle in COLONNES_FORCES:
            if nego.get("forces"):
                lignes.append({"element": nom, "valeur": f"{nego['forces'][cle]}/5",
                               "points": float(contrib[tranche].sum())})
        elif nego.get(cle):
            lignes.append({"element": nom, "valeur": nego[cle], "points": float(contrib[tranche].sum())})
    return sorted(lignes, key=lambda d: abs(d["points"]), reverse=True)


def controle_coherence(modele, nego, t=None, k=CONTROLE_NB_ARGUMENTS, seuil_alerte=CONTROLE_SEUIL_ALERTE):
    """Compare le classement saisi à celui des outputs similaires de l'historique."""
    t = t or _table_outputs(modele)
    cats = t["cats_tous"]
    if not len(cats):
        return []
    alertes = []
    for texte, cat in nego["arguments"]:
        sims = (t["U"] @ modele.emb.encode([texte])[0])[t["tid_tous"]]
        proches = np.argsort(-sims)[:k]
        w = np.clip(sims[proches], 0, None)
        parts = np.array([w[cats[proches] == c].sum() for c in range(3)]) / (w.sum() + 1e-12)
        m = int(np.argmax(parts))
        if CATEGORIES[m] != cat and parts[m] >= seuil_alerte:
            alertes.append(f"Classement inhabituel : « {texte} » classé en {cat.upper()}, mais les outputs similaires ont été "
                           f"classés en {CATEGORIES[m].upper()} dans {parts[m]:.0%} des cas.")
    return alertes


def ecarts_prevision(outputs):
    """Écart entre la catégorie prévue par l'utilisateur et l'historique : prévu consensus / compromis
    mais plus de chances de bloquer que d'aboutir (risque > 50 %), ou prévu conflit mais peu de
    risque (< 25 %)."""
    alertes = []
    for o in outputs:
        if o["categorie"] != "conflit" and o["conflit"] > 0.5:
            alertes.append(f"« {o['texte']} » prévu en {o['categorie'].upper()}, mais d'après l'historique son "
                           f"risque de conflit est de {o['conflit']:.0%}.")
        elif o["categorie"] == "conflit" and o["conflit"] < 0.25:
            alertes.append(f"« {o['texte']} » prévu en CONFLIT, mais d'après l'historique son risque de conflit "
                           f"n'est que de {o['conflit']:.0%}.")
    return alertes


def texte_contreparties(o):
    """« engagement de volume (60 %, 6 cas), durée de contrat plus longue (25 %, 2 cas) » ou « — »."""
    if not o["contreparties"]:
        return "—"
    return ", ".join(f"{cp} ({part:.0%}, {nb} cas)" for cp, part, nb in o["contreparties"])


def afficher_estimation(r):
    print(f"\n=== Négo {r['id']} === (seuil de conflit : {r['seuil']:.0%})")
    for i, o in enumerate(r["outputs"]):
        if i == r["ligne"]:
            print("   " + "-" * 20 + " jusqu'ici " + "-" * 20)
        print(f"{o['rang']:>2}. [{o['categorie']:<9}] {o['texte']}")
        print(f"      consensus {o['consensus']:.0%} · compromis {o['compromis']:.0%} · conflit {o['conflit']:.0%}"
              f"  ->  {o['position']}" + ("" if o["fiable"] else "   ⚠ peu d'exemples semblables"))
        if o["contreparties"]:
            print(f"      contreparties habituelles : {texte_contreparties(o)}")
        if o.get("avec_fournisseur"):
            print(f"      déjà avec {r['fournisseur']} : {texte_avec_fournisseur(o)}")
    if r["ligne"] == len(r["outputs"]):
        print("   " + "-" * 20 + " jusqu'ici (tous les outputs) " + "-" * 20)
    e = r["equilibre"]
    print(f"Ta liste : {e['compromis']} compromis pour {e['consensus + conflit']} consensus + conflit.")
    if r.get("fournisseur"):
        print(texte_negos_fournisseur(r))
    if r["facteurs"]:
        print("Effet du contexte sur le risque de conflit (appris sur l'historique) :")
        for f in r["facteurs"]:
            print(f"   {f['points'] * 100:+5.1f} pts  {f['element']} ({f['valeur']})")
    print("Négos les plus proches (id | ressemblance | part obtenue) :")
    for pid, s, res in r["precedents"][:10]:
        print(f"   {pid:<10} {s:.2f}   {res:.0%}")
    for a in r["alertes"]:
        print("  ⚠", a)


# ---------------------------------------------------------------------------
# 9. Évaluation : le risque de conflit annoncé est-il juste ?
#    On simule l'usage réel : modèle construit sur le passé, puis on estime chaque négo récente
#    AVANT de l'ajouter (comme en vrai), et on compare aux issues réelles des outputs abordés.
# ---------------------------------------------------------------------------
def evaluer(historique, part_test=0.2, sans_contexte=False, alpha=None, seuil=None):
    """Renvoie l'erreur (score de Brier : moyenne de (risque annoncé - issue réelle)², plus bas = mieux)
    de l'outil et de la référence naïve (taux moyen de conflit de tous les outputs), et le taux de
    conflit réel parmi les outputs annoncés au-dessus / en dessous du seuil.
    sans_contexte=True : on ignore forces, raison et situation (pour mesurer ce que le contexte apporte).
    alpha : pour tester une autre valeur d'ALPHA sans modifier le réglage."""
    global ALPHA
    if alpha is not None:
        ancien, ALPHA = ALPHA, alpha
        try:
            return evaluer(historique, part_test, sans_contexte, seuil=seuil)
        finally:
            ALPHA = ancien
    seuil = SEUIL_CONFLIT if seuil is None else seuil
    if sans_contexte:                     # ni contexte, ni fournisseur : les outputs seuls
        historique = [{**n, **{cle: None for cle in CLES_CONTEXTE + ["fournisseur"]}} for n in historique]
    historique = [n for n in historique if n["resultat"] is not None and not n.get("exclue")]
    n_test = max(1, int(len(historique) * part_test))
    train, test = historique[:-n_test], historique[-n_test:]
    m = ModeleNegociation(train)
    taux_ref = _table_outputs(m)["taux_conflit"]                # référence figée sur le passé
    annonce, reel, ref = [], [], []
    for n in test:
        r = estimer_outputs(m, {**n, "resultat": None}, seuil, complet=False)
        for o, issue in zip(r["outputs"], issues_outputs(n)):
            if issue is None:
                continue
            annonce.append(o["conflit"])
            reel.append(float(issue == "conflit"))
            ref.append(taux_ref)
        ajouter_negociation(m, n, ecrire_csv=False)
    annonce, reel, ref = np.array(annonce), np.array(reel), np.array(ref)
    haut = annonce > seuil
    res = {"n_test": n_test, "n_outputs": len(reel),
           "brier_outil": float(((annonce - reel) ** 2).mean()), "brier_reference": float(((ref - reel) ** 2).mean()),
           "seuil": seuil, "n_au_dessus": int(haut.sum()), "n_en_dessous": int((~haut).sum()),
           "conflit_au_dessus": float(reel[haut].mean()) if haut.any() else None,
           "conflit_en_dessous": float(reel[~haut].mean()) if (~haut).any() else None}
    print(f"{res['n_outputs']} outputs abordés dans les {n_test} négos les plus récentes : "
          f"erreur outil {res['brier_outil']:.3f}  |  référence {res['brier_reference']:.3f}")
    return res


# ---------------------------------------------------------------------------
# 10. Données FICTIVES (commande "demo" uniquement)
# ---------------------------------------------------------------------------
FAMILLES = {
    "prix": (["baisse de prix", "réduction tarifaire", "diminution du prix unitaire", "remise sur le tarif"], "conflit", 0.25),
    "delai": (["allonger le délai de paiement", "paiement à 60 jours", "étendre les délais de règlement"], "compromis", 0.10),
    "volume": (["maintien du volume annuel", "garder les volumes actuels", "engagement de volume constant"], "consensus", 0.0),
    "exclu": (["exclusivité sur la catégorie", "contrat exclusif", "être fournisseur unique"], "conflit", 0.30),
    "penal": (["pénalités de retard", "sanctions en cas de retard de livraison", "clause de pénalité logistique"], "conflit", 0.20),
    "duree": (["contrat sur trois ans", "allonger la durée du contrat", "engagement pluriannuel"], "compromis", 0.05),
    "logis": (["livraison franco", "transport pris en charge par le fournisseur", "partage des coûts logistiques"], "compromis", 0.10),
    "revis": (["clause de révision des prix", "indexation sur les matières premières", "formule d'indexation du prix"], "compromis", 0.15),
    "qualite": (["audit qualité annuel", "renforcement des contrôles qualité", "certificat de conformité systématique"], "consensus", 0.05),
}
FACILITE = {"consensus": 0.95, "compromis": 0.75, "conflit": 0.45}


def donnees_fictives(n, debut=0, seed=1):
    rng = np.random.default_rng(seed)
    negos = []
    for k in range(debut, debut + n):
        familles = rng.choice(list(FAMILLES), size=rng.integers(4, 8), replace=False)
        args = []
        for f in familles:
            phrases, cat_hab, _ = FAMILLES[f]
            cat = cat_hab if rng.random() < 0.75 else str(rng.choice(CATEGORIES))
            args.append((str(rng.choice(phrases)), cat, f))
        args.sort(key=lambda a: CATEGORIES.index(a[1]))
        obtenus = 0
        for _, cat, f in args:
            if rng.random() < FACILITE[cat] - FAMILLES[f][2]:
                obtenus += 1
            else:
                break
        negos.append({"id": f"N{k:04d}", "arguments": [(a, c) for a, c, _ in args],
                      "resultat": round(obtenus / len(args), 3)})
    return negos


def creer_demo():
    for f in ("demo_historique.csv", "demo_realisees.csv", "demo_future.csv"):
        if os.path.exists(f):
            os.remove(f)
    ecrire_negos_csv(donnees_fictives(400, 0, seed=1), "demo_historique.csv")
    ecrire_negos_csv(donnees_fictives(30, 400, seed=2), "demo_realisees.csv")
    future = {"id": "FUTURE01", "resultat": None, "arguments": [
        ("garder les volumes de l'an dernier", "consensus"),
        ("allonger les délais de paiement", "compromis"),
        ("baisse de prix", "compromis"),
        ("exclusivité sur la catégorie", "conflit")]}
    ecrire_negos_csv([future], "demo_future.csv")
    print("Fichiers FICTIFS créés : demo_historique.csv (400 négos), "
          "demo_realisees.csv (30 négos terminées), demo_future.csv (1 négo à estimer)")


# ---------------------------------------------------------------------------
# 11. Ligne de commande
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Estimation du résultat de négociations")
    p.add_argument("commande", choices=["init", "estimer", "ajouter", "reconstruire", "evaluer", "arbre", "demo"])
    p.add_argument("fichier", nargs="?", help="CSV d'entrée")
    p.add_argument("--modele", default=FICHIER_MODELE, help="fichier du modèle sauvegardé")
    a = p.parse_args()

    if a.commande == "demo":
        creer_demo()

    elif a.commande == "init":
        if not a.fichier:
            sys.exit("Indique le CSV historique : init historique.csv")
        m = ModeleNegociation(lire_negos_csv(a.fichier), chemin_csv=os.path.abspath(a.fichier))
        m.sauvegarder(a.modele)
        print(f"Arbre construit sur {len(m.negos)} négos ({m.emb.mode}). Modèle sauvegardé : {a.modele}")

    elif a.commande == "estimer":
        m = charger_modele(a.modele)
        for n in lire_negos_csv(a.fichier):
            afficher_estimation(estimer_outputs(m, n))
        m.sauvegarder(a.modele)                       # garde en cache les nouveaux embeddings

    elif a.commande == "ajouter":
        m = charger_modele(a.modele)
        ajoutees = 0
        for n in lire_negos_csv(a.fichier):
            try:
                ajouter_negociation(m, n)
                ajoutees += 1
            except ValueError as e:
                print("  ignorée :", e)
        m.sauvegarder(a.modele)
        print(f"{ajoutees} négo(s) ajoutée(s). Historique : {len(m.negos)} négos.")
        if reconstruction_conseillee(m):
            print("ℹ Beaucoup d'ajouts depuis la dernière construction : "
                  "lance 'reconstruire' pour réorganiser l'arbre au propre.")

    elif a.commande == "reconstruire":
        ancien = charger_modele(a.modele, verifier_version=False)   # on ne lit que le chemin du CSV
        m = ModeleNegociation(lire_negos_csv(ancien.chemin_csv), chemin_csv=ancien.chemin_csv)
        m.sauvegarder(a.modele)
        print(f"Arbre reconstruit sur {len(m.negos)} négos.")

    elif a.commande == "evaluer":
        source = a.fichier or charger_modele(a.modele, verifier_version=False).chemin_csv
        historique = lire_negos_csv(source)
        print("Outputs seuls (sans contexte) :")
        evaluer(historique, sans_contexte=True)
        if any(a_contexte(n) for n in historique):
            print("Outputs + contexte :")
            evaluer(historique)

    elif a.commande == "arbre":
        m = charger_modele(a.modele)
        afficher_arbre(m.arbre, m.y)


if __name__ == "__main__":
    # On passe par le module importé (et non __main__) pour que le modèle sauvegardé
    # puisse être relu aussi bien en ligne de commande que depuis l'interface (app.py).
    import importlib
    try:
        importlib.import_module(os.path.splitext(os.path.basename(__file__))[0]).main()
    except (ValueError, FileNotFoundError) as erreur:
        sys.exit(f"Erreur : {erreur}")
