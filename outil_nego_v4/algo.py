"""
Prévision d'une négociation à partir de l'historique des négociations passées
=============================================================================

Principe :
  1. Historique rempli APRÈS chaque négo : liste des arguments avancés + pour chacun, ce qui s'est
     passé (consensus = accepté, compromis = obtenu avec concession, conflit = refusé).
     Résultat de la négo = moyenne des points de ses arguments (SCORE_CATEGORIE dans reglages.py).
  2. Chaque phrase -> embedding (vecteur de sens).
  3. Chaque négo -> vecteur = moyenne des embeddings de ses arguments (+ bloc des forces).
     Pas de catégorie dans ce vecteur : une négo future ne la connaît pas encore.
  4. Arbre construit par découpages successifs selon la proximité (k-means en 2, récursif).
  5. Nouvelle négo AVANT de négocier (arguments SANS catégorie) -> descente dans l'arbre -> feuille ->
     comparaison fine argument par argument -> négos passées les plus proches (voisines).
  6. Pour chaque argument prévu : comment l'argument de même sens a fini dans les négos voisines
     -> probabilité consensus / compromis / conflit. Résultat global = moyenne des points attendus.
  7. Contexte (facultatif) : le rapport de force selon les 5 forces de Porter (notes de 1 à 5,
     5 = favorable au négociateur), la raison de la négo, le secteur de l'entreprise, la catégorie
     d'achat et sa sous-catégorie.
     - On apprend sur l'historique l'effet moyen de chaque force et de chaque raison sur le résultat.
     - Les voisins sont trouvés sur les ARGUMENTS ; ce qu'ils ont obtenu est ensuite CORRIGÉ de l'écart
       de contexte avec la négo à estimer. Chaque estimation est ainsi expliquée force par force.
     - En option (ALPHA < 1), le contexte compte aussi dans le choix des voisins et dans l'arbre.

Organisation des fichiers :
  - historique.csv     : LA source de vérité (une ligne par argument). Toute négo ajoutée y est écrite.
  - modele_nego.pkl    : l'arbre et les vecteurs déjà calculés (un "cache" qu'on peut toujours
                         reconstruire à partir du CSV).

Format CSV (séparateur ; ou , détecté automatiquement, une ligne par argument) :
    nego_id ; rang ; argument ; categorie [; resultat]
    - categorie : ce qui s'est passé pour l'argument : consensus / compromis / conflit.
                  Vide sur TOUTES les lignes d'une négo à estimer (on ne le sait pas encore).
    - resultat  : facultatif. Il est toujours recalculé à partir des catégories (et réécrit dans le CSV
                  pour lecture) : la colonne n'est pas utilisée à la lecture.
    Colonnes facultatives (même valeur sur toutes les lignes d'une négo) :
    force_fournisseur ; force_acheteur ; force_substituts ; force_entrants ; force_rivalite (1 à 5)
    raison (fin de contrat, renouvellement, ...) ; secteur ; categorie_achat ; sous_categorie
    (une ancienne colonne "domaine" est lue comme categorie_achat) ; toute autre colonne (date...) est conservée.

Commandes :
    python algo.py init historique.csv      construit l'arbre à partir de l'historique
    python algo.py estimer future.csv       prévoit les négos du fichier (catégories vides)
    python algo.py ajouter realisees.csv    ajoute des négos terminées (catégories remplies)
    python algo.py reconstruire             recalcule tout l'arbre depuis le CSV
    python algo.py evaluer                  teste la précision de l'outil
    python algo.py arbre                    affiche l'arbre
    python algo.py demo                     crée des fichiers d'exemple FICTIFS

Utilisation depuis Python :
    from algo import charger_modele, ajouter_negociation, estimer_negociation
"""
import os
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
    SCORE_CATEGORIE, TAILLE_MIN_FEUILLE, TAILLE_MAX_FEUILLE, PART_AVANT_RECONSTRUCTION,
    K_VOISINS, SEUIL_SIMILARITE, MIN_PROCHES, ALPHA, POIDS_FORCES, POIDS_SITUATION,
    SIMILARITE_AUTRE_SOUS_CATEGORIE, REGULARISATION, MIN_NEGOS_CONTEXTE, SEUIL_ARGUMENT,
    MIN_PRECEDENTS_ARGUMENT, NB_ARGUMENTS_ELARGI, MODELE_EMBEDDING, FORCES, RAISONS, SECTEURS, CATEGORIES_ACHAT,
)

CATEGORIES = ["consensus", "compromis", "conflit"]          # structure de la méthode : ne pas modifier
COLONNES_FORCES = [f[0] for f in FORCES]
# Éléments de contexte à valeur dans une liste : (nom affiché, clé dans la négo, valeurs possibles)
GROUPES_CONTEXTE = [("Raison", "raison", RAISONS), ("Secteur", "secteur", SECTEURS),
                    ("Catégorie d'achat", "categorie_achat", list(CATEGORIES_ACHAT))]
CLES_CONTEXTE = ["forces", "raison", "secteur", "categorie_achat", "sous_categorie"]
VERSION_MODELE = 4       # change quand le format du modèle sauvegardé change


def verifier_reglages():
    """Arrête le programme avec un message clair si une valeur de reglages.py est incohérente."""
    erreurs = []
    def entre(nom, v, bas, haut):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not bas <= v <= haut:
            erreurs.append(f"{nom} = {v!r} : doit être entre {bas} et {haut}")
    if set(SCORE_CATEGORIE) != set(CATEGORIES):
        erreurs.append("SCORE_CATEGORIE doit contenir exactement consensus, compromis, conflit")
    for c, v in SCORE_CATEGORIE.items():
        entre(f"SCORE_CATEGORIE[{c}]", v, 0, 1)
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
    if set(POIDS_SITUATION) != {"raison", "secteur", "categorie_achat"}:
        erreurs.append("POIDS_SITUATION doit contenir exactement raison, secteur, categorie_achat")
    for c, v in POIDS_SITUATION.items():
        entre(f"POIDS_SITUATION[{c}]", v, 0, 1e6)
    entre("SIMILARITE_AUTRE_SOUS_CATEGORIE", SIMILARITE_AUTRE_SOUS_CATEGORIE, 0, 1)
    if not isinstance(REGULARISATION, (int, float)) or REGULARISATION <= 0:
        erreurs.append(f"REGULARISATION = {REGULARISATION!r} : doit être strictement positif")
    entre("MIN_NEGOS_CONTEXTE", MIN_NEGOS_CONTEXTE, 1, 1e9)
    entre("SEUIL_ARGUMENT", SEUIL_ARGUMENT, 0, 1)
    entre("MIN_PRECEDENTS_ARGUMENT", MIN_PRECEDENTS_ARGUMENT, 1, 1e9)
    entre("NB_ARGUMENTS_ELARGI", NB_ARGUMENTS_ELARGI, 1, 1e9)
    for nom, v in (("TAILLE_MIN_FEUILLE", TAILLE_MIN_FEUILLE), ("TAILLE_MAX_FEUILLE", TAILLE_MAX_FEUILLE),
                   ("K_VOISINS", K_VOISINS), ("MIN_PROCHES", MIN_PROCHES), ("MIN_NEGOS_CONTEXTE", MIN_NEGOS_CONTEXTE),
                   ("MIN_PRECEDENTS_ARGUMENT", MIN_PRECEDENTS_ARGUMENT), ("NB_ARGUMENTS_ELARGI", NB_ARGUMENTS_ELARGI)):
        if isinstance(v, float) and not v.is_integer():
            erreurs.append(f"{nom} = {v} : doit être un nombre entier")
    if erreurs:
        raise ValueError("Réglage(s) incorrect(s) dans reglages.py :\n  - " + "\n  - ".join(erreurs))


try:
    verifier_reglages()
except ValueError as _erreur:
    if __name__ == "__main__":          # en ligne de commande : message clair, sans détail technique
        sys.exit(f"Erreur : {_erreur}")
    raise
SCORES = np.array([SCORE_CATEGORIE[c] for c in CATEGORIES], dtype=float)


def empreinte_reglages():
    """Réglages qui changent l'arbre ou les effets appris : s'ils changent, il faut reconstruire."""
    return repr((SCORE_CATEGORIE, TAILLE_MIN_FEUILLE, TAILLE_MAX_FEUILLE, ALPHA, REGULARISATION,
                 MIN_NEGOS_CONTEXTE, MODELE_EMBEDDING, FORCES, RAISONS, SECTEURS, CATEGORIES_ACHAT))


def score_nego(arguments):
    """Résultat d'une négo terminée = moyenne des points de ses arguments ; None si catégories inconnues."""
    cats = [c for _, c in arguments]
    if not cats or any(c is None for c in cats):
        return None
    return float(np.mean([SCORE_CATEGORIE[c] for c in cats]))


FICHIER_MODELE = "modele_nego.pkl"
COLONNES = ["nego_id", "rang", "argument", "categorie", "resultat"]
COLONNES_OBLIGATOIRES = ["nego_id", "rang", "argument", "categorie"]   # resultat : recalculé


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
    manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans {chemin} : {manquantes}")

    df["categorie"] = df["categorie"].fillna("").str.strip().str.lower()
    inconnues = set(df["categorie"]) - set(CATEGORIES) - {""}
    if inconnues:
        raise ValueError(f"Catégories inconnues dans {chemin} : {inconnues}")
    df["rang"] = df["rang"].astype(int)

    contexte = COLONNES_FORCES + ["raison", "secteur", "categorie_achat", "sous_categorie"]
    if "categorie_achat" not in df.columns and "domaine" in df.columns:
        contexte.append("domaine")      # anciens fichiers : la colonne "domaine" sert de catégorie d'achat
    extras = [c for c in df.columns if c not in COLONNES + contexte]   # ex. date : conservées
    negos = []
    for nid, g in df.sort_values(["nego_id", "rang"]).groupby("nego_id", sort=False):
        remplies = (g["categorie"] != "").sum()
        if 0 < remplies < len(g):
            raise ValueError(f"{chemin} : négo {nid}, catégorie remplie pour certains arguments seulement "
                             "(tout remplir pour une négo terminée, tout laisser vide pour une négo à estimer).")
        arguments = [(a, c or None) for a, c in zip(g["argument"].str.strip(), g["categorie"])]
        negos.append({"id": str(nid),
                      "arguments": arguments,
                      "resultat": score_nego(arguments),
                      "forces": _lire_forces(g, nid, chemin),
                      "raison": _premiere_valeur(g, "raison"),
                      "secteur": _premiere_valeur(g, "secteur"),
                      "categorie_achat": _premiere_valeur(g, "categorie_achat") or _premiere_valeur(g, "domaine"),
                      "sous_categorie": _premiere_valeur(g, "sous_categorie"),
                      "infos": {c: g[c].iloc[0] for c in extras if pd.notna(g[c].iloc[0])}})
    return negos


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
        for cle in ("raison", "secteur", "categorie_achat", "sous_categorie"):
            if n.get(cle):
                contexte[cle] = n[cle]
        for i, (a, c) in enumerate(n["arguments"]):
            lignes.append({**n.get("infos", {}), "nego_id": n["id"], **contexte, "rang": i + 1,
                           "argument": a, "categorie": c or "",
                           "resultat": "" if n["resultat"] is None else round(n["resultat"], 4)})
    nouveau = pd.DataFrame(lignes)
    if os.path.exists(chemin):
        encoding, sep = _detecter_format(chemin)
        ancien = pd.read_csv(chemin, sep=sep, encoding=encoding, dtype=str)
        ancien.columns = [c.strip() for c in ancien.columns]
        # colonnes de l'ancien fichier d'abord (même ordre), puis les nouvelles éventuelles (ex. forces)
        colonnes = list(ancien.columns) + [c for c in nouveau.columns if c not in ancien.columns]
        tout = pd.concat([ancien, nouveau.astype(str).replace("nan", "")], ignore_index=True)
    else:
        ordre = ["nego_id"] + [c for c in nouveau.columns if c not in COLONNES] + COLONNES[1:]
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
# 3. Vecteur d'une négo : moyenne des arguments (sans catégorie, inconnue avant la négo)
# ---------------------------------------------------------------------------
def vectoriser_nego(nego, emb):
    E = emb.encode([a for a, _ in nego["arguments"]])
    v = E.mean(0)
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
    for cle in ("raison", "secteur"):
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
# 4. Comparaison fine argument par argument (ressemblance des phrases)
# ---------------------------------------------------------------------------
def similarite_negos(nego_a, nego_b, emb):
    Ea = emb.encode([a for a, _ in nego_a["arguments"]])
    Eb = emb.encode([a for a, _ in nego_b["arguments"]])
    S = np.clip(Ea @ Eb.T, 0, 1)
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
        self.negos = [n for n in historique if n["resultat"] is not None]
        if not self.negos:
            raise ValueError("Aucune négo terminée (avec catégories remplies) dans l'historique.")
        if len(self.negos) < 2 * TAILLE_MIN_FEUILLE:
            print(f"⚠ Seulement {len(self.negos)} négos terminées : l'arbre n'aura qu'une feuille.")
        self.emb = Embedder([a for n in self.negos for a, _ in n["arguments"]])
        self.X = np.stack([vectoriser_nego(n, self.emb) for n in self.negos])
        self.y = np.array([n["resultat"] for n in self.negos])
        self.frequences = self.calculer_frequences()
        self.arbre = construire_arbre(self.X, np.arange(len(self.negos)))
        self.taille_a_la_construction = len(self.negos)
        self.ajouts_depuis_construction = 0
        self.version = VERSION_MODELE
        self.reglages = empreinte_reglages()
        self.ajuster_effets()

    def ajuster_effets(self):
        """Apprend l'effet moyen de chaque élément de contexte sur le résultat (régression ridge).
        self.effets = None tant qu'il n'y a pas assez de négos avec contexte."""
        idx = [i for i, n in enumerate(self.negos) if a_contexte(n)]
        self.effets = None
        if len(idx) < MIN_NEGOS_CONTEXTE:
            return
        C = np.array([vecteur_contexte(self.negos[i]) for i in idx])
        moy = C.mean(0)
        Cc, yc = C - moy, self.y[idx] - self.y[idx].mean()
        coef = np.linalg.solve(Cc.T @ Cc + REGULARISATION * np.eye(C.shape[1]), Cc.T @ yc)
        self.effets = {"moyenne": moy, "coef": coef, "n": len(idx)}

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
        """Écart de résultat attendu dû au contexte, par rapport à une négo de contexte moyen."""
        contrib = self.contributions(nego)
        return 0.0 if contrib is None else float(contrib.sum())

    def calculer_frequences(self):
        """Part de chaque catégorie sur tous les arguments de l'historique (repli si argument inconnu)."""
        cats = [c for n in self.negos for _, c in n["arguments"]]
        return np.array([cats.count(c) for c in CATEGORIES], dtype=float) / len(cats)

    def ids(self):
        return {n["id"] for n in self.negos}

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
    nego = {**nego, "resultat": score_nego(nego["arguments"])}     # toujours recalculé
    if nego["resultat"] is None:
        raise ValueError(f"La négo {nego['id']} n'a pas toutes ses catégories (ce qui s'est passé) : "
                         "utiliser estimer_negociation pour une négo future.")
    if nego["id"] in modele.ids():
        raise ValueError(f"La négo {nego['id']} existe déjà dans l'historique.")

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
    modele.frequences = modele.calculer_frequences()
    modele.ajuster_effets()
    if ecrire_csv and modele.chemin_csv:
        ecrire_negos_csv([nego], modele.chemin_csv)
    return modele


def reconstruction_conseillee(modele):
    return modele.ajouts_depuis_construction > PART_AVANT_RECONSTRUCTION * modele.taille_a_la_construction


# ---------------------------------------------------------------------------
# 8. ESTIMATION d'une négociation future (sans résultat)
# ---------------------------------------------------------------------------
def estimer_negociation(modele, nego):
    """Prévision d'une négo future : ses arguments sont saisis SANS catégorie (on ne sait pas encore)."""
    nego = {**nego, "arguments": [(a, None) for a, _ in nego["arguments"]]}   # catégories ignorées
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
    idx, s = idx_noeud[ordre], sims[ordre]
    w = np.clip(s, 1e-6, None)

    # Argument par argument : comment l'argument de même sens a fini dans les négos voisines.
    previsions = prevoir_arguments(modele, nego, idx, w)
    # Correction du contexte : chaque précédent est d'abord ramené à un contexte moyen (on retire l'effet
    # du rapport de force de SA négo), puis on ajoute l'effet du contexte de la négo à estimer. Ainsi :
    # estimation = estimation pour un contexte moyen + somme des contributions affichées.
    # (chaque argument reste entre 0 et 100 % : la somme des contributions retombe exactement sur
    # l'estimation, sauf si un argument atteint 0 ou 100 %)
    effet_nego = modele.effet_contexte(nego)
    for p in previsions:                            # points attendus de chaque argument, avec TON contexte
        p["points_avec_contexte"] = float(np.clip(p["points_contexte_moyen"] + effet_nego, 0, 1))
    base = float(np.mean([np.clip(p["points_contexte_moyen"], 0, 1) for p in previsions]))
    moyenne = float(np.mean([p["points_avec_contexte"] for p in previsions]))
    # Marge : dispersion des résultats (ramenés à un contexte moyen) des négos voisines.
    neutres = np.array([modele.y[i] - modele.effet_contexte(modele.negos[i]) for i in idx])
    centre = float((neutres * w).sum() / w.sum())
    ecart = float(np.sqrt((w * (neutres - centre) ** 2).sum() / w.sum()))
    return {
        "id": nego["id"],
        "estimation": moyenne,
        "estimation_contexte_moyen": base,
        "marge": (max(0.0, moyenne - ecart), min(1.0, moyenne + ecart)),
        "fiable": bool(s.max() >= SEUIL_SIMILARITE),
        "remonte_de_niveaux": feuille.profondeur - noeud.profondeur,
        "precedents": [(modele.negos[i]["id"], float(si), float(modele.y[i])) for i, si in zip(idx, s)],
        "arguments": previsions,
        "alertes": [p["remarque"] for p in previsions if p["remarque"]],
        "contexte_utilise": a_contexte(nego),
        "facteurs": facteurs_contexte(modele, nego),
    }


def prevoir_arguments(modele, nego, idx, w):
    """
    Pour chaque argument de la négo future :
      1. dans chaque négo voisine j (poids w_j), on prend l'argument passé le plus ressemblant ;
         s'il ressemble assez (≥ SEUIL_ARGUMENT), il compte avec le poids w_j × ressemblance ;
      2. probabilité de chaque catégorie = part du poids total des précédents finis dans cette catégorie ;
      3. si moins de MIN_PRECEDENTS_ARGUMENT négos voisines contiennent l'argument : on élargit à tout
         l'historique (les NB_ARGUMENTS_ELARGI arguments les plus semblables, contexte ignoré) ;
      4. si aucun argument de l'historique ne ressemble assez : fréquences générales des catégories.
    "probas" = ce qui s'est passé dans les précédents, tel quel (sans correction du contexte).
    "points_contexte_moyen" = points attendus (SCORE_CATEGORIE) avec l'effet du contexte de chaque
    précédent retiré ; estimer_negociation y ajoute l'effet du contexte de la négo à estimer
    ("points_avec_contexte"). Résultat global = moyenne de ces points.
    """
    textes = [a for a, _ in nego["arguments"]]
    E = modele.emb.encode(textes)
    effets = {}

    def effet(i):
        if i not in effets:
            effets[i] = modele.effet_contexte(modele.negos[i])
        return effets[i]

    voisins = []                                    # (indice, poids, embeddings, catégories)
    for i, wi in zip(idx, w):
        args = modele.negos[i]["arguments"]
        voisins.append((i, wi, modele.emb.encode([a for a, _ in args]),
                        np.array([CATEGORIES.index(c) for _, c in args])))

    previsions = []
    for k, texte in enumerate(textes):
        poids = np.zeros(3)
        points, total, nb = 0.0, 0.0, 0
        for i, wi, Ev, cv in voisins:
            sims = Ev @ E[k]
            j = int(np.argmax(sims))
            if sims[j] >= SEUIL_ARGUMENT:
                p = wi * sims[j]
                poids[cv[j]] += p
                points += p * (SCORES[cv[j]] - effet(i))
                total += p
                nb += 1
        source, remarque = "voisins", None
        if nb < MIN_PRECEDENTS_ARGUMENT:
            elargi = _prevision_elargie(modele, E[k], effet)
            if elargi is not None:
                poids, points, total, nb = elargi
                source = "historique"
                remarque = (f"« {texte} » : peu présent dans les négos comparables, prévision élargie à tout "
                            f"l'historique ({nb} arguments semblables, contexte non pris en compte).")
            else:
                poids = modele.frequences.copy()
                points, total, nb = float(poids @ SCORES), 1.0, 0
                source = "aucun"
                remarque = (f"« {texte} » : aucun argument semblable dans l'historique, prévision = "
                            "fréquences générales des catégories (peu informative).")
        probas = poids / poids.sum()
        previsions.append({
            "argument": texte,
            "probas": {c: float(p) for c, p in zip(CATEGORIES, probas)},
            "categorie_probable": CATEGORIES[int(np.argmax(probas))],
            "points_attendus": float(probas @ SCORES),
            "points_contexte_moyen": float(points / total),
            "nb_precedents": nb,
            "source": source,
            "remarque": remarque,
        })
    return previsions


def _prevision_elargie(modele, e, effet):
    """Les arguments les plus semblables de tout l'historique (sans tenir compte du contexte)."""
    textes, cats, origine = [], [], []
    for i, n in enumerate(modele.negos):
        for a, c in n["arguments"]:
            textes.append(a)
            cats.append(CATEGORIES.index(c))
            origine.append(i)
    sims = modele.emb.encode(textes) @ e
    proches = [j for j in np.argsort(-sims)[:NB_ARGUMENTS_ELARGI] if sims[j] >= SEUIL_ARGUMENT]
    if not proches:
        return None
    poids = np.zeros(3)
    points = total = 0.0
    for j in proches:
        poids[cats[j]] += sims[j]
        points += sims[j] * (SCORES[cats[j]] - effet(origine[j]))
        total += sims[j]
    return poids, points, total, len(proches)


def facteurs_contexte(modele, nego):
    """Contribution de chaque élément de contexte renseigné, en points de résultat, par rapport à une
    négo de contexte moyen. Leur somme = estimation - estimation pour un contexte moyen.
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


def afficher_estimation(r):
    print(f"\n=== Négo {r['id']} ===")
    if not r["fiable"]:
        print("⚠ pas assez de précédents comparables")
    print("Prévision par argument (consensus / compromis / conflit) :")
    for p in r["arguments"]:
        pr = p["probas"]
        print(f"   {pr['consensus']:4.0%} / {pr['compromis']:4.0%} / {pr['conflit']:4.0%}  "
              f"→ {p['categorie_probable']:<9}  {p['argument']}  ({p['nb_precedents']} précédent(s))")
    if r["remonte_de_niveaux"]:
        print(f"(peu de négos très proches dans le groupe : recherche élargie de {r['remonte_de_niveaux']} niveau(x))")
    print("Négos les plus proches (id | ressemblance | résultat) :")
    for pid, s, res in r["precedents"]:
        print(f"   {pid:<10} {s:.2f}   {res:.0%}")
    for a in r["alertes"]:
        print("  ⚠", a)


# ---------------------------------------------------------------------------
# 9. Évaluation : l'outil fait-il mieux que la moyenne de toute la base ?
#    On simule l'usage réel : arbre construit sur le passé, puis on estime chaque négo récente
#    AVANT de l'ajouter (comme en vrai).
# ---------------------------------------------------------------------------
def evaluer(historique, part_test=0.2, sans_contexte=False, alpha=None):
    """sans_contexte=True : on ignore forces et raison (pour mesurer ce que le contexte apporte).
    alpha : pour tester une autre valeur d'ALPHA sans modifier le réglage."""
    global ALPHA
    if alpha is not None:
        ancien, ALPHA = ALPHA, alpha
        try:
            return evaluer(historique, part_test, sans_contexte)
        finally:
            ALPHA = ancien
    if sans_contexte:
        historique = [{**n, **{cle: None for cle in CLES_CONTEXTE}} for n in historique]
    historique = [n for n in historique if n["resultat"] is not None]
    n_test = max(1, int(len(historique) * part_test))
    train, test = historique[:-n_test], historique[-n_test:]
    m = ModeleNegociation(train)
    err_outil, err_base = [], []
    bons_outil, bons_base, nb_args = 0, 0, 0
    for n in test:
        r = estimer_negociation(m, n)               # catégories ignorées : prévision AVANT la négo
        err_outil.append(abs(r["estimation"] - n["resultat"]))
        err_base.append(abs(m.y.mean() - n["resultat"]))
        plus_frequente = CATEGORIES[int(np.argmax(m.frequences))]
        for p, (_, vraie) in zip(r["arguments"], n["arguments"]):
            bons_outil += p["categorie_probable"] == vraie
            bons_base += plus_frequente == vraie
            nb_args += 1
        ajouter_negociation(m, n, ecrire_csv=False)
    res = {"n_test": n_test, "erreur_outil": float(np.mean(err_outil)), "erreur_moyenne": float(np.mean(err_base)),
           "n_arguments": nb_args, "bonne_categorie_outil": bons_outil / nb_args,
           "bonne_categorie_base": bons_base / nb_args}
    print(f"Catégorie bien prévue sur {nb_args} arguments : outil {res['bonne_categorie_outil']:.1%}  |  "
          f"toujours la catégorie la plus fréquente {res['bonne_categorie_base']:.1%}")
    return res


# ---------------------------------------------------------------------------
# 10. Données FICTIVES (commande "demo" uniquement)
# ---------------------------------------------------------------------------
FAMILLES = {   # arguments = raisons avancées au fournisseur (pas des demandes)
    "concurrence": (["un concurrent nous propose moins cher", "nous avons reçu deux offres plus compétitives",
                     "plusieurs fournisseurs peuvent livrer le même produit"], "compromis", 0.10),
    "poids": (["nous représentons une grosse part de votre chiffre d'affaires", "nous sommes l'un de vos premiers clients",
               "notre commande est la plus importante de votre carnet"], "compromis", 0.05),
    "anciennete": (["nous travaillons ensemble depuis dix ans", "notre relation est ancienne et sans litige",
                    "nous sommes un client fidèle"], "consensus", 0.05),
    "croissance": (["nos volumes vont augmenter l'an prochain", "nous ouvrons deux nouveaux sites",
                    "notre activité est en forte croissance"], "consensus", 0.08),
    "cours": (["les cours des matières premières ont baissé", "le prix de l'énergie a reculé",
               "vos coûts de production sont en baisse"], "compromis", 0.15),
    "budget": (["notre direction impose une baisse des coûts", "notre budget achats est réduit",
                "nos marges sont sous pression"], "conflit", 0.25),
    "qualite": (["plusieurs lots ont été refusés au contrôle qualité", "vous avez eu trop de défauts cette année",
                 "nous avons subi des réclamations clients à cause de vos produits"], "compromis", 0.12),
    "substitution": (["nous étudions une alternative technologique", "nous pouvons changer de matériau",
                      "une solution interne pourrait remplacer votre offre"], "conflit", 0.18),
    "visibilite": (["nous vous transmettons nos prévisions sur 12 mois", "nous pouvons nous engager sur trois ans",
                    "nos commandes sont régulières et prévisibles"], "consensus", 0.02),
}
FACILITE = {"consensus": 0.95, "compromis": 0.75, "conflit": 0.45}


def _issue(niveau, rng):
    """Ce qui s'est passé pour un argument (FICTIF) : plus le niveau est haut, plus il passe facilement."""
    v = niveau + rng.normal(0, 0.15)
    return "consensus" if v >= 0.8 else ("compromis" if v >= 0.55 else "conflit")


def donnees_fictives(n, debut=0, seed=1):
    rng = np.random.default_rng(seed)
    negos = []
    for k in range(debut, debut + n):
        familles = rng.choice(list(FAMILLES), size=rng.integers(4, 8), replace=False)
        args = []
        for f in familles:
            phrases, cat_hab, difficulte = FAMILLES[f]
            args.append((str(rng.choice(phrases)), _issue(FACILITE[cat_hab] - difficulte, rng)))
        negos.append({"id": f"N{k:04d}", "arguments": args, "resultat": score_nego(args)})
    return negos


def creer_demo():
    for f in ("demo_historique.csv", "demo_realisees.csv", "demo_future.csv"):
        if os.path.exists(f):
            os.remove(f)
    ecrire_negos_csv(donnees_fictives(400, 0, seed=1), "demo_historique.csv")
    ecrire_negos_csv(donnees_fictives(30, 400, seed=2), "demo_realisees.csv")
    future = {"id": "FUTURE01", "resultat": None, "arguments": [
        ("nous travaillons ensemble depuis dix ans", None),
        ("un concurrent nous propose moins cher", None),
        ("les cours des matières premières ont baissé", None),
        ("notre budget achats est réduit", None)]}
    ecrire_negos_csv([future], "demo_future.csv")
    print("Fichiers FICTIFS créés : demo_historique.csv (400 négos), "
          "demo_realisees.csv (30 négos terminées), demo_future.csv (1 négo à estimer)")


# ---------------------------------------------------------------------------
# 11. Ligne de commande
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Prévision de négociations à partir de l'historique")
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
            afficher_estimation(estimer_negociation(m, n))
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
        print("Arguments seuls :")
        evaluer(historique, sans_contexte=True)
        if any(a_contexte(n) for n in historique):
            print("Arguments + contexte :")
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
