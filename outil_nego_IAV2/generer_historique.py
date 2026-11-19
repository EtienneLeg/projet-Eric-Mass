"""
Génère un historique FICTIF mais plausible de négociations, au format attendu par algo.py.
Une ligne par output recherché : nego_id ; date ; secteur ; categorie_achat ; sous_categorie ; raison ;
5 forces ; rang ; argument (= l'output) ; categorie ; resultat ; contrepartie
(tout sauf la date sert de contexte à l'algorithme).

Comment les résultats sont fabriqués (pour savoir ce que l'algo peut ou non retrouver) :
  - chaque output a une difficulté RÉELLE qui dépend de son type (sa catégorie habituelle + une
    difficulté propre) et, pour les outputs chiffrés, du niveau demandé (baisse de 10 % plus dure que
    4 %). La catégorie saisie par le négociateur n'est qu'une PRÉVISION (juste environ 7 fois sur 10) :
    elle ne change pas les chances d'obtenir l'output ;
  - chaque négo a un rapport de force décrit par les 5 forces de Porter (notes 1 à 5, dans le CSV),
    une raison (fin de contrat, hausse de prix...), un secteur, une catégorie d'achat et une
    sous-catégorie, qui jouent sur le résultat (effets FICTIFS choisis ci-dessous) ;
  - s'y ajoutent une part de rapport de force non mesurée et une "forme du jour", cachées ;
  - on obtient les outputs DANS L'ORDRE de la colonne jusqu'au premier échec
    -> résultat = part des outputs obtenus (niveau atteint) ;
  - une négo sur deux contient une ÉCHELLE de paliers sur le prix ou le délai de paiement
    (ex. baisse de 2 %, 4 %, 6 %, 8 %), rangés du plus facile au plus dur : beaucoup d'options
    intermédiaires (compromis), avec un risque qui monte à chaque palier ;
  - chaque négo se fait avec un fournisseur (codes FICTIFS : EMB-01...) qui revient plusieurs fois,
    certains plus durs que d'autres (effet caché) ;
  - le trimestre de la négo joue un peu (effet FICTIF : plus facile en fin d'année, plus dur l'été) ;
  - environ 1,5 % de négos exceptionnelles (énorme coup de chance ou de malchance), marquées exclue = oui ;
  - un output obtenu l'est souvent en échange d'une contrepartie (compromis), tirée parmi les
    contreparties plausibles pour ce type d'output ; réglé pour que les compromis représentent environ
    la moitié des issues (« autant de compromis que de consensus + conflit », M. Mass).
"""
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)

DOMAINES = ["Emballage", "Matières premières", "Transport et logistique", "Services informatiques", "Maintenance industrielle"]

# famille: (formulations, catégorie habituelle, difficulté 0-0.35, domaines concernés ou None = tous)
F = {
 "prix":        (["baisse de prix de {p} %", "réduction tarifaire de {p} %", "diminution du prix unitaire", "remise de {p} % sur le tarif catalogue", "alignement du prix sur la concurrence"], "compromis", 0.2, None),
 "delai_paie":  (["paiement à {j} jours", "allonger le délai de paiement à {j} jours", "étendre les délais de règlement", "passage du paiement de 30 à {j} jours"], "compromis", 0.12, None),
 "volume":      (["maintien du volume annuel", "garder les volumes actuels", "engagement de volume sur l'année", "volume constant par rapport à l'an dernier"], "consensus", 0.02, None),
 "remise_vol":  (["remise progressive selon les volumes", "ristourne de fin d'année sur le volume", "paliers de remise par tranche de volume"], "compromis", 0.12, None),
 "duree":       (["contrat sur {a} ans", "allonger la durée du contrat", "engagement pluriannuel", "reconduction tacite du contrat"], "compromis", 0.06, None),
 "exclu":       (["exclusivité sur la catégorie", "contrat exclusif", "devenir fournisseur unique", "exclusivité régionale"], "conflit", 0.30, None),
 "penal":       (["pénalités de retard", "sanctions en cas de retard de livraison", "clause de pénalité de {p} % par semaine de retard"], "conflit", 0.20, ["Emballage", "Matières premières", "Transport et logistique", "Maintenance industrielle"]),
 "franco":      (["livraison franco de port", "transport pris en charge par le fournisseur", "partage des coûts logistiques"], "compromis", 0.10, ["Emballage", "Matières premières", "Maintenance industrielle"]),
 "indexation":  (["clause de révision des prix", "indexation sur les matières premières", "formule d'indexation trimestrielle", "plafonnement des hausses à {p} %"], "compromis", 0.18, ["Emballage", "Matières premières", "Transport et logistique"]),
 "qualite":     (["audit qualité annuel", "renforcement des contrôles qualité", "certificat de conformité systématique", "taux de défaut inférieur à {d} %"], "consensus", 0.05, ["Emballage", "Matières premières", "Maintenance industrielle"]),
 "stock":       (["stock de sécurité chez le fournisseur", "dépôt de stock en consignation", "stock tampon de {s} semaines"], "compromis", 0.15, ["Emballage", "Matières premières", "Maintenance industrielle"]),
 "planning":    (["planning de livraison partagé", "prévisions de commande transmises chaque mois", "point de suivi trimestriel"], "consensus", 0.0, None),
 "carburant":   (["surcharge carburant plafonnée", "suppression de la surcharge carburant", "indexation gazole encadrée"], "conflit", 0.22, ["Transport et logistique"]),
 "delai_liv":   (["délai de livraison réduit à {h} heures", "livraison express sans surcoût", "créneaux de livraison garantis"], "compromis", 0.15, ["Transport et logistique", "Maintenance industrielle"]),
 "sla":         (["engagement de niveau de service à {sla} %", "SLA avec pénalités", "temps de rétablissement garanti en {h} heures"], "compromis", 0.15, ["Services informatiques", "Maintenance industrielle"]),
 "licences":    (["gel du prix des licences", "licences supplémentaires offertes", "réduction du coût par utilisateur"], "conflit", 0.22, ["Services informatiques"]),
 "reversibilite": (["clause de réversibilité des données", "accompagnement en fin de contrat", "export des données sans frais"], "compromis", 0.10, ["Services informatiques"]),
 "astreinte":   (["astreinte {h}h sur 24 incluse", "intervention sur site sous {h} heures", "technicien dédié sur site"], "conflit", 0.20, ["Maintenance industrielle", "Services informatiques"]),
 "pieces":      (["pièces détachées incluses dans le forfait", "remise sur les pièces de rechange", "garantie pièces étendue à {a} ans"], "compromis", 0.14, ["Maintenance industrielle"]),
 "rse":         (["engagement sur l'empreinte carbone", "matériaux recyclés à {r} %", "reporting RSE annuel"], "consensus", 0.06, None),
 "innovation":  (["accès prioritaire aux nouveaux produits", "atelier d'innovation commun", "feuille de route produit partagée"], "consensus", 0.04, ["Emballage", "Services informatiques"]),
 "codev_emb":   (["co-développement d'un nouvel emballage", "allègement des emballages de {r} %", "test de nouveaux formats d'emballage"], "consensus", 0.06, ["Emballage"]),
 "formation":   (["formation des équipes incluse", "session de formation annuelle offerte", "support utilisateur renforcé"], "consensus", 0.03, ["Services informatiques", "Maintenance industrielle"]),
 "acompte":     (["suppression de l'acompte", "acompte réduit à {p} %", "paiement à la livraison"], "conflit", 0.18, None),
 "garantie":    (["extension de garantie gratuite", "garantie de {a} ans au lieu de 1", "remplacement sous garantie sans frais de port"], "compromis", 0.12, ["Emballage", "Maintenance industrielle", "Services informatiques"]),
}
BONUS_CAT = {"consensus": 0.93, "compromis": 0.75, "conflit": 0.48}
# Effet FICTIF du niveau demandé : (repère dans la formulation, difficulté ajoutée par unité au-dessus
# du niveau de référence, niveau de référence). Ex. prix : +0,03 par point de baisse au-delà de 5 %.
EFFET_NIVEAU = {"prix": ("p", 0.03, 5), "delai_paie": ("j", 0.004, 60), "penal": ("p", 0.02, 5),
                "indexation": ("p", -0.02, 5), "acompte": ("p", -0.02, 5), "delai_liv": ("h", -0.003, 24),
                "astreinte": ("h", -0.003, 24), "sla": ("sla", 0.1, 99), "rse": ("r", 0.002, 50),
                "codev_emb": ("r", 0.002, 50), "qualite": ("d", -0.05, 1)}
# Contreparties plausibles pour chaque type d'output (mêmes libellés que CONTREPARTIES dans reglages.py)
VOL, DUREE, PAIE, PRIX = "engagement de volume", "durée de contrat plus longue", "paiement plus rapide", \
    "concession sur le prix"
ACOMPTE, LIV, GROUPE, EXCLU = "acompte à la commande", "délais de livraison assouplis", "commandes regroupées", \
    "exclusivité accordée"
CONTREPARTIES_PLAUSIBLES = {
    "prix": [VOL, DUREE, PAIE], "delai_paie": [PRIX, VOL, ACOMPTE], "volume": [DUREE], "remise_vol": [VOL, DUREE],
    "duree": [PRIX, VOL], "exclu": [PRIX, VOL, DUREE], "penal": [PRIX, LIV], "franco": [GROUPE, VOL, PRIX],
    "indexation": [DUREE, VOL], "qualite": [PRIX], "stock": [VOL, DUREE, PRIX], "planning": [GROUPE],
    "carburant": [DUREE, VOL], "delai_liv": [GROUPE, PRIX], "sla": [PRIX, DUREE], "licences": [DUREE, VOL],
    "reversibilite": [DUREE], "astreinte": [PRIX, DUREE], "pieces": [DUREE, VOL], "rse": [PRIX, DUREE],
    "innovation": [EXCLU, VOL], "codev_emb": [EXCLU, VOL], "formation": [DUREE], "acompte": [VOL, DUREE],
    "garantie": [PRIX, DUREE],
}
# Probabilité qu'un output obtenu l'ait été en échange d'une contrepartie, selon le type d'output
# (sa catégorie habituelle : un output naturellement « compromis » s'obtient souvent contre quelque chose)
PROBA_CONTREPARTIE = {"consensus": 0.5, "compromis": 0.95, "conflit": 0.9}
# Effet FICTIF de chaque force (par point au-dessus ou en dessous de 3) et de chaque raison.
# Scénario retenu : le rapport de force compte nettement (hypothèse, pas une mesure de la réalité).
EFFET_FORCE = {"force_fournisseur": 0.07, "force_acheteur": 0.07, "force_substituts": 0.05,
               "force_entrants": 0.02, "force_rivalite": 0.05}
EFFET_RAISON = {"fin de contrat": 0.0, "renouvellement": 0.02, "nouveau besoin": 0.03,
                "hausse de prix du fournisseur": -0.05, "saisonnalité": -0.02, "autre": 0.0}
PROBA_RAISON = [0.25, 0.3, 0.15, 0.15, 0.1, 0.05]
# Situation : secteur de l'entreprise qui négocie, catégorie d'achat (= domaine) et sous-catégorie.
EFFET_SECTEUR = {"Grande distribution": 0.05, "Industrie agroalimentaire": 0.0, "Industrie manufacturière": 0.0,
                 "Transport et logistique": -0.02, "BTP et construction": -0.03, "Énergie": 0.02,
                 "Santé et pharmacie": -0.04, "Services": 0.0}
PROBA_SECTEUR = [0.2, 0.15, 0.2, 0.1, 0.1, 0.08, 0.07, 0.1]
SOUS_CATEGORIES = {"Emballage": ["Verre", "Plastique", "Carton", "Métal"],
                   "Matières premières": ["Métaux", "Produits chimiques", "Produits agricoles", "Énergie"],
                   "Transport et logistique": ["Routier", "Maritime", "Aérien", "Entreposage"],
                   "Services informatiques": ["Logiciels et licences", "Infogérance", "Matériel"],
                   "Maintenance industrielle": ["Pièces détachées", "Interventions sur site", "Contrats de maintenance"]}
EFFET_CATEGORIE = {"Emballage": 0.02, "Matières premières": -0.04, "Transport et logistique": 0.0,
                   "Services informatiques": -0.03, "Maintenance industrielle": 0.01}
EFFET_SOUS_CATEGORIE = {"Verre": -0.04, "Carton": 0.03, "Métaux": -0.03, "Maritime": -0.03, "Routier": 0.02,
                        "Logiciels et licences": -0.03}
REMPL = {"p": [2, 3, 4, 5, 6, 8, 10], "j": [45, 60, 75, 90], "a": [2, 3, 4], "d": [0.5, 1, 2], "s": [2, 3, 4],
         "h": [4, 8, 24, 48], "sla": [98, 99, 99.5], "r": [30, 50, 70]}


# Échelles de paliers : (famille, formulation, repère du niveau)
ECHELLES = [("prix", "baisse de prix de {p} %", "p"), ("delai_paie", "paiement à {j} jours", "j")]
# Effet FICTIF du trimestre (hypothèse : en fin d'année les fournisseurs veulent atteindre leurs objectifs)
EFFET_TRIMESTRE = {"T1": 0.0, "T2": 0.0, "T3": -0.03, "T4": 0.04}
# Fournisseurs FICTIFS : 8 par catégorie d'achat, chacun avec un effet caché (plus ou moins dur)
PREFIXES = {"Emballage": "EMB", "Matières premières": "MAT", "Transport et logistique": "TRA",
            "Services informatiques": "INF", "Maintenance industrielle": "MAI"}
FOURNISSEURS = {dom: [f"{pre}-{i:02d}" for i in range(1, 9)] for dom, pre in PREFIXES.items()}
PROBA_FOURNISSEUR = np.array([0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.07, 0.06])   # certains reviennent souvent
EFFET_FOURNISSEUR = {f: float(rng.normal(0, 0.06)) for liste in FOURNISSEURS.values() for f in liste}
PART_EXCEPTIONNELLES = 0.015


def trimestre(jour):
    return f"T{(jour.month - 1) // 3 + 1}"


def formuler(gabarit, famille):
    """Remplit la formulation ; renvoie (texte, difficulté ajoutée par le niveau demandé)."""
    ajout = 0.0
    for k, vals in REMPL.items():
        if "{" + k + "}" in gabarit:
            v = rng.choice(vals)
            gabarit = gabarit.replace("{" + k + "}", str(v).replace(".", ","))
            if famille in EFFET_NIVEAU and EFFET_NIVEAU[famille][0] == k:
                _, pente, ref = EFFET_NIVEAU[famille]
                ajout += pente * (float(v) - ref)
    return gabarit, ajout


def generer(n=2000):
    lignes = []
    dates = pd.date_range("2024-11-01", "2026-10-15", periods=n).normalize()
    for k in range(n):
        dom = str(rng.choice(DOMAINES, p=[0.25, 0.2, 0.2, 0.15, 0.2]))
        fournisseur = str(rng.choice(FOURNISSEURS[dom], p=PROBA_FOURNISSEUR))
        trim = trimestre(dates[k])
        exceptionnelle = rng.random() < PART_EXCEPTIONNELLES
        # échelle de paliers (une négo sur deux) : sa famille n'est pas tirée une seconde fois
        echelle = ECHELLES[int(rng.integers(len(ECHELLES)))] if rng.random() < 0.5 else None
        possibles = [f for f, v in F.items() if (v[3] is None or dom in v[3]) and (not echelle or f != echelle[0])]
        nb = int(rng.integers(3, 8)) if echelle else int(rng.integers(4, 10))
        fams = rng.choice(possibles, size=min(nb, len(possibles)), replace=False)
        forces = {c: int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.22, 0.36, 0.22, 0.1])) for c in EFFET_FORCE}
        raison = str(rng.choice(list(EFFET_RAISON), p=PROBA_RAISON))
        secteur = str(rng.choice(list(EFFET_SECTEUR), p=PROBA_SECTEUR))
        sous_cat = str(rng.choice(SOUS_CATEGORIES[dom]))
        rapport_force = (sum(EFFET_FORCE[c] * (v - 3) for c, v in forces.items()) + EFFET_RAISON[raison]
                         + EFFET_SECTEUR[secteur] + EFFET_CATEGORIE[dom] + EFFET_SOUS_CATEGORIE.get(sous_cat, 0.0)
                         + EFFET_TRIMESTRE[trim] + EFFET_FOURNISSEUR[fournisseur]
                         + rng.normal(0, 0.03))  # + part non mesurée
        forme = rng.normal(0, 0.06)             # caché : talent / forme du jour
        if exceptionnelle:                      # coup de chance ou de malchance hors norme
            forme += float(rng.choice([-0.45, 0.45]))
        args = []
        for f in fams:
            formulations, cat_hab, diff, _ = F[f]
            r = rng.random()
            if r < 0.72:
                cat = cat_hab
            else:                               # le négociateur classe parfois autrement
                autres = [c for c in BONUS_CAT if c != cat_hab]
                cat = str(rng.choice(autres))
            texte, ajout = formuler(str(rng.choice(formulations)), f)
            args.append((texte, cat, diff + ajout, f))
        paliers = []
        if echelle:
            f, gabarit, cle = echelle
            nb_paliers = int(rng.integers(2, 5))
            valeurs = sorted(rng.choice(REMPL[cle], size=nb_paliers, replace=False))
            _, pente, ref = EFFET_NIVEAU[f]
            for i, v in enumerate(valeurs):    # prévision : dernier palier d'une échelle de 3 ou plus classé conflit
                cat = "conflit" if (i == nb_paliers - 1 and nb_paliers >= 3) else "compromis"
                paliers.append((gabarit.replace("{" + cle + "}", str(v)), cat, F[f][2] + pente * (float(v) - ref), f))
        args += paliers
        rng.shuffle(args)
        args.sort(key=lambda a: list(BONUS_CAT).index(a[1]))   # colonne : consensus -> compromis -> conflit
        if paliers:                             # les paliers d'une échelle restent du plus facile au plus dur
            places = [i for i, a in enumerate(args) if any(a is p for p in paliers)]
            for i, p in zip(places, paliers):
                args[i] = p
        obtenus, contreparties = 0, []
        for _, cat, diff, f in args:
            reel = F[f][1]                      # difficulté réelle : type d'output, PAS la catégorie saisie
            p = BONUS_CAT[reel] - diff + rapport_force + forme
            if rng.random() < np.clip(p, 0.02, 0.99):
                obtenus += 1
                contreparties.append(str(rng.choice(CONTREPARTIES_PLAUSIBLES[f]))
                                     if rng.random() < PROBA_CONTREPARTIE[reel] else "")
            else:
                break
        contreparties += [""] * (len(args) - len(contreparties))    # outputs non obtenus : pas de contrepartie
        res = round(obtenus / len(args), 3)
        for i, ((texte, cat, _, _), cp) in enumerate(zip(args, contreparties), 1):
            lignes.append({"nego_id": f"N{k + 1:04d}", "date": dates[k].strftime("%Y-%m-%d"), "trimestre": trim,
                           "fournisseur": fournisseur, "secteur": secteur,
                           "categorie_achat": dom, "sous_categorie": sous_cat, "raison": raison, **forces,
                           "rang": i, "argument": texte, "categorie": cat, "resultat": res, "contrepartie": cp,
                           "exclue": "oui" if exceptionnelle else ""})
    return pd.DataFrame(lignes)


if __name__ == "__main__":
    sortie = sys.argv[1] if len(sys.argv) > 1 else "historique.csv"
    df = generer(2000)
    df.to_csv(sortie, sep=";", index=False, encoding="utf-8-sig")
    par_nego = df.groupby("nego_id").agg(n=("rang", "size"), res=("resultat", "first"))
    print(f"{df.nego_id.nunique()} négos, {len(df)} arguments -> {sortie}")
    print(f"Arguments par négo : {par_nego.n.min()} à {par_nego.n.max()} (moyenne {par_nego.n.mean():.1f})")
    print(f"Résultat moyen : {par_nego.res.mean():.0%} ; médiane {par_nego.res.median():.0%}")
    print("Catégories saisies :", df.categorie.value_counts(normalize=True).round(2).to_dict())
    print(f"Négos avec échelle de paliers : {df[df.argument.str.match(r'(baisse de prix de|paiement à) [0-9]+')].groupby('nego_id').size().ge(2).sum()}"
          f" · fournisseurs : {df.fournisseur.nunique()} · négos exclues (exceptionnelles) : "
          f"{df[df.exclue == 'oui'].nego_id.nunique()}")
    # issues réelles des outputs abordés (obtenus ou bloquants)
    issues = []
    for _, g in df.groupby("nego_id"):
        k = int(round(g.resultat.iloc[0] * len(g)))
        for i, cp in enumerate(g.contrepartie):
            if i < k:
                issues.append("compromis" if cp else "consensus")
            elif i == k:
                issues.append("conflit")
    print("Issues réelles :", pd.Series(issues).value_counts(normalize=True).round(2).to_dict())
