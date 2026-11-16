"""
Génère un historique FICTIF mais plausible de négociations, au format attendu par algo.py.
Une ligne par argument : nego_id ; date ; secteur ; categorie_achat ; sous_categorie ; raison ; 5 forces ;
rang ; argument ; categorie ; resultat  (tout sauf la date sert de contexte à l'algorithme).

Comment les résultats sont fabriqués (pour savoir ce que l'algo peut ou non retrouver) :
  - un argument est une RAISON avancée au fournisseur (« un concurrent nous propose moins cher »),
    pas une demande (« baisse de prix ») ;
  - chaque argument a une facilité propre selon sa famille (un argument factuel, comme une offre
    concurrente, passe mieux qu'une contrainte interne, comme un budget réduit) ;
  - chaque négo a un rapport de force décrit par les 5 forces de Porter (notes 1 à 5, dans le CSV),
    une raison (fin de contrat, hausse de prix...), un secteur, une catégorie d'achat et une
    sous-catégorie, qui jouent sur le résultat (effets FICTIFS choisis ci-dessous) ;
  - s'y ajoutent une part de rapport de force non mesurée et une "forme du jour", cachées ;
  - la catégorie de chaque argument est CE QUI S'EST PASSÉ (remplie après la négo) :
    consensus = argument accepté, compromis = accepté en partie / contre une concession, conflit = rejeté ;
  - résultat = moyenne des points (consensus 1, compromis 0,5, conflit 0), comme dans algo.py.
"""
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)

DOMAINES = ["Emballage", "Matières premières", "Transport et logistique", "Services informatiques", "Maintenance industrielle"]

# Familles d'ARGUMENTS, c'est-à-dire les raisons avancées au fournisseur pour appuyer une demande
# (pas les demandes elles-mêmes).
# famille: (formulations, issue habituelle, difficulté 0-0.35, domaines concernés ou None = tous)
#   issue habituelle = comment l'argument est le plus souvent reçu (FICTIF) : un argument factuel et
#   vérifiable passe mieux qu'une contrainte purement interne.
F = {
 "concurrence":  (["un concurrent nous propose un prix inférieur de {p} %", "nous avons reçu deux offres plus compétitives",
                   "plusieurs fournisseurs peuvent livrer le même produit", "un nouvel entrant nous a fait une offre à -{p} %"], "compromis", 0.10, None),
 "poids_client": (["nous représentons {r} % de votre chiffre d'affaires", "nous sommes l'un de vos trois premiers clients",
                   "notre commande est la plus importante de votre carnet", "nous pesons lourd dans votre activité"], "compromis", 0.05, None),
 "anciennete":   (["nous travaillons ensemble depuis {a} ans", "notre relation est ancienne et sans litige",
                   "nous sommes un client fidèle depuis plus de {a} ans"], "consensus", 0.06, None),
 "paiement":     (["nous avons toujours payé à l'échéance", "aucun incident de paiement depuis le début du contrat",
                   "nous pouvons payer plus vite en échange d'un geste commercial"], "consensus", 0.03, None),
 "croissance":   (["nos volumes vont augmenter de {r} % l'an prochain", "nous ouvrons deux nouveaux sites cette année",
                   "notre activité est en forte croissance", "nous prévoyons de doubler nos achats sur {a} ans"], "consensus", 0.08, None),
 "visibilite":   (["nous vous transmettons nos prévisions sur 12 mois", "nous pouvons nous engager sur {a} ans",
                   "un contrat long vous donne de la visibilité", "nos commandes sont régulières et prévisibles"], "consensus", 0.02, None),
 "cours_baisse": (["les cours des matières premières ont baissé de {p} %", "l'indice de vos coûts de production est en baisse",
                   "le prix de l'énergie a reculé depuis six mois", "vos coûts d'achat ont diminué cette année"], "compromis", 0.15, ["Emballage", "Matières premières", "Transport et logistique", "Maintenance industrielle"]),
 "inflation":    (["votre hausse dépasse largement l'inflation", "votre augmentation n'est pas justifiée par vos coûts",
                   "les autres fournisseurs n'ont pas augmenté leurs prix", "une hausse de {p} % est hors marché"], "conflit", 0.15, None),
 "qualite_def":  (["vous avez eu {d} % de défauts cette année", "plusieurs lots ont été refusés au contrôle qualité",
                   "nous avons subi des réclamations clients à cause de vos produits"], "compromis", 0.12, ["Emballage", "Matières premières", "Maintenance industrielle"]),
 "retards":      (["vous avez livré en retard à {s} reprises ce trimestre", "vos délais de livraison se sont dégradés",
                   "nos lignes se sont arrêtées à cause de vos retards"], "compromis", 0.10, ["Emballage", "Matières premières", "Transport et logistique", "Maintenance industrielle"]),
 "budget":       (["notre direction impose une baisse des coûts de {p} %", "notre budget achats est réduit cette année",
                   "nous devons tenir un objectif d'économies fixé par le groupe", "nos marges sont sous pression"], "conflit", 0.25, None),
 "substitution": (["nous étudions une alternative technologique", "nous pouvons changer de matériau si besoin",
                   "une solution interne pourrait remplacer votre offre", "nous testons déjà un produit de substitution"], "conflit", 0.18, None),
 "reference":    (["travailler avec nous est une référence pour vous", "nous pouvons vous recommander à nos filiales",
                   "notre marque vous ouvre d'autres clients", "nous acceptons de témoigner pour vous"], "consensus", 0.10, None),
 "simplicite":   (["nous regroupons nos commandes pour réduire vos coûts", "nous acceptons des livraisons groupées",
                   "nous pouvons simplifier notre cahier des charges", "nous standardisons nos références"], "consensus", 0.05, ["Emballage", "Matières premières", "Transport et logistique", "Maintenance industrielle"]),
 "rse":          (["nos clients exigent des engagements environnementaux", "la réglementation nous impose plus de recyclé",
                   "un partenariat RSE valoriserait votre entreprise"], "consensus", 0.08, None),
 "carburant":    (["le prix du gazole a baissé de {p} % depuis l'an dernier", "votre surcharge carburant ne suit pas la baisse du gazole",
                   "nos flux sont réguliers et optimisent vos tournées"], "compromis", 0.12, ["Transport et logistique"]),
 "fret_marche":  (["les tarifs du fret ont chuté sur le marché", "les transporteurs ont des capacités disponibles",
                   "le marché du transport est en surcapacité"], "compromis", 0.08, ["Transport et logistique"]),
 "licences_usage": (["nous n'utilisons que {r} % des licences achetées", "une partie des utilisateurs ne se connecte jamais",
                     "notre usage réel est bien inférieur au contrat"], "compromis", 0.10, ["Services informatiques"]),
 "open_source":  (["une solution open source couvre l'essentiel de notre besoin", "un concurrent propose une offre SaaS moins chère",
                   "nous pouvons internaliser une partie du service"], "conflit", 0.15, ["Services informatiques"]),
 "sla_rate":     (["vos engagements de service n'ont pas été tenus", "le temps de rétablissement a dépassé {h} heures plusieurs fois",
                   "nous avons subi des interruptions répétées"], "compromis", 0.12, ["Services informatiques", "Maintenance industrielle"]),
 "parc_machines": (["notre parc de machines va être renouvelé", "vos techniciens connaissent déjà nos installations",
                    "nous standardisons nos équipements sur votre marque"], "consensus", 0.06, ["Maintenance industrielle"]),
 "interventions": (["vos interventions dépassent souvent {h} heures", "plusieurs pannes n'ont pas été résolues du premier coup",
                    "le coût de vos pièces a augmenté sans explication"], "conflit", 0.12, ["Maintenance industrielle"]),
 "emballage_codev": (["nous pouvons co-développer un emballage plus léger", "nos nouvelles gammes vont utiliser vos formats",
                      "un allègement de {r} % réduirait vos coûts de matière"], "consensus", 0.06, ["Emballage"]),
}
BONUS_CAT = {"consensus": 0.93, "compromis": 0.75, "conflit": 0.48}   # facilité selon la famille d'argument
SEUILS = (0.8, 0.55)        # niveau ≥ 0,8 : consensus ; ≥ 0,55 : compromis ; sinon conflit (FICTIF)
POINTS = {"consensus": 1.0, "compromis": 0.5, "conflit": 0.0}
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
REMPL = {"p": [2, 3, 4, 5, 8, 10], "j": [45, 60, 75, 90], "a": [2, 3, 4], "d": [0.5, 1, 2], "s": [2, 3, 4],
         "h": [4, 8, 24, 48], "sla": [98, 99, 99.5], "r": [30, 50, 70]}


def formuler(gabarit):
    for k, vals in REMPL.items():
        gabarit = gabarit.replace("{" + k + "}", str(rng.choice(vals)).replace(".", ","))
    return gabarit


def generer(n=500):
    lignes = []
    dates = pd.date_range("2024-11-01", "2026-10-15", periods=n).normalize()
    for k in range(n):
        dom = rng.choice(DOMAINES, p=[0.25, 0.2, 0.2, 0.15, 0.2])
        possibles = [f for f, v in F.items() if v[3] is None or dom in v[3]]
        nb = int(rng.integers(4, 10))
        fams = rng.choice(possibles, size=min(nb, len(possibles)), replace=False)
        forces = {c: int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.22, 0.36, 0.22, 0.1])) for c in EFFET_FORCE}
        raison = str(rng.choice(list(EFFET_RAISON), p=PROBA_RAISON))
        secteur = str(rng.choice(list(EFFET_SECTEUR), p=PROBA_SECTEUR))
        sous_cat = str(rng.choice(SOUS_CATEGORIES[dom]))
        rapport_force = (sum(EFFET_FORCE[c] * (v - 3) for c, v in forces.items()) + EFFET_RAISON[raison]
                         + EFFET_SECTEUR[secteur] + EFFET_CATEGORIE[dom] + EFFET_SOUS_CATEGORIE.get(sous_cat, 0.0)
                         + rng.normal(0, 0.03))  # + part non mesurée
        forme = rng.normal(0, 0.06)             # caché : talent / forme du jour
        args = []
        for f in fams:                          # ordre dans lequel le négociateur les a avancés
            formulations, cat_hab, diff, _ = F[f]
            niveau = BONUS_CAT[cat_hab] - diff + rapport_force + forme + rng.normal(0, 0.15)
            cat = "consensus" if niveau >= SEUILS[0] else ("compromis" if niveau >= SEUILS[1] else "conflit")
            args.append((formuler(str(rng.choice(formulations))), cat))
        res = round(float(np.mean([POINTS[c] for _, c in args])), 4)
        for i, (texte, cat) in enumerate(args, 1):
            lignes.append({"nego_id": f"N{k + 1:04d}", "date": dates[k].strftime("%Y-%m-%d"), "secteur": secteur,
                           "categorie_achat": dom, "sous_categorie": sous_cat, "raison": raison, **forces,
                           "rang": i, "argument": texte, "categorie": cat, "resultat": res})
    return pd.DataFrame(lignes)


if __name__ == "__main__":
    sortie = sys.argv[1] if len(sys.argv) > 1 else "historique.csv"
    df = generer(500)
    df.to_csv(sortie, sep=";", index=False, encoding="utf-8-sig")
    par_nego = df.groupby("nego_id").agg(n=("rang", "size"), res=("resultat", "first"))
    print(f"{df.nego_id.nunique()} négos, {len(df)} arguments -> {sortie}")
    print(f"Arguments par négo : {par_nego.n.min()} à {par_nego.n.max()} (moyenne {par_nego.n.mean():.1f})")
    print(f"Résultat moyen : {par_nego.res.mean():.0%} ; médiane {par_nego.res.median():.0%}")
    print("Ce qui s'est passé (part des arguments) :", df.categorie.value_counts(normalize=True).round(2).to_dict())
