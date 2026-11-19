"""
RÉGLAGES DE L'OUTIL DE NÉGOCIATION
==================================

Tous les paramètres modifiables sont ici. Le reste du code (algo.py, interface.py) les lit dans ce
fichier : il ne faut rien changer ailleurs.

Pour chaque réglage :
  - ce qu'il fait ;
  - « ↑ » : ce qui se passe si on l'augmente, « ↓ » : si on le diminue ;
  - [IMMÉDIAT] : pris en compte au prochain lancement de l'interface ;
    [RECONSTRUCTION] : l'arbre doit être recalculé. L'interface le fait automatiquement au lancement
    si elle détecte le changement ; en ligne de commande : python3 algo.py reconstruire

Méthode conseillée : changer UN réglage à la fois, puis lancer l'onglet « Évaluer » (ou
python3 algo.py evaluer) et garder la valeur qui donne la plus petite erreur. Attention : si on règle
tout sur le même historique de test, on finit par « apprendre le test » ; garder des réglages simples.

Une vérification automatique signale au lancement toute valeur incohérente (négative, hors limites...).
"""

# =============================================================================
# 1. COMPARAISON DES ARGUMENTS
# =============================================================================

# Ressemblance accordée à deux arguments selon leur catégorie (consensus / compromis / conflit).
# Le score de deux arguments = ressemblance des phrases × ce coefficient.
#   ex. « baisse de prix » en conflit vs « réduction tarifaire » en compromis : 0,95 × 0,5
# Mettre 1 partout = ignorer le classement ; mettre 0 hors diagonale = n'accepter que la même catégorie.
# Valeurs entre 0 et 1, le tableau doit rester symétrique.                               [IMMÉDIAT]
COEF_CATEGORIE = {
    "consensus": {"consensus": 1.0, "compromis": 0.5, "conflit": 0.0},
    "compromis": {"consensus": 0.5, "compromis": 1.0, "conflit": 0.5},
    "conflit":   {"consensus": 0.0, "compromis": 0.5, "conflit": 1.0},
}

# Poids de chaque tiroir dans le vecteur d'une négo (utilisé pour construire l'arbre).
# ↑ sur « conflit » : l'arbre regroupe d'abord les négos qui ont les mêmes points de conflit.
# Valeurs ≥ 0 ; 1 = poids normal.                                                   [RECONSTRUCTION]
POIDS_COMPARTIMENTS = {"consensus": 1.0, "compromis": 1.0, "conflit": 1.0}


# =============================================================================
# 2. L'ARBRE
# =============================================================================

# Nombre minimal de négos dans un groupe (feuille) de l'arbre.
# ↑ : groupes plus gros, moyennes plus stables mais moins précises (on mélange des négos différentes).
# ↓ : groupes plus fins, mais moins de négos pour estimer. Entier ≥ 2.              [RECONSTRUCTION]
TAILLE_MIN_FEUILLE = 15

# Au-delà de cette taille, un groupe est recoupé en deux quand on ajoute une négo.
# Doit valoir au moins 2 × TAILLE_MIN_FEUILLE.                                      [RECONSTRUCTION]
TAILLE_MAX_FEUILLE = 60

# Part de négos ajoutées (par rapport à la dernière construction) au-delà de laquelle l'outil
# conseille de reconstruire l'arbre. 0,25 = après +25 %. Entre 0 et 1.                     [IMMÉDIAT]
PART_AVANT_RECONSTRUCTION = 0.25


# =============================================================================
# 3. ESTIMATION
# =============================================================================

# Nombre de négos passées les plus proches utilisées pour calculer l'estimation.
# ↑ : estimation plus stable mais plus « moyenne » ; ↓ : plus spécifique mais plus bruitée.
# Entier ≥ 1.                                                                             [IMMÉDIAT]
K_VOISINS = 25

# Ressemblance minimale (entre 0 et 1) pour qu'une négo passée soit jugée « vraiment proche ».
# En dessous pour toutes les négos : l'outil affiche « pas assez de précédents comparables ».
# À réajuster si on change de modèle d'embedding (le vrai modèle donne des ressemblances plus
# élevées que le repli TF-IDF).                                                          [IMMÉDIAT]
SEUIL_SIMILARITE = 0.45

# Nombre minimal de négos « vraiment proches » dans le groupe d'arrivée ; sinon la recherche remonte
# d'un niveau dans l'arbre (groupe plus large). Entier ≥ 1.                              [IMMÉDIAT]
MIN_PROCHES = 5

# Poids des ARGUMENTS dans le choix des négos voisines ; le reste (1 - ALPHA) va au contexte.
#   1   : voisins choisis sur les arguments seuls (le contexte ne sert alors qu'à la correction) ;
#   0,8 : 80 % arguments, 20 % contexte (forces, raison, secteur, catégorie, sous-catégorie).
# Entre 0 et 1.                                                                     [RECONSTRUCTION]
ALPHA = 0.8


# =============================================================================
# 4. IMPORTANCE DU CONTEXTE DANS LE CHOIX DES VOISINS
#    (ne joue que si ALPHA < 1)
# =============================================================================
# Remarque importante : l'effet de chaque élément sur l'ESTIMATION (« +6,5 pts Secteur ») n'est pas
# réglé à la main : il est APPRIS sur l'historique. Les poids ci-dessous règlent seulement quelles
# négos passées sont jugées comparables. Valeurs ≥ 0 ; 0 = élément ignoré ; 2 = deux fois plus important.

# Poids de chaque force (mêmes clés que FORCES plus bas).                                [IMMÉDIAT]
POIDS_FORCES = {
    "force_fournisseur": 1.0,
    "force_acheteur": 1.0,
    "force_substituts": 1.0,
    "force_entrants": 1.0,
    "force_rivalite": 1.0,
}

# Poids des éléments de situation (une force vaut 1 : les 5 forces pèsent donc 5 au total par défaut).
# [IMMÉDIAT]
POIDS_SITUATION = {
    "raison": 1.0,
    "secteur": 1.0,
    "categorie_achat": 1.0,      # inclut la sous-catégorie, réglée juste en dessous
    "trimestre": 1.0,            # moment de l'année (T1 à T4), déduit de la date de la négo
}

# Ressemblance de deux négos de même catégorie d'achat mais de sous-catégories différentes
# (même sous-catégorie = 1, catégorie différente = 0). Entre 0 et 1.                     [IMMÉDIAT]
SIMILARITE_AUTRE_SOUS_CATEGORIE = 0.5


# =============================================================================
# 5. APPRENTISSAGE DES EFFETS DU CONTEXTE
# =============================================================================

# Prudence de l'apprentissage des effets (régression « ridge »).
# ↑ : effets plus modérés (moins de risque de sur-interpréter le hasard) ;
# ↓ : effets plus marqués. Nombre > 0.                                              [RECONSTRUCTION]
REGULARISATION = 1.0

# Nombre minimal de négos avec contexte pour apprendre les effets. En dessous, le contexte
# n'est pas utilisé dans l'estimation. Entier ≥ 1.                                  [RECONSTRUCTION]
MIN_NEGOS_CONTEXTE = 30


# =============================================================================
# 6. CONTRÔLE DU CLASSEMENT DES ARGUMENTS
# =============================================================================

# Nombre d'arguments passés similaires regardés pour vérifier le classement d'un argument.
CONTROLE_NB_ARGUMENTS = 15            # entier ≥ 1                                       [IMMÉDIAT]
# Part minimale (entre 0 et 1) d'arguments similaires classés autrement pour afficher une alerte.
# ↑ : moins d'alertes, seulement les cas nets ; ↓ : plus d'alertes.                       [IMMÉDIAT]
CONTROLE_SEUIL_ALERTE = 0.6


# =============================================================================
# 7. JUSQU'OÙ ALLER : RISQUE DE CONFLIT PAR OUTPUT
# =============================================================================
# Pour chaque output recherché, l'outil estime la probabilité de chaque issue, à partir des outputs
# semblables déjà négociés dans des contextes comparables :
#   consensus = obtenu sans contrepartie ; compromis = obtenu avec une contrepartie ;
#   conflit   = pas obtenu (la négo a bloqué sur ce point).
# On descend la liste dans l'ordre donné par l'utilisateur ; la ligne « jusqu'ici » se place juste
# avant le premier output dont le risque de conflit dépasse ce seuil.
# ↑ : on accepte plus de risque, la ligne descend ; ↓ : plus prudent, la ligne remonte.
# Entre 0 et 1 (0,40 = 40 %). Modifiable aussi dans l'interface.                         [IMMÉDIAT]
SEUIL_CONFLIT = 0.40

# Nombre maximal d'outputs passés semblables utilisés pour estimer un output.
# ↑ : estimation plus stable mais plus « moyenne » ; ↓ : plus spécifique mais plus bruitée.
# Entier ≥ 1.                                                                             [IMMÉDIAT]
K_ARGUMENTS = 40

# Prudence quand il y a peu d'outputs semblables : l'estimation est tirée vers le taux moyen de conflit
# de tout l'historique, comme si on ajoutait ce nombre d'exemples « moyens ».
# ↑ : estimations plus prudentes (proches de la moyenne) ; 0 : aucune correction. Nombre ≥ 0.  [IMMÉDIAT]
POIDS_A_PRIORI = 5.0

# Ressemblance accordée à deux outputs selon la catégorie choisie par l'utilisateur, pour estimer le
# risque de conflit. Par défaut 1 partout : on compare à TOUS les outputs semblables, quelle que soit
# leur catégorie. La catégorie est une PRÉVISION du négociateur : la changer ne doit pas changer le
# risque estimé (elle sert seulement à signaler un écart entre la prévision et l'historique).
# Mettre moins de 1 hors diagonale pour privilégier les outputs classés comme le tien.
# Valeurs entre 0 et 1, tableau symétrique.                                              [IMMÉDIAT]
COEF_CATEGORIE_OUTPUTS = {
    "consensus": {"consensus": 1.0, "compromis": 1.0, "conflit": 1.0},
    "compromis": {"consensus": 1.0, "compromis": 1.0, "conflit": 1.0},
    "conflit":   {"consensus": 1.0, "compromis": 1.0, "conflit": 1.0},
}

# Prise en compte du niveau chiffré : « baisse de prix de 4 % » ressemble moins à « baisse de prix
# de 10 % » qu'à « baisse de prix de 5 % ». Ressemblance multipliée par (petit / grand) ** ce réglage
# quand les deux outputs contiennent un nombre. 0 = ignorer les chiffres ; 1 = effet normal ;
# 2 = effet fort. Nombre ≥ 0.                                                             [IMMÉDIAT]
POIDS_NIVEAU = 1.0

# Négociation avec un fournisseur / interlocuteur déjà rencontré : les outputs déjà négociés avec lui
# comptent ce nombre de fois plus que les autres dans l'estimation.
# 1 = aucun traitement particulier ; 2 = deux fois plus ; ↑ : l'outil se fie davantage à l'historique avec
# cette personne. Nombre ≥ 1.                                                             [IMMÉDIAT]
POIDS_MEME_FOURNISSEUR = 2.0

# Contreparties proposées à la saisie d'une négo terminée (liste fermée, pour que l'outil puisse
# compter les contreparties habituelles). À compléter avec M. Mass.                       [IMMÉDIAT]
CONTREPARTIES = ["engagement de volume", "durée de contrat plus longue", "paiement plus rapide",
                 "concession sur le prix", "acompte à la commande", "délais de livraison assouplis",
                 "commandes regroupées", "exclusivité accordée", "autre"]


# =============================================================================
# 8. MODÈLES D'IA
# =============================================================================

# Modèle d'embedding (transforme les phrases en vecteurs). Téléchargé une fois, puis local.
MODELE_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"                         # [RECONSTRUCTION]

# Assistant IA de la v3 (Ollama). Modèle à télécharger avec : ollama pull <nom>         [IMMÉDIAT]
MODELE_OLLAMA = "qwen2.5:3b"
URL_OLLAMA = "http://localhost:11434"     # à changer pour un serveur IA interne à l'entreprise
DELAI_OLLAMA_SECONDES = 180               # le premier appel charge le modèle : il peut être long


# =============================================================================
# 9. LISTES PROPOSÉES À LA SAISIE
#    Les modifier change les menus de l'interface.                                  [RECONSTRUCTION]
# =============================================================================

# Les forces : (colonne dans le CSV, nom court, question résumée, signification de 1, de 5).
# Toutes notées de 1 à 5, 5 = situation FAVORABLE au négociateur.
# La question résumée sert à l'assistant IA ; l'interface pose les questions de QUESTIONS_FORCES.
# Ajouter ou retirer une force : modifier cette liste, POIDS_FORCES et QUESTIONS_FORCES ensemble.
FORCES = [
    ("force_fournisseur", "Alternatives fournisseurs",
     "Combien de fournisseurs pourraient vous livrer la même chose ?", "aucun", "beaucoup"),
    ("force_acheteur", "Notre poids d'acheteur",
     "Quel poids représentez-vous dans le chiffre d'affaires du fournisseur ?", "négligeable", "client majeur"),
    ("force_substituts", "Substituts",
     "Existe-t-il une alternative technologique crédible ?", "aucune", "plusieurs"),
    ("force_entrants", "Nouveaux entrants",
     "De nouveaux fournisseurs peuvent-ils arriver facilement sur ce marché ?", "non", "oui, facilement"),
    ("force_rivalite", "Rivalité entre fournisseurs",
     "Les fournisseurs se battent-ils pour gagner des clients ?", "pas du tout", "forte concurrence"),
]

# Questions posées dans l'interface pour chaque force, qui s'enchaînent selon les réponses
# (demande de M. Mass, réunion du 16/11). Chaque chemin de réponses aboutit à une note de 1 à 5
# (5 = favorable au négociateur) : le reste de l'outil utilise cette note, comme avant.
# Format : {"question": texte, "reponses": [(réponse, note 1-5  OU  autre question au même format)]}
# Textes et notes = PROPOSITION À VALIDER avec M. Mass : on peut les changer librement ici
# (l'historique enregistre les notes, et le chemin de réponses pour information).          [IMMÉDIAT]
QUESTIONS_FORCES = {
    "force_fournisseur": {
        "question": "Pouvez-vous acheter ce produit auprès d'un autre fournisseur ?",
        "reponses": [
            ("Oui", {"question": "Combien de fournisseurs pourraient vous livrer la même chose ?",
                     "reponses": [("Un seul autre", 3), ("Deux ou trois", 4), ("Beaucoup", 5)]}),
            ("Non", {"question": "Pourquoi ?",
                     "reponses": [("Produit spécifique ou breveté", 1),
                                  ("Contrat ou homologation qui nous lie", 2)]}),
        ]},
    "force_acheteur": {
        "question": "Êtes-vous un client important pour ce fournisseur ?",
        "reponses": [
            ("Oui", {"question": "Avez-vous des volumes supplémentaires à mettre dans la balance ?",
                     "reponses": [("Oui", 5), ("Non", 4)]}),
            ("Non", {"question": "Êtes-vous un client de référence pour lui (image, accès à un marché) ?",
                     "reponses": [("Oui", 3), ("Un peu", 2), ("Non", 1)]}),
        ]},
    "force_substituts": {
        "question": "Pourriez-vous remplacer ce produit par un autre (autre technologie, autre matériau, "
                    "faire en interne) ?",
        "reponses": [
            ("Oui", {"question": "Le changement serait-il coûteux ou long ?",
                     "reponses": [("Non, il est facile", 5), ("Oui, mais faisable", 4), ("Très coûteux", 3)]}),
            ("À étudier", 2),
            ("Non", 1),
        ]},
    "force_entrants": {
        "question": "De nouveaux fournisseurs apparaissent-ils sur ce marché ?",
        "reponses": [
            ("Oui", {"question": "Les avez-vous déjà consultés ?",
                     "reponses": [("Oui, nous avons une offre", 5), ("Oui, sans offre pour l'instant", 4),
                                  ("Non", 3)]}),
            ("Non", {"question": "Un nouveau fournisseur pourrait-il s'installer facilement ?",
                     "reponses": [("Oui", 3), ("Difficilement", 2), ("Non", 1)]}),
        ]},
    "force_rivalite": {
        "question": "Achetez-vous ce produit à des périodes définies ?",
        "reponses": [
            ("Oui", {"question": "À cette période, qui a l'avantage ?",
                     "reponses": [("Nous (les fournisseurs cherchent des commandes)", 5),
                                  ("Personne en particulier", 3),
                                  ("Le fournisseur (forte demande, stocks bas)", 1)]}),
            ("Non", {"question": "Les fournisseurs se battent-ils pour gagner des clients ?",
                     "reponses": [("Forte concurrence", 5), ("Un peu", 3), ("Pas du tout", 1)]}),
        ]},
}

RAISONS = ["fin de contrat", "renouvellement", "nouveau besoin", "hausse de prix du fournisseur",
           "saisonnalité", "autre"]

# Secteur de l'entreprise qui négocie.
SECTEURS = ["Grande distribution", "Industrie agroalimentaire", "Industrie manufacturière",
            "Transport et logistique", "BTP et construction", "Énergie", "Santé et pharmacie", "Services", "Autre"]

# Catégorie d'achat (ce qui est négocié) et ses sous-catégories.
CATEGORIES_ACHAT = {
    "Emballage": ["Verre", "Plastique", "Carton", "Métal"],
    "Matières premières": ["Métaux", "Produits chimiques", "Produits agricoles", "Énergie"],
    "Transport et logistique": ["Routier", "Maritime", "Aérien", "Entreposage"],
    "Services informatiques": ["Logiciels et licences", "Infogérance", "Matériel"],
    "Maintenance industrielle": ["Pièces détachées", "Interventions sur site", "Contrats de maintenance"],
    "Autre": [],
}
