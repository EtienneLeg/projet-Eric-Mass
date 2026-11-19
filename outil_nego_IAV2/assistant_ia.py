"""
Assistant IA local (Ollama) pour l'outil de négociation.

Rôle de l'IA (et rien de plus) :
  1. interpreter_contexte : à partir d'une description libre de la situation, proposer une note
     de 1 à 5 pour chaque force (avec une justification), la raison de la négo, le secteur,
     la catégorie d'achat et la sous-catégorie.
     Le négociateur VALIDE ces propositions avant qu'elles servent à l'estimation.
  2. expliquer_estimation : rédiger une explication en français de la préparation calculée par algo.py
     (jusqu'où aller dans la liste d'outputs, risques de conflit, contreparties habituelles).
     Un contrôle signale tout chiffre cité par l'IA qui ne figure pas dans les résultats de l'outil.

L'estimation elle-même reste calculée par algo.py : l'IA ne produit aucun chiffre de résultat.
Tout tourne en local : les données ne quittent pas l'ordinateur.

Prérequis (une seule fois) :
  1. installer Ollama : https://ollama.com/download
  2. télécharger un modèle, par exemple :  ollama pull qwen2.5:3b
Ollama doit être lancé (l'application tourne en arrière-plan) pendant l'utilisation.

Essai en ligne de commande :  python3 assistant_ia.py
"""
import json
import re
import urllib.error
import urllib.request

import algo
from reglages import MODELE_OLLAMA as MODELE, URL_OLLAMA, DELAI_OLLAMA_SECONDES as DELAI_SECONDES  # noqa: F401


class OllamaIndisponible(Exception):
    """Ollama n'est pas lancé, pas installé, ou le modèle n'est pas téléchargé."""


# ---------------------------------------------------------------------------
# Accès à Ollama (bibliothèque standard uniquement : rien à installer en plus)
# ---------------------------------------------------------------------------
def _requete(chemin, donnees=None, delai=DELAI_SECONDES):
    url = URL_OLLAMA + chemin
    corps = None if donnees is None else json.dumps(donnees).encode("utf-8")
    req = urllib.request.Request(url, data=corps, headers={"Content-Type": "application/json"},
                                 method="GET" if corps is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=delai) as rep:
            return json.loads(rep.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            message = json.loads(e.read().decode("utf-8")).get("error", str(e))
        except Exception:                      # noqa: BLE001
            message = str(e)
        raise OllamaIndisponible(f"Ollama a répondu une erreur : {message}") from None
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise OllamaIndisponible(
            "Impossible de joindre Ollama. Vérifie qu'il est installé et lancé "
            f"(https://ollama.com/download), puis réessaie. Détail : {e}") from None


def verifier_ollama(modele=None):
    """Renvoie (True, message) si Ollama tourne et que le modèle est téléchargé, sinon (False, message)."""
    modele = modele or MODELE
    try:
        tags = _requete("/api/tags", delai=5)
    except OllamaIndisponible as e:
        return False, str(e)
    noms = [m.get("name", "") for m in tags.get("models", [])]
    if modele in noms or f"{modele}:latest" in noms:
        return True, f"Ollama est prêt (modèle {modele})."
    dispo = ", ".join(noms) if noms else "aucun"
    return False, (f"Le modèle {modele} n'est pas téléchargé. Dans un terminal : ollama pull {modele}  "
                   f"(modèles présents : {dispo})")


def _chat(messages, schema=None, modele=None):
    donnees = {"model": modele or MODELE, "messages": messages, "stream": False,
               "options": {"temperature": 0}}
    if schema is not None:
        donnees["format"] = schema
    rep = _requete("/api/chat", donnees)
    contenu = (rep.get("message") or {}).get("content")
    if not isinstance(contenu, str) or not contenu.strip():
        raise OllamaIndisponible("Réponse vide ou inattendue d'Ollama.")
    return contenu


# ---------------------------------------------------------------------------
# 1. Description libre -> notes des forces (propositions à valider)
# ---------------------------------------------------------------------------
INCONNUE = "inconnue"


def _schema_contexte():
    force = {"type": "object",
             "properties": {"note": {"type": "integer"}, "justification": {"type": "string"}},
             "required": ["note", "justification"]}
    props = {col: force for col in algo.COLONNES_FORCES}
    props["raison"] = {"type": "string", "enum": algo.RAISONS + [INCONNUE]}
    props["secteur"] = {"type": "string", "enum": algo.SECTEURS + [INCONNUE]}
    props["categorie_achat"] = {"type": "string", "enum": list(algo.CATEGORIES_ACHAT) + [INCONNUE]}
    sous = sorted({s for liste in algo.CATEGORIES_ACHAT.values() for s in liste})
    props["sous_categorie"] = {"type": "string", "enum": sous + [INCONNUE]}
    champs = ["raison", "secteur", "categorie_achat", "sous_categorie"]
    return {"type": "object", "properties": props, "required": algo.COLONNES_FORCES + champs}


def _consigne_contexte():
    lignes = ["Tu aides un acheteur à préparer une négociation avec un fournisseur.",
              "À partir de sa description, évalue chaque force ci-dessous par une note entière de 1 à 5,",
              "où 5 = situation FAVORABLE à l'acheteur et 1 = situation défavorable.",
              "RÈGLE ABSOLUE : si la description ne donne aucune information sur une force, mets la note 0 "
              "(inconnue). N'invente rien et ne devine pas.",
              "Pour chaque note, écris une justification courte en français qui cite ce que dit la description.",
              "", "Forces :"]
    for col, nom, question, bas, haut in algo.FORCES:
        lignes.append(f'- "{col}" ({nom}) : {question} 1 = {bas}, 5 = {haut}.')
    lignes += ["", "Raison de la négociation : une valeur parmi " + ", ".join(f'"{r}"' for r in algo.RAISONS)
               + f', ou "{INCONNUE}" si la description ne la donne pas.',
               "Secteur de l'entreprise de l'acheteur : une valeur parmi "
               + ", ".join(f'"{x}"' for x in algo.SECTEURS) + f', ou "{INCONNUE}".',
               "Catégorie d'achat (ce qui est acheté) et sous-catégorie, parmi : "
               + " ; ".join(f'"{c}" ({", ".join(sc) or "pas de sous-catégorie"})'
                            for c, sc in algo.CATEGORIES_ACHAT.items())
               + f'. La sous-catégorie doit appartenir à la catégorie choisie ; sinon "{INCONNUE}".',
               "", "Réponds uniquement en JSON, selon ce schéma : " + json.dumps(_schema_contexte(), ensure_ascii=False)]
    return "\n".join(lignes)


def interpreter_contexte(description, modele=None):
    """
    Renvoie {"forces": {col: note 1-5 ou None}, "justifications": {col: texte},
             "raison", "secteur", "categorie_achat", "sous_categorie": valeur ou None,
             "inconnues": [noms des forces non évaluées]}.
    Toute valeur hors des bornes est traitée comme inconnue (jamais devinée).
    """
    if not description.strip():
        raise ValueError("La description est vide.")
    contenu = _chat([{"role": "system", "content": _consigne_contexte()},
                     {"role": "user", "content": description.strip()}],
                    schema=_schema_contexte(), modele=modele)
    try:
        brut = json.loads(contenu)
    except json.JSONDecodeError:
        raise OllamaIndisponible("L'IA n'a pas renvoyé un JSON valide. Réessaie ou change de modèle.") from None

    forces, justifs, inconnues = {}, {}, []
    for col, nom, *_ in algo.FORCES:
        bloc = brut.get(col) if isinstance(brut.get(col), dict) else {}
        note = bloc.get("note")
        ok = isinstance(note, int) and not isinstance(note, bool) and 1 <= note <= 5
        forces[col] = note if ok else None
        justifs[col] = str(bloc.get("justification", "")).strip()
        if not ok:
            inconnues.append(nom)
    raison = brut.get("raison")
    raison = raison if raison in algo.RAISONS else None
    secteur = brut.get("secteur") if brut.get("secteur") in algo.SECTEURS else None
    categorie = brut.get("categorie_achat") if brut.get("categorie_achat") in algo.CATEGORIES_ACHAT else None
    sous = brut.get("sous_categorie")
    sous = sous if categorie and sous in algo.CATEGORIES_ACHAT[categorie] else None   # cohérence catégorie
    return {"forces": forces, "justifications": justifs, "raison": raison, "secteur": secteur,
            "categorie_achat": categorie, "sous_categorie": sous, "inconnues": inconnues}


# ---------------------------------------------------------------------------
# 2. Explication de l'estimation, avec contrôle des chiffres cités
# ---------------------------------------------------------------------------
def _faits(nego, r):
    """Tous les chiffres que l'IA a le droit de citer, déjà mis en forme."""
    outs = r["outputs"]
    lignes = []
    for o in outs:
        texte = (f"{o['rang']}. {o['texte']} (classé {o['categorie']} par l'utilisateur) : consensus "
                 f"{o['consensus']:.0%}, compromis {o['compromis']:.0%}, conflit {o['conflit']:.0%} ; "
                 f"position : {o['position']}")
        if o["contreparties"]:
            texte += " ; contreparties habituelles : " + algo.texte_contreparties(o)
        if not o["fiable"]:
            texte += " ; peu d'exemples semblables"
        if o.get("avec_fournisseur"):
            texte += f" ; déjà avec {r['fournisseur']} : " + algo.texte_avec_fournisseur(o)
        lignes.append(texte)
    faits = {
        "seuil_de_conflit": f"{r['seuil']:.0%}",
        "outputs_a_pousser": r["ligne"],
        "nombre_outputs": len(outs),
        "dernier_output_a_pousser": outs[r["ligne"] - 1]["texte"] if r["ligne"] > 0 else "aucun",
        "premier_point_bloquant": outs[r["ligne"]]["texte"] if r["ligne"] < len(outs) else "aucun",
        "outputs": lignes,
        "alertes_de_classement": r["alertes"],
    }
    if r.get("fournisseur"):
        faits["historique_avec_ce_fournisseur"] = algo.texte_negos_fournisseur(r)
    if r.get("facteurs"):
        faits["effet_total_du_contexte_sur_le_risque_de_conflit"] = f"{r['effet_contexte'] * 100:+.1f} pts"
        faits["contributions_du_contexte_au_risque_de_conflit"] = [
            f"{f['points'] * 100:+.1f} pts : {f['element']} ({f['valeur']})" for f in r["facteurs"]]
        noms_forces = {nom for _, nom, *_ in algo.FORCES}       # seules les forces sont des leviers d'action
        hausses = [f for f in r["facteurs"] if f["points"] > 0 and f["element"] in noms_forces]
        if hausses:
            pire = max(hausses, key=lambda f: f["points"])
            faits["piste_de_travail"] = (f"{pire['element']} ({pire['valeur']}) augmente le plus le risque de "
                                         f"conflit ({pire['points'] * 100:+.1f} pts)")
    return faits


def _consigne_explication():
    return ("Tu expliques à un négociateur, en français, la préparation calculée par un outil statistique. "
            "Le négociateur a classé ses outputs recherchés par ordre de priorité ; l'outil indique pour chacun "
            "la probabilité de consensus (obtenu sans contrepartie), de compromis (obtenu avec une contrepartie) "
            "et de conflit (pas obtenu), et jusqu'où aller dans la liste avant que le risque de conflit dépasse "
            "le seuil.\n"
            "Règles :\n"
            "- 4 à 6 phrases, ton direct et professionnel, sans titre ni liste.\n"
            "- Dis jusqu'où aller dans la liste et quel output est le premier point bloquant probable.\n"
            "- Pour les outputs à pousser qui passent souvent par un compromis, cite la contrepartie habituelle.\n"
            "- Si un historique avec ce fournisseur est fourni, dis ce qui avait bloqué ou été obtenu avec lui.\n"
            "- Utilise UNIQUEMENT les chiffres fournis dans les faits. N'invente, ne calcule et n'arrondis "
            "aucun autre chiffre. Il n'y a pas de résultat global : n'en donne pas.\n"
            "- Dis clairement qu'il s'agit d'estimations fondées sur les négociations passées comparables, "
            "pas de certitudes.\n"
            "- S'il y a des alertes de classement, mentionne-les.\n"
            "- Si une piste de travail est fournie, termine par elle ; sinon, n'en propose pas.")


_NOMBRE = re.compile(r"([+-]?\d+(?:[.,]\d+)?)\s*(%|pts\b|points?\b)", re.IGNORECASE)


def chiffres_non_verifies(texte, faits):
    """Liste des chiffres (en % ou en points) cités dans le texte qui ne figurent pas dans les faits."""
    autorises = set()
    for m in _NOMBRE.finditer(json.dumps(faits, ensure_ascii=False)):
        autorises.add(abs(float(m.group(1).replace(",", "."))))
    suspects = []
    for m in _NOMBRE.finditer(texte):
        v = abs(float(m.group(1).replace(",", ".")))
        if not any(abs(v - a) <= 0.051 for a in autorises):
            suspects.append(m.group(0))
    return suspects


def expliquer_estimation(nego, r, modele=None):
    """Renvoie (explication, chiffres_suspects)."""
    faits = _faits(nego, r)
    texte = _chat([{"role": "system", "content": _consigne_explication()},
                   {"role": "user", "content": "Faits (JSON) :\n" + json.dumps(faits, ensure_ascii=False, indent=1)}],
                  modele=modele).strip()
    return texte, chiffres_non_verifies(texte, faits)


# ---------------------------------------------------------------------------
# Essai en ligne de commande
# ---------------------------------------------------------------------------
def main():
    ok, msg = verifier_ollama()
    print(msg)
    if not ok:
        return
    modele = algo.charger_modele()
    print("\nDécris la situation de ta prochaine négo en quelques phrases (fournisseur, alternatives, enjeux…) :")
    description = input("> ")
    print("\nAnalyse par l'IA…")
    prop = interpreter_contexte(description)
    for col, nom, *_ in algo.FORCES:
        note = prop["forces"][col]
        print(f"  {nom:<30} {note if note else '?'}/5  — {prop['justifications'][col]}")
    print(f"  Raison : {prop['raison'] or '?'} · Secteur : {prop['secteur'] or '?'} · "
          f"Catégorie : {prop['categorie_achat'] or '?'} / {prop['sous_categorie'] or '?'}")
    if prop["inconnues"]:
        print("  Forces non évaluées (note 3 par défaut, à corriger) : " + ", ".join(prop["inconnues"]))
    forces = {c: (v or 3) for c, v in prop["forces"].items()}

    print("\nTes outputs recherchés, par ordre de priorité, un par ligne au format  catégorie ; output  "
          "(ligne vide pour terminer) :")
    args = []
    while True:
        ligne = input("> ").strip()
        if not ligne:
            break
        cat, _, texte = ligne.partition(";")
        cat = cat.strip().lower()
        if cat not in algo.CATEGORIES or not texte.strip():
            print("  format attendu : compromis ; baisse de prix de 4 %")
            continue
        args.append((texte.strip(), cat))
    if not args:
        return
    nego = {"id": "NOUVELLE", "arguments": args, "resultat": None, "forces": forces, "raison": prop["raison"],
            "secteur": prop["secteur"], "categorie_achat": prop["categorie_achat"],
            "sous_categorie": prop["sous_categorie"]}
    r = algo.estimer_outputs(modele, nego)
    algo.afficher_estimation(r)
    print("\nExplication par l'IA…")
    texte, suspects = expliquer_estimation(nego, r)
    print(texte)
    if suspects:
        print("\n⚠ Chiffres cités par l'IA absents des résultats de l'outil : " + ", ".join(suspects))


if __name__ == "__main__":
    main()
