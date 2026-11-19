"""
Interface graphique simple (fenêtre Tkinter) pour essayer algo.py.

Lancement : python3 interface.py      (ou bouton « Run » de VS Code)
Rien à installer en plus : Tkinter est fourni avec Python.
algo.py et historique.csv doivent être dans le même dossier que ce fichier.
"""
import os
import sys
import threading
from datetime import date
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

DOSSIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DOSSIER)
try:
    import algo  # noqa: E402
except ValueError as erreur_reglages:      # valeur incorrecte dans reglages.py : on l'explique et on s'arrête
    _r = tk.Tk(); _r.withdraw()
    messagebox.showerror("Réglages incorrects", str(erreur_reglages))
    sys.exit(str(erreur_reglages))
try:
    import assistant_ia  # noqa: E402  (v3 : assistant IA local via Ollama)
except ImportError:
    assistant_ia = None

CSV_HISTORIQUE = os.path.join(DOSSIER, "historique.csv")
FICHIER_MODELE = os.path.join(DOSSIER, "modele_nego.pkl")
CATS = algo.CATEGORIES
SYMBOLE = {"consensus": "●", "compromis": "◐", "conflit": "○"}
COULEUR = {"consensus": "#2e7d32", "compromis": "#ef6c00", "conflit": "#c62828"}


def pct(x):
    return f"{x:.0%}"


def resume_arguments(args, contreparties=None):
    contreparties = contreparties or [None] * len(args)
    return " · ".join(f"[{c}] {a}" + (f" (contre : {cp})" if cp else "") for (a, c), cp in zip(args, contreparties))


AUCUNE = "— aucune —"


# ---------------------------------------------------------------------------
# Éditeur d'arguments réutilisable (onglets Estimer et Ajouter)
# ---------------------------------------------------------------------------
class EditeurArguments(ttk.Frame):
    """Liste ordonnée d'outputs recherchés, par ordre de priorité de l'utilisateur (le premier en haut).
    avec_contreparties=True (négo terminée) : on peut noter la contrepartie concédée pour chaque output."""

    def __init__(self, parent, on_change=None, avec_contreparties=False):
        super().__init__(parent)
        self.on_change = on_change
        self.avec_contreparties = avec_contreparties

        saisie = ttk.Frame(self)
        saisie.pack(fill="x", pady=(0, 4))
        ttk.Label(saisie, text="Output recherché :").pack(side="left")
        self.var_texte = tk.StringVar()
        champ = ttk.Entry(saisie, textvariable=self.var_texte, width=50)
        champ.pack(side="left", fill="x", expand=True, padx=4)
        champ.bind("<Return>", lambda e: self.ajouter())
        self.var_cat = tk.StringVar(value="compromis")
        ttk.Combobox(saisie, textvariable=self.var_cat, values=CATS, state="readonly", width=11).pack(side="left")
        ttk.Button(saisie, text="Ajouter", command=self.ajouter).pack(side="left", padx=4)

        if avec_contreparties:
            ligne_cp = ttk.Frame(self)
            ligne_cp.pack(fill="x", pady=(0, 4))
            ttk.Label(ligne_cp, text="Contrepartie concédée pour l'output sélectionné :").pack(side="left")
            self.var_cp = tk.StringVar(value=AUCUNE)
            ttk.Combobox(ligne_cp, textvariable=self.var_cp, values=[AUCUNE] + algo.CONTREPARTIES,
                         state="readonly", width=30).pack(side="left", padx=4)
            ttk.Button(ligne_cp, text="Appliquer", command=self.definir_contrepartie).pack(side="left")

        corps = ttk.Frame(self)
        corps.pack(fill="both", expand=True)
        colonnes = [("rang", "N°", 40), ("categorie", "Catégorie", 100), ("argument", "Output recherché", 380)]
        if avec_contreparties:
            colonnes.append(("contrepartie", "Contrepartie", 200))
        self.liste = ttk.Treeview(corps, columns=[c for c, _, _ in colonnes], show="headings", height=7,
                                  selectmode="browse")
        for col, titre, larg in colonnes:
            self.liste.heading(col, text=titre)
            self.liste.column(col, width=larg, stretch=(col == "argument"), anchor="w")
        if avec_contreparties:
            self.liste.bind("<<TreeviewSelect>>", lambda e: self.maj_combo_contrepartie())
        for c in CATS:
            self.liste.tag_configure(c, foreground=COULEUR[c])
        self.liste.pack(side="left", fill="both", expand=True)

        boutons = ttk.Frame(corps)
        boutons.pack(side="left", fill="y", padx=(6, 0))
        for texte, cmd in [("▲ Monter", lambda: self.deplacer(-1)), ("▼ Descendre", lambda: self.deplacer(1)),
                           ("Changer catégorie", self.changer_categorie), ("Supprimer", self.supprimer),
                           ("Tout effacer", self.effacer)]:
            ttk.Button(boutons, text=texte, command=cmd).pack(fill="x", pady=2)

        self.arguments = []      # [(texte, categorie)] dans l'ordre de priorité
        self.contreparties = []  # contrepartie de chaque output (None = aucune), même ordre

    def ajouter(self):
        texte = self.var_texte.get().strip()
        if not texte:
            return
        self.arguments.append((texte, self.var_cat.get()))
        self.contreparties.append(None)
        self.var_texte.set("")
        self.rafraichir(len(self.arguments) - 1)

    def definir_contrepartie(self):
        i = self.index_selection()
        if i is None:
            messagebox.showinfo("Contrepartie", "Sélectionne d'abord un output dans la liste.")
            return
        cp = self.var_cp.get()
        self.contreparties[i] = None if cp == AUCUNE else cp
        self.rafraichir(i)

    def maj_combo_contrepartie(self):
        i = self.index_selection()
        if i is not None:
            self.var_cp.set(self.contreparties[i] or AUCUNE)

    def index_selection(self):
        sel = self.liste.selection()
        return self.liste.index(sel[0]) if sel else None

    def deplacer(self, sens):
        i = self.index_selection()
        if i is None or not 0 <= i + sens < len(self.arguments):
            return
        for a in (self.arguments, self.contreparties):
            a[i], a[i + sens] = a[i + sens], a[i]
        self.rafraichir(i + sens)

    def changer_categorie(self):
        i = self.index_selection()
        if i is None:
            return
        texte, cat = self.arguments[i]
        self.arguments[i] = (texte, CATS[(CATS.index(cat) + 1) % len(CATS)])
        self.rafraichir(i)

    def supprimer(self):
        i = self.index_selection()
        if i is not None:
            del self.arguments[i]
            del self.contreparties[i]
            self.rafraichir(min(i, len(self.arguments) - 1))

    def effacer(self):
        self.definir([])

    def definir(self, arguments, contreparties=None):
        self.arguments = list(arguments)
        self.contreparties = list(contreparties) if contreparties else [None] * len(self.arguments)
        self.rafraichir()

    def rafraichir(self, selection=None):
        self.liste.delete(*self.liste.get_children())
        for i, ((texte, cat), cp) in enumerate(zip(self.arguments, self.contreparties), 1):
            valeurs = [i, f"{SYMBOLE[cat]} {cat}", texte] + ([cp or ""] if self.avec_contreparties else [])
            self.liste.insert("", "end", values=valeurs, tags=(cat,))
        if selection is not None and 0 <= selection < len(self.arguments):
            item = self.liste.get_children()[selection]
            self.liste.selection_set(item)
            self.liste.see(item)
        if self.on_change:
            self.on_change()


# ---------------------------------------------------------------------------
# Saisie du contexte : questions qui s'enchaînent pour chaque force + raison de la négo
# ---------------------------------------------------------------------------
class ContexteForces(ttk.LabelFrame):
    """Rapport de force : pour chaque force de algo.FORCES, des questions qui s'enchaînent selon les
    réponses (QUESTIONS_FORCES dans reglages.py). Chaque chemin complet donne une note de 1 à 5
    (5 = favorable pour vous), utilisée par l'algorithme comme avant."""

    def __init__(self, parent, largeur_texte=360):
        super().__init__(parent, text="Rapport de force", padding=8)
        self.largeur = largeur_texte
        self.var_actif = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Renseigner le rapport de force", variable=self.var_actif,
                        command=self.maj_actif).pack(anchor="w")
        self.chemins = {col: [] for col in algo.COLONNES_FORCES}     # réponses choisies, dans l'ordre
        self.fixees = {col: None for col in algo.COLONNES_FORCES}    # note reprise sans les réponses
        self.cadres = {}
        for col, nom, *_ in algo.FORCES:
            cadre = ttk.Frame(self)
            cadre.pack(fill="x", pady=(8, 0))
            entete = ttk.Frame(cadre)
            entete.pack(fill="x")
            ttk.Label(entete, text=nom, font=("TkDefaultFont", 10, "bold")).pack(side="left")
            var_note = tk.StringVar()
            ttk.Label(entete, textvariable=var_note, foreground="#555").pack(side="right")
            corps = ttk.Frame(cadre)
            corps.pack(fill="x")
            self.cadres[col] = (corps, var_note)
        bas = ttk.Frame(self)
        bas.pack(fill="x", pady=(10, 0))
        ttk.Label(bas, text="Raison de la négo", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.var_raison = tk.StringVar(value=algo.RAISONS[0])
        self.cb_raison = ttk.Combobox(bas, textvariable=self.var_raison, values=algo.RAISONS, state="readonly",
                                      width=26)
        self.cb_raison.pack(side="right")
        self.maj_actif()

    # ----- affichage -----
    def afficher(self, col):
        corps, var_note = self.cadres[col]
        for w in corps.winfo_children():
            w.destroy()
        etat = "readonly" if self.var_actif.get() else "disabled"
        if self.fixees[col] is not None and not self.chemins[col]:
            ttk.Label(corps, text=f"Note reprise telle quelle : {self.fixees[col]}/5 (assistant IA ou négo "
                                  "déjà enregistrée).", foreground="#555", wraplength=self.largeur).pack(anchor="w")
            ttk.Button(corps, text="Répondre aux questions", command=lambda c=col: self.repondre(c),
                       state="normal" if self.var_actif.get() else "disabled").pack(anchor="w")
            var_note.set(f"note {self.fixees[col]}/5")
            return
        posees, note = algo.suivre_questions(col, self.chemins[col])
        for rang_q, (question, reponses, choix) in enumerate(posees):
            ttk.Label(corps, text=question, wraplength=self.largeur).pack(anchor="w", pady=(2, 0))
            var = tk.StringVar(value=choix or "")
            cb = ttk.Combobox(corps, textvariable=var, values=reponses, state=etat, width=42)
            cb.pack(anchor="w")
            cb.bind("<<ComboboxSelected>>", lambda e, c=col, n=rang_q, v=var: self.choisir(c, n, v.get()))
        var_note.set(f"note {note}/5" if note else "à compléter")

    def choisir(self, col, rang_q, reponse):
        self.chemins[col] = self.chemins[col][:rang_q] + [reponse]
        self.fixees[col] = None
        self.after_idle(lambda: self.afficher(col))      # la liste cliquée est détruite : après l'événement

    def repondre(self, col):
        self.fixees[col] = None
        self.chemins[col] = []
        self.afficher(col)

    def maj_actif(self):
        for col in algo.COLONNES_FORCES:
            self.afficher(col)
        self.cb_raison.configure(state="readonly" if self.var_actif.get() else "disabled")

    # ----- lecture / écriture -----
    def note(self, col):
        _, note = algo.suivre_questions(col, self.chemins[col])
        if note is None and not self.chemins[col]:
            note = self.fixees[col]
        return note

    def valeurs(self):
        """(forces, raison) ou (None, None) si le contexte n'est pas renseigné.
        Une force dont les questions ne sont pas complétées compte 3/5 (neutre) : voir manquantes()."""
        if not self.var_actif.get():
            return None, None
        return {col: (self.note(col) or 3) for col in algo.COLONNES_FORCES}, self.var_raison.get()

    def manquantes(self):
        if not self.var_actif.get():
            return []
        return [nom for col, nom, *_ in algo.FORCES if self.note(col) is None]

    def reponses(self):
        """Réponses choisies, pour les garder dans l'historique (colonnes reponses_<force>)."""
        if not self.var_actif.get():
            return {}
        return {f"reponses_{col}": " > ".join(ch) for col, ch in self.chemins.items() if ch}

    def definir(self, forces, raison, reponses=None):
        """Reprend un contexte : les réponses si elles mènent à la même note, sinon la note seule."""
        reponses = reponses or {}
        self.var_actif.set(bool(forces))
        for col in algo.COLONNES_FORCES:
            texte = reponses.get(f"reponses_{col}") or ""
            chemin = texte.split(" > ") if texte else []
            if forces and chemin and algo.suivre_questions(col, chemin)[1] == forces[col]:
                self.chemins[col], self.fixees[col] = chemin, None
            else:
                self.chemins[col], self.fixees[col] = [], (int(forces[col]) if forces else None)
        if raison:
            self.var_raison.set(raison)
        self.maj_actif()


AUCUN = "—"


class ContexteSituation(ttk.Frame):
    """Fournisseur, période (trimestre), secteur de l'entreprise, catégorie d'achat et sous-catégorie
    (liste dépendante de la catégorie). vertical=True : une ligne par élément (colonne de gauche)."""

    def __init__(self, parent, vertical=False):
        super().__init__(parent)
        self.var_fournisseur = tk.StringVar()
        self.var_trimestre = tk.StringVar(value=AUCUN)
        self.var_secteur = tk.StringVar(value=AUCUN)
        self.var_categorie = tk.StringVar(value=AUCUN)
        self.var_sous = tk.StringVar(value=AUCUN)
        elements = (("Fournisseur", self.var_fournisseur, []),
                    ("Période", self.var_trimestre, [AUCUN] + algo.TRIMESTRES),
                    ("Secteur", self.var_secteur, [AUCUN] + algo.SECTEURS),
                    ("Catégorie d'achat", self.var_categorie, [AUCUN] + list(algo.CATEGORIES_ACHAT)),
                    ("Sous-catégorie", self.var_sous, [AUCUN]))
        if vertical:
            ttk.Label(self, text="Situation", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")
        else:
            ttk.Label(self, text="Situation :", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        for k, (texte, var, valeurs) in enumerate(elements, start=1):
            # fournisseur : saisie libre, avec la liste des fournisseurs déjà rencontrés
            etat = "normal" if var is self.var_fournisseur else "readonly"
            cb = ttk.Combobox(self, textvariable=var, values=valeurs, state=etat, width=26)
            if vertical:
                ttk.Label(self, text=texte).grid(row=k, column=0, sticky="w", pady=1)
                cb.grid(row=k, column=1, sticky="w", padx=(8, 0), pady=1)
            else:
                ttk.Label(self, text=texte).pack(side="left", padx=(12, 4))
                cb.pack(side="left")
            if var is self.var_categorie:
                cb.bind("<<ComboboxSelected>>", lambda e: self.maj_sous_categories())
            if var is self.var_sous:
                self.cb_sous = cb
            if var is self.var_fournisseur:
                self.cb_fournisseur = cb

    def definir_fournisseurs(self, noms):
        self.cb_fournisseur.configure(values=list(noms))

    def maj_sous_categories(self):
        sous = algo.CATEGORIES_ACHAT.get(self.var_categorie.get(), [])
        self.cb_sous.configure(values=[AUCUN] + sous)
        if self.var_sous.get() not in sous:
            self.var_sous.set(AUCUN)

    def valeurs(self):
        val = lambda v: None if v.get() == AUCUN else v.get()
        fournisseur = self.var_fournisseur.get().strip() or None
        if fournisseur:                       # « emb-01 » -> « EMB-01 » s'il est déjà connu
            fournisseur = next((f for f in self.cb_fournisseur.cget("values") if algo.meme_fournisseur(f, fournisseur)),
                               fournisseur)
        return {"fournisseur": fournisseur, "trimestre": val(self.var_trimestre),
                "secteur": val(self.var_secteur), "categorie_achat": val(self.var_categorie),
                "sous_categorie": val(self.var_sous)}

    def definir(self, situation):
        situation = situation or {}
        self.var_fournisseur.set(situation.get("fournisseur") or "")
        self.var_trimestre.set(situation.get("trimestre") or AUCUN)
        self.var_secteur.set(situation.get("secteur") or AUCUN)
        self.var_categorie.set(situation.get("categorie_achat") or AUCUN)
        self.maj_sous_categories()
        self.var_sous.set(situation.get("sous_categorie") or AUCUN)


def situation_de(nego):
    return {cle: nego.get(cle) for cle in ("fournisseur", "trimestre", "secteur", "categorie_achat", "sous_categorie")}


def texte_situation(nego):
    cat = nego.get("categorie_achat")
    if cat and nego.get("sous_categorie"):
        cat = f"{cat} / {nego['sous_categorie']}"
    morceaux = [m for m in (nego.get("fournisseur"), nego.get("trimestre"), nego.get("secteur"), cat) if m]
    return " · ".join(morceaux) if morceaux else "—"


def texte_contexte(nego):
    if not nego.get("forces"):
        return "—"
    notes = "-".join(str(nego["forces"][c]) for c in algo.COLONNES_FORCES)
    return f"{notes}  ({nego.get('raison') or '?'})"


class ColonneDefilante(ttk.Frame):
    """Cadre avec barre de défilement verticale (pour la colonne du contexte, qui peut être haute).
    Mettre les éléments dans self.interieur."""

    def __init__(self, parent, largeur=420):
        super().__init__(parent)
        self.canevas = tk.Canvas(self, width=largeur, highlightthickness=0)
        barre = ttk.Scrollbar(self, orient="vertical", command=self.canevas.yview)
        self.interieur = ttk.Frame(self.canevas)
        self.interieur.bind("<Configure>", lambda e: self.canevas.configure(scrollregion=self.canevas.bbox("all")))
        fenetre = self.canevas.create_window((0, 0), window=self.interieur, anchor="nw")
        # le contenu prend toute la largeur de la colonne (rien n'est coupé à droite)
        self.canevas.bind("<Configure>", lambda e: self.canevas.itemconfigure(fenetre, width=e.width))
        self.canevas.configure(yscrollcommand=barre.set)
        self.canevas.pack(side="left", fill="both", expand=True)
        barre.pack(side="left", fill="y")
        # molette : seulement quand la souris est au-dessus de la colonne
        self.canevas.bind("<Enter>", lambda e: self.canevas.bind_all("<MouseWheel>", self._molette))
        self.canevas.bind("<Leave>", lambda e: self.canevas.unbind_all("<MouseWheel>"))

    def _molette(self, e):
        self.canevas.yview_scroll(int(-e.delta / 120) if abs(e.delta) >= 120 else -int(e.delta), "units")


def colonne_contexte(parent):
    """Colonne de gauche des onglets Estimer / Ajouter : situation + questions du rapport de force."""
    col = ColonneDefilante(parent)
    col.pack(side="left", fill="y", padx=(0, 10))
    situation = ContexteSituation(col.interieur, vertical=True)
    situation.pack(fill="x", pady=(0, 8))
    forces = ContexteForces(col.interieur)
    forces.pack(fill="x")
    return situation, forces


# Exemple pré-rempli dans l'onglet Estimer (mêmes notes que l'exemple de la v3 : 4-3-2-3-3)
EXEMPLE_REPONSES = {"reponses_force_fournisseur": "Oui > Deux ou trois",
                    "reponses_force_acheteur": "Non > Oui",
                    "reponses_force_substituts": "À étudier",
                    "reponses_force_entrants": "Oui > Non",
                    "reponses_force_rivalite": "Oui > Personne en particulier"}
EXEMPLE_FORCES = {"force_fournisseur": 4, "force_acheteur": 3, "force_substituts": 2,
                  "force_entrants": 3, "force_rivalite": 3}
POSITION_TAG = {"à pousser": "pousser", "point bloquant probable": "bloquant",
                "risque élevé": "eleve", "faible risque, après le blocage": "apres"}


# ---------------------------------------------------------------------------
# Fenêtre principale
# ---------------------------------------------------------------------------
class Application(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=8)
        self.root = root
        self.modele = None
        self.derniere = None          # (nego, résultat) de la dernière estimation
        self.occupe = False
        root.title("Fantasy de la négociation")
        root.geometry("1440x920")
        root.minsize(1200, 780)
        self.pack(fill="both", expand=True)

        haut = ttk.Frame(self)
        haut.pack(fill="x", pady=(0, 6))
        self.var_etat = tk.StringVar(value="Chargement…")
        ttk.Label(haut, textvariable=self.var_etat).pack(side="left")
        ttk.Button(haut, text="Changer d'historique…", command=self.changer_historique).pack(side="right")
        ttk.Button(haut, text="Reconstruire l'arbre", command=self.reconstruire).pack(side="right", padx=4)

        self.onglets = ttk.Notebook(self)
        self.onglets.pack(fill="both", expand=True)
        self.construire_onglet_estimer()
        self.construire_onglet_ajouter()
        self.construire_onglet_historique()
        self.construire_onglet_arbre()
        self.construire_onglet_evaluer()
        if assistant_ia is not None:
            self.construire_onglet_assistant()

        self.root.after(100, self.charger_modele)

    # ----- calculs longs en arrière-plan (la fenêtre reste réactive) -----
    def en_fond(self, message, calcul, suite):
        if self.occupe:
            return
        self.occupe = True
        self.var_etat.set(message)
        self.root.config(cursor="watch")
        resultat = {}

        def travail():
            try:
                resultat["ok"] = calcul()
            except Exception as e:          # noqa: BLE001 — on affiche toute erreur à l'utilisateur
                resultat["erreur"] = e

        fil = threading.Thread(target=travail, daemon=True)
        fil.start()

        def attendre():
            if fil.is_alive():
                self.root.after(150, attendre)
                return
            self.occupe = False
            self.root.config(cursor="")
            if "erreur" in resultat:
                self.maj_etat()
                messagebox.showerror("Erreur", str(resultat["erreur"]))
            else:
                suite(resultat["ok"])
                self.maj_etat()

        attendre()

    # ----- modèle -----
    def charger_modele(self):
        def calcul():
            chemin = CSV_HISTORIQUE
            if os.path.exists(FICHIER_MODELE):
                try:
                    return algo.charger_modele(FICHIER_MODELE)
                except Exception:           # noqa: BLE001 — autre version ou réglages modifiés : on reconstruit
                    try:                    # ... à partir du même historique que l'ancien modèle
                        ancien = algo.charger_modele(FICHIER_MODELE, verifier_version=False)
                        if ancien.chemin_csv and os.path.exists(ancien.chemin_csv):
                            chemin = ancien.chemin_csv
                    except Exception:       # noqa: BLE001
                        pass
            if not os.path.exists(chemin):
                raise FileNotFoundError(f"Aucun historique trouvé : {chemin}")
            m = algo.ModeleNegociation(algo.lire_negos_csv(chemin), chemin_csv=chemin)
            m.sauvegarder(FICHIER_MODELE)
            return m
        self.en_fond("Chargement du modèle (calcul des embeddings au premier lancement)…", calcul, self.modele_pret)

    def modele_pret(self, m):
        self.modele = m
        for sit in (self.sit_estimer, self.sit_ajout):
            sit.definir_fournisseurs(m.fournisseurs())
        self.remplir_historique()
        self.remplir_arbre()
        self.proposer_id()

    def maj_etat(self):
        m = self.modele
        if m is None:
            self.var_etat.set("Aucun modèle chargé.")
            return
        nb_exclues = len(getattr(m, "exclues", []))
        txt = (f"{len(m.negos)} négos" + (f" (+ {nb_exclues} exclue(s))" if nb_exclues else "")
               + f"  ·  {self.nb_feuilles(m.arbre)} groupes  ·  "
               f"{m.ajouts_depuis_construction} ajout(s) récent(s)  ·  embeddings : "
               f"{'modèle de phrases' if m.emb.mode == 'sentence-transformers' else 'TF-IDF (repli)'}  ·  "
               f"contexte : {'effets sur le risque de conflit appris sur ' + str(m.effets['n']) + ' négos' if m.effets else 'non utilisé'}")
        if algo.reconstruction_conseillee(m):
            txt += "  ·  ⚠ reconstruction conseillée"
        self.var_etat.set(txt)

    def nb_feuilles(self, n):
        return 1 if n.est_feuille() else sum(self.nb_feuilles(e) for e in n.enfants)

    def reconstruire(self):
        if self.modele is None:
            return
        chemin = self.modele.chemin_csv

        def calcul():
            m = algo.ModeleNegociation(algo.lire_negos_csv(chemin), chemin_csv=chemin)
            m.sauvegarder(FICHIER_MODELE)
            return m
        self.en_fond("Reconstruction de l'arbre…", calcul, self.modele_pret)

    def changer_historique(self):
        chemin = filedialog.askopenfilename(title="Choisir un historique CSV", filetypes=[("CSV", "*.csv")])
        if not chemin:
            return

        def calcul():
            m = algo.ModeleNegociation(algo.lire_negos_csv(chemin), chemin_csv=chemin)
            m.sauvegarder(FICHIER_MODELE)
            return m
        self.en_fond("Construction de l'arbre sur le nouvel historique…", calcul, self.modele_pret)

    # ----- onglet Estimer -----
    def construire_onglet_estimer(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Préparer une négo")
        self.sit_estimer, self.ctx_estimer = colonne_contexte(f)
        self.sit_estimer.definir({"trimestre": algo.trimestre_de(date.today().isoformat()),
                                  "secteur": "Grande distribution", "categorie_achat": "Emballage",
                                  "sous_categorie": "Verre"})
        self.ctx_estimer.definir(EXEMPLE_FORCES, "fin de contrat", EXEMPLE_REPONSES)

        droite = ttk.Frame(f)
        droite.pack(side="left", fill="both", expand=True)
        ttk.Label(droite, text="Outputs recherchés, par ordre de priorité (le plus important en haut), avec la "
                               "catégorie que tu prévois (ta prévision ne change pas le risque calculé) :"
                  ).pack(anchor="w")
        self.ed_estimer = EditeurArguments(droite)
        self.ed_estimer.pack(fill="x", pady=4)
        self.ed_estimer.definir([("planning de livraison partagé", "consensus"),
                                 ("paiement à 60 jours", "compromis"),
                                 ("baisse de prix de 2 %", "compromis"),
                                 ("baisse de prix de 4 %", "compromis"),
                                 ("baisse de prix de 6 %", "compromis"),
                                 ("baisse de prix de 8 %", "conflit"),
                                 ("exclusivité sur la catégorie", "conflit")])
        ligne = ttk.Frame(droite)
        ligne.pack(fill="x", pady=4)
        ttk.Button(ligne, text="Jusqu'où aller ?", command=self.estimer).pack(side="left")
        ttk.Label(ligne, text="Seuil de conflit :").pack(side="left", padx=(16, 4))
        self.var_seuil = tk.StringVar(value=str(int(round(algo.SEUIL_CONFLIT * 100))))
        ttk.Spinbox(ligne, from_=5, to=95, increment=5, textvariable=self.var_seuil, width=5,
                    command=self.changer_seuil).pack(side="left")
        ttk.Label(ligne, text="%   (la ligne se place avant le premier output qui dépasse ce risque)").pack(side="left")

        self.txt_resultat = tk.Text(droite, height=9, wrap="word", relief="flat", background="#f6f6f6",
                                    font="TkDefaultFont")
        self.txt_resultat.pack(fill="x", pady=4)
        self.txt_resultat.tag_configure("alerte", foreground="#b26a00")
        self.txt_resultat.tag_configure("gras", font=("TkDefaultFont", 10, "bold"))
        self.txt_resultat.tag_configure("plus", foreground="#c62828")       # + de risque de conflit
        self.txt_resultat.tag_configure("moins", foreground="#2e7d32")      # - de risque de conflit
        self.txt_resultat.tag_configure("discret", foreground="#777777")
        self.txt_resultat.configure(state="disabled")

        self.tab_outputs = self.tableau(droite, [("n", "N°", 30), ("output", "Output recherché", 185),
                                                 ("cons", "Consensus", 78), ("comp", "Compromis", 82),
                                                 ("conf", "Conflit", 62), ("issue", "Issue probable", 98),
                                                 ("cp", "Contreparties habituelles", 230),
                                                 ("pos", "Position", 165),
                                                 ("fourn", "Déjà avec ce fournisseur", 260)], hauteur=8, etirer="cp")
        self.colonnes_sans_fourn = ("n", "output", "cons", "comp", "conf", "issue", "cp", "pos")
        self.tab_outputs.configure(displaycolumns=self.colonnes_sans_fourn)
        for tag, couleur in (("pousser", "#2e7d32"), ("bloquant", "#c62828"), ("eleve", "#c62828"),
                             ("apres", "#777777"), ("ligne", "#000000")):
            self.tab_outputs.tag_configure(tag, foreground=couleur)
        self.tab_outputs.tag_configure("ligne", background="#e0e0e0")

        ttk.Label(droite, text="Négos passées les plus proches :").pack(anchor="w", pady=(6, 0))
        self.tab_voisins = self.tableau(droite, [("nego", "Négo", 60), ("ress", "Ressemblance", 90),
                                                 ("res", "Part obtenue", 80), ("sit", "Situation", 230),
                                                 ("ctx", "Forces (raison)", 190), ("args", "Outputs", 500)],
                                        hauteur=5)

    def seuil(self):
        try:
            v = float(self.var_seuil.get().replace(",", ".")) / 100
        except ValueError:
            v = algo.SEUIL_CONFLIT
        return min(max(v, 0.0), 1.0)

    def estimer(self):
        if self.modele is None or not self.ed_estimer.arguments:
            return
        forces, raison = self.ctx_estimer.valeurs()
        nego = {"id": "NOUVELLE", "arguments": list(self.ed_estimer.arguments), "resultat": None,
                "forces": forces, "raison": raison, **self.sit_estimer.valeurs(),
                "reponses": self.ctx_estimer.reponses(), "manquantes": self.ctx_estimer.manquantes()}
        seuil = self.seuil()
        self.en_fond("Recherche des outputs comparables…",
                     lambda: algo.estimer_outputs(self.modele, nego, seuil),
                     lambda r: self.afficher_estimation(nego, r))

    def changer_seuil(self):
        """Nouveau seuil : on replace seulement la ligne (pas de nouveau calcul)."""
        if not self.derniere:
            return
        nego, r = self.derniere
        r["seuil"] = self.seuil()
        r["ligne"] = algo.placer_ligne(r["outputs"], r["seuil"])
        self.afficher_estimation(nego, r)

    def afficher_estimation(self, nego, r):
        self.derniere = (nego, r)
        outs, ligne, n = r["outputs"], r["ligne"], len(r["outputs"])
        t = self.txt_resultat
        t.configure(state="normal")
        t.delete("1.0", "end")
        if ligne == n:
            t.insert("end", f"Aucun output ne dépasse {pct(r['seuil'])} de risque de conflit : tous peuvent être "
                            "poussés.\n", "gras")
        elif ligne == 0:
            t.insert("end", f"Dès le premier output, le risque de conflit dépasse {pct(r['seuil'])} : "
                            f"« {outs[0]['texte']} » est probablement bloquant.\n", "gras")
        else:
            t.insert("end", f"Jusqu'ici : {ligne} output(s) sur {n}, jusqu'à « {outs[ligne - 1]['texte']} ». "
                            f"Au-delà, « {outs[ligne]['texte']} » a {outs[ligne]['conflit']:.1%} de risque de "
                            f"conflit (seuil {pct(r['seuil'])}).\n", "gras")
        if not r["contexte_utilise"]:
            t.insert("end", "Contexte non renseigné : estimation à partir des outputs seuls.\n")
        elif not self.modele.effets:
            t.insert("end", "Pas assez de négos avec contexte dans l'historique pour en tenir compte.\n")
        elif r["facteurs"]:
            t.insert("end", f"Effet du contexte sur le risque de conflit de chaque output : "
                            f"{r['effet_contexte'] * 100:+.1f} pts (appris sur l'historique) : ")
            for k, fct in enumerate(r["facteurs"]):
                pts = fct["points"] * 100
                if abs(pts) < 0.05:
                    continue
                t.insert("end", (" · " if k else "") + f"{fct['element']} ({fct['valeur']}) ")
                t.insert("end", f"{pts:+.1f}", "plus" if pts > 0 else "moins")
            t.insert("end", "\n")
        if nego.get("manquantes"):
            t.insert("end", "⚠ Questions non complétées (comptées 3/5, neutre) : " + ", ".join(nego["manquantes"])
                     + "\n", "alerte")
        if r.get("fournisseur"):
            t.insert("end", algo.texte_negos_fournisseur(r) + "\n")
            if any(o.get("avec_fournisseur") for o in outs):
                t.insert("end", "Colonne « Déjà avec » : ce qui s'est passé avec ce fournisseur pour des outputs "
                                "semblables (3 plus récents).\n", "discret")
        e = r["equilibre"]
        t.insert("end", f"Ta liste : {e['compromis']} compromis pour {e['consensus + conflit']} consensus + conflit "
                        "(repère de M. Mass : à peu près autant de compromis que de consensus + conflit).\n",
                 "discret")
        peu = [o["texte"] for o in outs if not o["fiable"]]
        if peu:
            t.insert("end", "⚠ Peu d'outputs semblables dans l'historique (estimation proche de la moyenne, "
                            "marqué ⚠ dans le tableau) : " + " ; ".join(peu) + "\n", "alerte")
        for a in r["alertes"]:
            t.insert("end", "⚠ " + a + "\n", "alerte")
        t.configure(state="disabled")

        self.vider(self.tab_outputs)
        for i, o in enumerate(outs):
            if i == ligne:
                self.ligne_jusquici(r)
            self.tab_outputs.insert("", "end", values=(
                o["rang"], f"{SYMBOLE[o['categorie']]} {o['texte']}", pct(o["consensus"]),
                pct(o["compromis"]), pct(o["conflit"]), o["issue_probable"], algo.texte_contreparties(o),
                o["position"] + ("" if o["fiable"] else " ⚠"), algo.texte_avec_fournisseur_court(o)),
                tags=(POSITION_TAG[o["position"]],))
        if ligne == n:
            self.ligne_jusquici(r)
        # colonne « Déjà avec ce fournisseur » seulement s'il y a un historique avec lui
        avec = any(o.get("avec_fournisseur") for o in outs)
        colonnes = list(self.colonnes_sans_fourn)
        if avec:                                       # juste après « Issue probable », pour rester visible
            colonnes.insert(colonnes.index("issue") + 1, "fourn")
        self.tab_outputs.configure(displaycolumns=colonnes)
        self.tab_outputs.heading("fourn", text=f"Déjà avec {r.get('fournisseur') or 'ce fournisseur'}")

        par_id = {x["id"]: x for x in self.modele.toutes_negos()}
        self.vider(self.tab_voisins)
        for pid, s, res in r["precedents"]:
            p = par_id[pid]
            self.tab_voisins.insert("", "end", values=(pid, f"{s:.2f}", pct(res), texte_situation(p),
                                                       texte_contexte(p),
                                                       resume_arguments(p["arguments"], p.get("contreparties"))))

    def ligne_jusquici(self, r):
        self.tab_outputs.insert("", "end", values=("", "──────── jusqu'ici ────────", "", "", "", "", "",
                                                   f"seuil de conflit : {pct(r['seuil'])}", ""), tags=("ligne",))

    # ----- onglet Ajouter -----
    def construire_onglet_ajouter(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Ajouter une négo terminée")
        self.sit_ajout, self.ctx_ajout = colonne_contexte(f)
        droite = ttk.Frame(f)
        droite.pack(side="left", fill="both", expand=True)
        ttk.Button(droite, text="Reprendre les outputs et le contexte de la dernière préparation",
                   command=self.reprendre_estimation).pack(anchor="w")
        ttk.Label(droite, wraplength=800, justify="left", foreground="#555", text=(
            "Outputs dans l'ordre où ils ont été négociés. Indique combien ont été obtenus (dans l'ordre) : l'output "
            "suivant est celui qui a bloqué (conflit). Pour chaque output obtenu en échange de quelque chose "
            "(compromis), sélectionne-le et choisis la contrepartie concédée.")).pack(anchor="w", pady=(6, 0))
        self.ed_ajout = EditeurArguments(droite, on_change=self.maj_spin, avec_contreparties=True)
        self.ed_ajout.pack(fill="both", expand=True, pady=6)

        grille = ttk.Frame(droite)
        grille.pack(fill="x", pady=4)
        self.var_id = tk.StringVar()
        self.var_date = tk.StringVar(value=date.today().isoformat())
        self.var_obtenus = tk.IntVar(value=0)
        ttk.Label(grille, text="Identifiant").grid(row=0, column=0, sticky="w")
        ttk.Entry(grille, textvariable=self.var_id, width=12).grid(row=1, column=0, sticky="w", padx=(0, 12))
        ttk.Label(grille, text="Date (AAAA-MM-JJ)").grid(row=0, column=2, sticky="w")
        ttk.Entry(grille, textvariable=self.var_date, width=12).grid(row=1, column=2, sticky="w", padx=(0, 12))
        ttk.Label(grille, text="Outputs obtenus (dans l'ordre)").grid(row=0, column=3, sticky="w")
        self.spin = ttk.Spinbox(grille, from_=0, to=0, textvariable=self.var_obtenus, width=6, command=self.maj_spin)
        self.spin.grid(row=1, column=3, sticky="w")
        self.var_res_txt = tk.StringVar()
        ttk.Label(grille, textvariable=self.var_res_txt).grid(row=1, column=4, sticky="w", padx=8)
        self.var_exclue = tk.BooleanVar(value=False)
        ttk.Checkbutton(droite, variable=self.var_exclue, text="Négo exceptionnelle : la garder dans l'historique "
                        "mais ne pas s'en servir pour estimer").pack(anchor="w", pady=(6, 0))
        ttk.Label(droite, foreground="#555", text="Période : si elle n'est pas choisie, elle est déduite de la date."
                  ).pack(anchor="w")
        ttk.Button(droite, text="Ajouter à l'historique", command=self.ajouter).pack(anchor="w", pady=8)
        self.var_msg_ajout = tk.StringVar()
        ttk.Label(droite, textvariable=self.var_msg_ajout, foreground="#2e7d32", wraplength=800).pack(anchor="w")

    def maj_spin(self):
        n = len(self.ed_ajout.arguments)
        self.spin.configure(to=n)
        try:
            k = min(max(int(self.var_obtenus.get()), 0), n)
        except (tk.TclError, ValueError):
            k = 0
        self.var_obtenus.set(k)
        if not n:
            self.var_res_txt.set("")
        elif k < n:
            self.var_res_txt.set(f"→ {k}/{n} obtenus ; bloqué sur n° {k + 1} « {self.ed_ajout.arguments[k][0]} »")
        else:
            self.var_res_txt.set(f"→ {k}/{n} obtenus : tout a été obtenu")

    def proposer_id(self):
        if self.modele is None:
            return
        nums = [int(x["id"][1:]) for x in self.modele.toutes_negos() if x["id"][:1] == "N" and x["id"][1:].isdigit()]
        self.var_id.set(f"N{(max(nums) + 1) if nums else 1:04d}")

    def reprendre_estimation(self):
        if self.derniere:
            nego = self.derniere[0]
            self.ed_ajout.definir(nego["arguments"])
            self.ctx_ajout.definir(nego.get("forces"), nego.get("raison"), nego.get("reponses"))
            self.sit_ajout.definir(situation_de(nego))

    def ajouter(self):
        if self.modele is None:
            return
        args = list(self.ed_ajout.arguments)
        if not args:
            messagebox.showwarning("Ajout", "Ajoute au moins un output.")
            return
        self.maj_spin()
        k = int(self.var_obtenus.get())
        try:
            jour = date.fromisoformat(self.var_date.get().strip()).isoformat()
        except ValueError:
            messagebox.showwarning("Ajout", "Date invalide : format attendu AAAA-MM-JJ.")
            return
        manquantes = self.ctx_ajout.manquantes()
        if manquantes and not messagebox.askyesno(
                "Ajout", "Questions non complétées (comptées 3/5, neutre) : " + ", ".join(manquantes)
                + ".\nAjouter quand même ?"):
            return
        # une contrepartie n'a de sens que pour un output obtenu
        contreparties = [cp if i < k else None for i, cp in enumerate(self.ed_ajout.contreparties)]
        ignorees = sum(1 for i, cp in enumerate(self.ed_ajout.contreparties) if cp and i >= k)
        forces, raison = self.ctx_ajout.valeurs()
        situation = self.sit_ajout.valeurs()
        situation["trimestre"] = situation["trimestre"] or algo.trimestre_de(jour)
        exclue = bool(self.var_exclue.get())
        nego = {"id": self.var_id.get().strip(), "arguments": args, "contreparties": contreparties,
                "resultat": round(k / len(args), 3), "forces": forces, "raison": raison, **situation,
                "exclue": exclue, "infos": {"date": jour, **self.ctx_ajout.reponses()}}

        def calcul():
            algo.ajouter_negociation(self.modele, nego)
            self.modele.sauvegarder(FICHIER_MODELE)
            return nego

        def fini(n):
            msg = (f"Négo {n['id']} ajoutée ({k}/{len(args)} obtenus, "
                   f"{sum(1 for cp in contreparties if cp)} contrepartie(s)) : "
                   + ("marquée exclue (pas utilisée pour estimer), " if exclue else "arbre mis à jour et ")
                   + f"écrite dans {os.path.basename(self.modele.chemin_csv)}.")
            if ignorees:
                msg += f" {ignorees} contrepartie(s) ignorée(s) : outputs non obtenus."
            self.var_msg_ajout.set(msg)
            self.ed_ajout.effacer()
            self.var_exclue.set(False)
            for sit in (self.sit_estimer, self.sit_ajout):
                sit.definir_fournisseurs(self.modele.fournisseurs())
            self.remplir_historique()
            self.remplir_arbre()
            self.proposer_id()
        self.en_fond("Ajout de la négo…", calcul, fini)

    # ----- onglet Historique -----
    def construire_onglet_historique(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Historique")
        barre = ttk.Frame(f)
        barre.pack(fill="x")
        ttk.Label(barre, text="Catégorie d'achat :").pack(side="left")
        self.var_filtre = tk.StringVar(value="Tous")
        self.cb_filtre = ttk.Combobox(barre, textvariable=self.var_filtre, state="readonly", width=24)
        self.cb_filtre.pack(side="left", padx=4)
        self.cb_filtre.bind("<<ComboboxSelected>>", lambda e: self.remplir_historique())
        ttk.Label(barre, text="Fournisseur :").pack(side="left", padx=(12, 0))
        self.var_filtre_f = tk.StringVar(value="Tous")
        self.cb_filtre_f = ttk.Combobox(barre, textvariable=self.var_filtre_f, state="readonly", width=16)
        self.cb_filtre_f.pack(side="left", padx=4)
        self.cb_filtre_f.bind("<<ComboboxSelected>>", lambda e: self.remplir_historique())
        ttk.Button(barre, text="Exclure / réintégrer la négo sélectionnée",
                   command=self.basculer_exclusion).pack(side="right")
        self.var_stats = tk.StringVar()
        ttk.Label(barre, textvariable=self.var_stats).pack(side="left", padx=12)
        ttk.Label(f, foreground="#555", text="Négo exclue : exceptionnelle, gardée dans l'historique mais pas "
                                             "utilisée pour estimer (ni pour les contreparties, ni pour l'évaluation)."
                  ).pack(anchor="w", pady=(4, 0))
        self.tab_hist = self.tableau(f, [("nego", "Négo", 60), ("date", "Date", 90), ("excl", "Exclue", 55),
                                         ("sit", "Situation", 300), ("nb", "Nb outputs", 70),
                                         ("res", "Part obtenue", 80), ("ctx", "Forces (raison)", 200),
                                         ("args", "Outputs (et contreparties)", 600)], hauteur=20)
        self.tab_hist.tag_configure("exclue", foreground="#999999")

    def remplir_historique(self):
        if self.modele is None:
            return
        negos = self.modele.toutes_negos()
        categories = sorted({x.get("categorie_achat") for x in negos} - {None})
        self.cb_filtre.configure(values=["Tous"] + categories)
        self.cb_filtre_f.configure(values=["Tous"] + self.modele.fournisseurs())
        choix = self.var_filtre.get()
        if choix != "Tous":
            negos = [x for x in negos if x.get("categorie_achat") == choix]
        choix_f = self.var_filtre_f.get()
        if choix_f != "Tous":
            negos = [x for x in negos if algo.meme_fournisseur(choix_f, x.get("fournisseur"))]
        self.vider(self.tab_hist)
        for x in sorted(negos, key=lambda x: x.get("infos", {}).get("date", ""), reverse=True):
            i = x.get("infos", {})
            self.tab_hist.insert("", "end", iid=x["id"], values=(
                x["id"], i.get("date", ""), "oui" if x.get("exclue") else "", texte_situation(x),
                len(x["arguments"]), pct(x["resultat"]), texte_contexte(x),
                resume_arguments(x["arguments"], x.get("contreparties"))),
                tags=("exclue",) if x.get("exclue") else ())
        if negos:
            moy = sum(x["resultat"] for x in negos) / len(negos)
            nb_excl = sum(1 for x in negos if x.get("exclue"))
            self.var_stats.set(f"{len(negos)} négos" + (f" dont {nb_excl} exclue(s)" if nb_excl else "")
                               + f" · part obtenue en moyenne {pct(moy)}")
        else:
            self.var_stats.set("aucune négo")

    def basculer_exclusion(self):
        """Exclut (ou réintègre) la négo sélectionnée : écrit dans le CSV puis reconstruit le modèle."""
        sel = self.tab_hist.selection()
        if self.modele is None or not sel:
            messagebox.showinfo("Exclusion", "Sélectionne d'abord une négo dans la liste.")
            return
        nego_id = sel[0]
        nego = next(x for x in self.modele.toutes_negos() if x["id"] == nego_id)
        exclure = not nego.get("exclue")
        question = (f"Exclure la négo {nego_id} ? Elle restera dans l'historique mais ne servira plus à estimer."
                    if exclure else f"Réintégrer la négo {nego_id} dans les estimations ?")
        if not messagebox.askyesno("Exclusion", question + "\nLe modèle sera reconstruit."):
            return
        chemin = self.modele.chemin_csv

        def calcul():
            algo.changer_exclusion(chemin, nego_id, exclure)
            m = algo.ModeleNegociation(algo.lire_negos_csv(chemin), chemin_csv=chemin)
            m.sauvegarder(FICHIER_MODELE)
            return m
        self.en_fond("Mise à jour de l'exclusion et reconstruction…", calcul, self.modele_pret)

    # ----- onglet Arbre -----
    def construire_onglet_arbre(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Arbre")
        ttk.Label(f, text="Clique sur un nœud pour voir ses négos. Les nœuds marqués « groupe » sont les feuilles, "
                          "utilisées pour estimer.").pack(anchor="w")
        panneau = ttk.PanedWindow(f, orient="horizontal")
        panneau.pack(fill="both", expand=True, pady=4)
        cadre_arbre = ttk.Frame(panneau)
        self.vue_arbre = ttk.Treeview(cadre_arbre, columns=("n", "moy", "ecart"), selectmode="browse")
        self.vue_arbre.heading("#0", text="Nœud")
        self.vue_arbre.heading("n", text="Négos")
        self.vue_arbre.heading("moy", text="Part obtenue moy.")
        self.vue_arbre.heading("ecart", text="Écart-type")
        self.vue_arbre.column("#0", width=170)
        for c, larg in (("n", 70), ("moy", 115), ("ecart", 90)):
            self.vue_arbre.column(c, width=larg, anchor="e")
        self.vue_arbre.pack(fill="both", expand=True)
        self.vue_arbre.bind("<<TreeviewSelect>>", lambda e: self.afficher_noeud())
        panneau.add(cadre_arbre, weight=1)
        cadre_negos = ttk.Frame(panneau)
        self.tab_noeud = self.tableau(cadre_negos, [("nego", "Négo", 60), ("res", "Part obtenue", 80),
                                                    ("sit", "Situation", 260), ("ctx", "Forces (raison)", 200),
                                                    ("args", "Outputs", 500)], hauteur=20)
        panneau.add(cadre_negos, weight=2)
        self.noeuds = {}

    def remplir_arbre(self):
        if self.modele is None:
            return
        self.vue_arbre.delete(*self.vue_arbre.get_children())
        self.noeuds = {}
        y = self.modele.y

        def ajouter(noeud, parent, nom):
            r = y[noeud.indices]
            item = self.vue_arbre.insert(parent, "end", text=nom + ("  (groupe)" if noeud.est_feuille() else ""),
                                         values=(len(noeud.indices), pct(r.mean()), pct(r.std())),
                                         open=noeud.profondeur < 2)
            self.noeuds[item] = noeud
            for k, e in enumerate(noeud.enfants, 1):
                ajouter(e, item, f"{nom}.{k}")
        ajouter(self.modele.arbre, "", "1")

    def afficher_noeud(self):
        sel = self.vue_arbre.selection()
        if not sel:
            return
        noeud = self.noeuds[sel[0]]
        self.vider(self.tab_noeud)
        for i in noeud.indices:
            x = self.modele.negos[i]
            self.tab_noeud.insert("", "end", values=(x["id"], pct(x["resultat"]), texte_situation(x), texte_contexte(x),
                                                     resume_arguments(x["arguments"], x.get("contreparties"))))

    # ----- onglet Évaluer -----
    def construire_onglet_evaluer(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Évaluer")
        ttk.Label(f, wraplength=1000, justify="left", text=(
            "On construit le modèle sur les négos les plus anciennes, puis on prépare chacune des plus récentes AVANT "
            "de l'ajouter, comme en usage réel. Pour chaque output abordé (obtenu ou bloquant), on compare le risque "
            "de conflit annoncé à ce qui s'est passé. Erreur = moyenne de (risque annoncé − issue réelle)², avec "
            "issue = 1 si conflit, 0 si obtenu : plus bas = mieux (score de Brier). Référence naïve : le taux moyen "
            "de conflit de tous les outputs de l'historique. On le fait deux fois : outputs "
            "seuls, puis avec le contexte. Le calcul peut prendre une ou deux minutes.")).pack(anchor="w")
        ligne = ttk.Frame(f)
        ligne.pack(anchor="w", pady=8)
        ttk.Label(ligne, text="Part de l'historique utilisée pour le test :").pack(side="left")
        self.var_part = tk.StringVar(value="0.2")
        ttk.Spinbox(ligne, from_=0.1, to=0.4, increment=0.05, textvariable=self.var_part, width=6,
                    format="%.2f").pack(side="left", padx=4)
        ttk.Button(ligne, text="Lancer l'évaluation", command=self.evaluer).pack(side="left", padx=8)
        self.var_eval = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.var_eval, font=("TkDefaultFont", 12), justify="left").pack(anchor="w", pady=8)

    def evaluer(self):
        if self.modele is None:
            return
        try:
            part = min(max(float(self.var_part.get().replace(",", ".")), 0.05), 0.5)
        except ValueError:
            part = 0.2
        chemin = self.modele.chemin_csv
        seuil = self.seuil()

        def calcul():
            historique = algo.lire_negos_csv(chemin)
            seuls = algo.evaluer(historique, part_test=part, sans_contexte=True, seuil=seuil)
            avec = (algo.evaluer(historique, part_test=part, seuil=seuil)
                    if any(algo.a_contexte(n) for n in historique) else None)
            return seuls, avec

        def fini(res):
            seuls, avec = res
            lignes = [f"{seuls['n_outputs']} outputs abordés dans les {seuls['n_test']} négos de test. "
                      "Erreur sur le risque de conflit (plus bas = mieux) :",
                      f"   {seuls['brier_reference']:.3f}   référence naïve (taux moyen de conflit de tous les outputs)",
                      f"   {seuls['brier_outil']:.3f}   outil, outputs seuls"]
            dernier = seuls
            if avec:
                lignes.append(f"   {avec['brier_outil']:.3f}   outil, outputs + contexte (forces, raison, situation)")
                dernier = avec
            else:
                lignes.append("   (l'historique ne contient pas de contexte : pas de comparaison possible)")
            lignes.append(f"\nAvec le seuil de {pct(dernier['seuil'])} (outil{' + contexte' if avec else ''}) :")
            for cle, n_cle, texte in (("conflit_au_dessus", "n_au_dessus", "annoncés AU-DESSUS du seuil"),
                                      ("conflit_en_dessous", "n_en_dessous", "annoncés en dessous du seuil")):
                if dernier[cle] is not None:
                    lignes.append(f"   {dernier[n_cle]} outputs {texte} : {pct(dernier[cle])} ont réellement bloqué")
            self.var_eval.set("\n".join(lignes))
        self.en_fond("Évaluation en cours (deux passes)…", calcul, fini)

    # ----- onglet Assistant IA (Ollama, local) -----
    def construire_onglet_assistant(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Assistant IA")
        ttk.Label(f, wraplength=1150, justify="left", text=(
            "L'assistant utilise une IA qui tourne sur cet ordinateur (Ollama) : rien n'est envoyé à l'extérieur. "
            "Il ne calcule pas l'estimation, qui reste faite par l'algorithme. Il propose des notes de rapport de "
            "force à partir d'une description libre (que vous validez), et explique en français jusqu'où aller."
        )).pack(anchor="w")

        barre = ttk.Frame(f)
        barre.pack(fill="x", pady=6)
        ttk.Label(barre, text="Modèle Ollama :").pack(side="left")
        self.var_modele_ia = tk.StringVar(value=assistant_ia.MODELE)
        ttk.Entry(barre, textvariable=self.var_modele_ia, width=18).pack(side="left", padx=4)
        ttk.Button(barre, text="Vérifier Ollama", command=self.verifier_ia).pack(side="left", padx=4)
        self.var_etat_ia = tk.StringVar(value="Ollama non vérifié.")
        ttk.Label(barre, textvariable=self.var_etat_ia, wraplength=800).pack(side="left", padx=8)

        cadre1 = ttk.LabelFrame(f, text="1. Décrire la situation → proposition de rapport de force", padding=8)
        cadre1.pack(fill="x", pady=4)
        self.txt_description = tk.Text(cadre1, height=4, wrap="word")
        self.txt_description.pack(fill="x")
        self.txt_description.insert("1.0", "Renouvellement du contrat d'emballages avec notre verrier habituel. "
                                    "Il n'y a que deux autres fournisseurs capables de livrer ces volumes et nous "
                                    "représentons une petite part de son chiffre d'affaires. Passer au plastique "
                                    "n'est pas envisageable pour nos clients.")
        ligne = ttk.Frame(cadre1)
        ligne.pack(fill="x", pady=4)
        ttk.Button(ligne, text="Proposer un rapport de force", command=self.proposer_contexte).pack(side="left")
        self.btn_appliquer = ttk.Button(ligne, text="Appliquer à l'onglet Préparer", command=self.appliquer_contexte,
                                        state="disabled")
        self.btn_appliquer.pack(side="left", padx=6)
        self.var_raison_ia = tk.StringVar()
        ttk.Label(ligne, textvariable=self.var_raison_ia).pack(side="left", padx=8)
        self.tab_propositions = self.tableau(cadre1, [("force", "Force", 190), ("note", "Note proposée", 125),
                                                     ("args", "Justification de l'IA", 700)], hauteur=5)
        self.proposition = None

        cadre2 = ttk.LabelFrame(f, text="2. Expliquer la dernière préparation (onglet Préparer une négo)", padding=8)
        cadre2.pack(fill="both", expand=True, pady=4)
        ttk.Button(cadre2, text="Expliquer avec l'IA", command=self.expliquer_ia).pack(anchor="w")
        self.txt_explication = tk.Text(cadre2, height=8, wrap="word", relief="flat", background="#f6f6f6")
        self.txt_explication.pack(fill="both", expand=True, pady=4)
        self.txt_explication.tag_configure("alerte", foreground="#b26a00")
        self.txt_explication.configure(state="disabled")

    def modele_ia(self):
        return self.var_modele_ia.get().strip() or assistant_ia.MODELE

    def verifier_ia(self):
        modele = self.modele_ia()
        self.en_fond("Vérification d'Ollama…", lambda: assistant_ia.verifier_ollama(modele),
                     lambda res: self.var_etat_ia.set(("✅ " if res[0] else "⚠ ") + res[1]))

    def proposer_contexte(self):
        description = self.txt_description.get("1.0", "end").strip()
        if not description:
            messagebox.showwarning("Assistant IA", "Écris d'abord une description de la situation.")
            return
        modele = self.modele_ia()
        self.en_fond("L'IA analyse la description (le premier appel peut prendre une minute)…",
                     lambda: assistant_ia.interpreter_contexte(description, modele), self.afficher_proposition)

    def afficher_proposition(self, prop):
        self.proposition = prop
        self.vider(self.tab_propositions)
        for col, nom, *_ in algo.FORCES:
            note = prop["forces"][col]
            self.tab_propositions.insert("", "end", values=(nom, f"{note}/5" if note else "inconnue",
                                                            prop["justifications"][col] or "—"))
        self.var_raison_ia.set(f"Raison : {prop['raison'] or 'inconnue'}  ·  Situation : "
                               f"{texte_situation(prop) if any(situation_de(prop).values()) else 'inconnue'}")
        self.btn_appliquer.configure(state="normal")
        self.var_etat_ia.set("Proposition reçue : vérifie-la avant de l'appliquer.")

    def appliquer_contexte(self):
        if not self.proposition:
            return
        actuelles, raison_actuelle = self.ctx_estimer.valeurs()
        actuelles = actuelles or {c: 3 for c in algo.COLONNES_FORCES}
        forces = {c: (v if v else actuelles[c]) for c, v in self.proposition["forces"].items()}
        # forces non évaluées par l'IA : on garde les réponses déjà données aux questions
        self.ctx_estimer.definir(forces, self.proposition["raison"] or raison_actuelle, self.ctx_estimer.reponses())
        actuelle = self.sit_estimer.valeurs()
        proposee = situation_de(self.proposition)
        if proposee["categorie_achat"] and proposee["categorie_achat"] != actuelle["categorie_achat"]:
            actuelle["sous_categorie"] = None          # l'ancienne sous-catégorie ne correspond plus
        self.sit_estimer.definir({k: (v or actuelle[k]) for k, v in proposee.items()})
        self.onglets.select(0)
        if self.proposition["inconnues"]:
            messagebox.showinfo("Assistant IA", "Notes appliquées. Forces que l'IA n'a pas pu évaluer (valeur "
                                "précédente conservée, à vérifier) : " + ", ".join(self.proposition["inconnues"]))

    def expliquer_ia(self):
        if not self.derniere:
            messagebox.showwarning("Assistant IA", "Fais d'abord une préparation dans l'onglet Préparer une négo.")
            return
        nego, r = self.derniere
        modele = self.modele_ia()

        def fini(res):
            texte, suspects = res
            t = self.txt_explication
            t.configure(state="normal")
            t.delete("1.0", "end")
            t.insert("end", texte + "\n")
            if suspects:
                t.insert("end", "\n⚠ Chiffres cités par l'IA qui ne figurent pas dans les résultats de l'outil : "
                         + ", ".join(suspects) + ". Ne pas s'y fier.", "alerte")
            t.configure(state="disabled")
        self.en_fond("L'IA rédige l'explication…", lambda: assistant_ia.expliquer_estimation(nego, r, modele), fini)

    # ----- utilitaires -----
    def tableau(self, parent, colonnes, hauteur=10, etirer="args"):
        cadre = ttk.Frame(parent)
        cadre.pack(fill="both", expand=True)
        t = ttk.Treeview(cadre, columns=[c for c, _, _ in colonnes], show="headings", height=hauteur)
        for c, titre, larg in colonnes:
            t.heading(c, text=titre)
            t.column(c, width=larg, stretch=(c == etirer), anchor="w")
        sy = ttk.Scrollbar(cadre, orient="vertical", command=t.yview)
        sx = ttk.Scrollbar(cadre, orient="horizontal", command=t.xview)
        t.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        t.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        cadre.rowconfigure(0, weight=1)
        cadre.columnconfigure(0, weight=1)
        return t

    @staticmethod
    def vider(tableau):
        tableau.delete(*tableau.get_children())


def main():
    root = tk.Tk()
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
