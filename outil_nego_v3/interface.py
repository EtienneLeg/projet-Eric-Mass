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


def resume_arguments(args):
    return " · ".join(f"[{c}] {a}" for a, c in args)


# ---------------------------------------------------------------------------
# Éditeur d'arguments réutilisable (onglets Estimer et Ajouter)
# ---------------------------------------------------------------------------
class EditeurArguments(ttk.Frame):
    """Liste ordonnée d'arguments : du plus favorable (en haut) au moins favorable (en bas)."""

    def __init__(self, parent, on_change=None):
        super().__init__(parent)
        self.on_change = on_change

        saisie = ttk.Frame(self)
        saisie.pack(fill="x", pady=(0, 4))
        ttk.Label(saisie, text="Argument :").pack(side="left")
        self.var_texte = tk.StringVar()
        champ = ttk.Entry(saisie, textvariable=self.var_texte, width=50)
        champ.pack(side="left", fill="x", expand=True, padx=4)
        champ.bind("<Return>", lambda e: self.ajouter())
        self.var_cat = tk.StringVar(value="compromis")
        ttk.Combobox(saisie, textvariable=self.var_cat, values=CATS, state="readonly", width=11).pack(side="left")
        ttk.Button(saisie, text="Ajouter", command=self.ajouter).pack(side="left", padx=4)

        corps = ttk.Frame(self)
        corps.pack(fill="both", expand=True)
        self.liste = ttk.Treeview(corps, columns=("rang", "categorie", "argument"), show="headings", height=7,
                                  selectmode="browse")
        for col, titre, larg in [("rang", "N°", 40), ("categorie", "Catégorie", 100), ("argument", "Argument", 420)]:
            self.liste.heading(col, text=titre)
            self.liste.column(col, width=larg, stretch=(col == "argument"), anchor="w")
        for c in CATS:
            self.liste.tag_configure(c, foreground=COULEUR[c])
        self.liste.pack(side="left", fill="both", expand=True)

        boutons = ttk.Frame(corps)
        boutons.pack(side="left", fill="y", padx=(6, 0))
        for texte, cmd in [("▲ Monter", lambda: self.deplacer(-1)), ("▼ Descendre", lambda: self.deplacer(1)),
                           ("Changer catégorie", self.changer_categorie), ("Supprimer", self.supprimer),
                           ("Tout effacer", self.effacer)]:
            ttk.Button(boutons, text=texte, command=cmd).pack(fill="x", pady=2)

        self.arguments = []      # [(texte, categorie)] dans l'ordre de la colonne

    def ajouter(self):
        texte = self.var_texte.get().strip()
        if not texte:
            return
        self.arguments.append((texte, self.var_cat.get()))
        self.var_texte.set("")
        self.rafraichir(len(self.arguments) - 1)

    def index_selection(self):
        sel = self.liste.selection()
        return self.liste.index(sel[0]) if sel else None

    def deplacer(self, sens):
        i = self.index_selection()
        if i is None or not 0 <= i + sens < len(self.arguments):
            return
        a = self.arguments
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
            self.rafraichir(min(i, len(self.arguments) - 1))

    def effacer(self):
        self.definir([])

    def definir(self, arguments):
        self.arguments = list(arguments)
        self.rafraichir()

    def rafraichir(self, selection=None):
        self.liste.delete(*self.liste.get_children())
        for i, (texte, cat) in enumerate(self.arguments, 1):
            self.liste.insert("", "end", values=(i, f"{SYMBOLE[cat]} {cat}", texte), tags=(cat,))
        if selection is not None and 0 <= selection < len(self.arguments):
            item = self.liste.get_children()[selection]
            self.liste.selection_set(item)
            self.liste.see(item)
        if self.on_change:
            self.on_change()


# ---------------------------------------------------------------------------
# Saisie du contexte : 5 forces (1 à 5) + raison de la négo
# ---------------------------------------------------------------------------
class ContexteForces(ttk.LabelFrame):
    """Rapport de force selon les forces définies dans algo.FORCES (5 = favorable au négociateur)."""

    def __init__(self, parent):
        super().__init__(parent, text="Rapport de force (5 = favorable pour vous)", padding=8)
        self.var_actif = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Renseigner le rapport de force", variable=self.var_actif,
                        command=self.maj_actif).grid(row=0, column=0, columnspan=2, sticky="w")
        self.vars, self.widgets = {}, []
        for k, (col, nom, question, bas, haut) in enumerate(algo.FORCES, start=1):
            cadre = ttk.Frame(self)
            cadre.grid(row=k, column=0, sticky="w", pady=2)
            ttk.Label(cadre, text=nom, font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            ttk.Label(cadre, text=question, foreground="#555", wraplength=250).pack(anchor="w")
            var = tk.StringVar(value="3")
            valeurs = [f"1 – {bas}", "2", "3", "4", f"5 – {haut}"]
            cb = ttk.Combobox(self, textvariable=var, values=valeurs, state="readonly", width=17)
            cb.grid(row=k, column=1, sticky="ne", padx=(8, 0), pady=(18, 0))
            self.vars[col] = var
            self.widgets.append(cb)
        r = len(algo.FORCES) + 1
        ttk.Label(self, text="Raison de la négo", font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0,
                                                                                          sticky="w", pady=(6, 0))
        self.var_raison = tk.StringVar(value=algo.RAISONS[0])
        cb = ttk.Combobox(self, textvariable=self.var_raison, values=algo.RAISONS, state="readonly", width=17)
        cb.grid(row=r, column=1, sticky="e", padx=(8, 0), pady=(6, 0))
        self.widgets.append(cb)

    def maj_actif(self):
        for w in self.widgets:
            w.configure(state="readonly" if self.var_actif.get() else "disabled")

    def valeurs(self):
        """(forces, raison) ou (None, None) si le contexte n'est pas renseigné."""
        if not self.var_actif.get():
            return None, None
        forces = {col: int(var.get()[0]) for col, var in self.vars.items()}
        return forces, self.var_raison.get()

    def definir(self, forces, raison):
        self.var_actif.set(bool(forces))
        if forces:
            for col, var in self.vars.items():
                note = int(forces[col])
                var.set(self.widgets[list(self.vars).index(col)].cget("values")[note - 1])
        if raison:
            self.var_raison.set(raison)
        self.maj_actif()


AUCUN = "—"


class ContexteSituation(ttk.Frame):
    """Secteur de l'entreprise, catégorie d'achat et sous-catégorie (liste dépendante de la catégorie)."""

    def __init__(self, parent):
        super().__init__(parent)
        ttk.Label(self, text="Situation :", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.var_secteur = tk.StringVar(value=AUCUN)
        self.var_categorie = tk.StringVar(value=AUCUN)
        self.var_sous = tk.StringVar(value=AUCUN)
        for texte, var, valeurs, larg in (("Secteur", self.var_secteur, algo.SECTEURS, 24),
                                          ("Catégorie d'achat", self.var_categorie, list(algo.CATEGORIES_ACHAT), 22)):
            ttk.Label(self, text=texte).pack(side="left", padx=(12, 4))
            cb = ttk.Combobox(self, textvariable=var, values=[AUCUN] + valeurs, state="readonly", width=larg)
            cb.pack(side="left")
            if var is self.var_categorie:
                cb.bind("<<ComboboxSelected>>", lambda e: self.maj_sous_categories())
        ttk.Label(self, text="Sous-catégorie").pack(side="left", padx=(12, 4))
        self.cb_sous = ttk.Combobox(self, textvariable=self.var_sous, values=[AUCUN], state="readonly", width=22)
        self.cb_sous.pack(side="left")

    def maj_sous_categories(self):
        sous = algo.CATEGORIES_ACHAT.get(self.var_categorie.get(), [])
        self.cb_sous.configure(values=[AUCUN] + sous)
        if self.var_sous.get() not in sous:
            self.var_sous.set(AUCUN)

    def valeurs(self):
        val = lambda v: None if v.get() == AUCUN else v.get()
        return {"secteur": val(self.var_secteur), "categorie_achat": val(self.var_categorie),
                "sous_categorie": val(self.var_sous)}

    def definir(self, situation):
        situation = situation or {}
        self.var_secteur.set(situation.get("secteur") or AUCUN)
        self.var_categorie.set(situation.get("categorie_achat") or AUCUN)
        self.maj_sous_categories()
        self.var_sous.set(situation.get("sous_categorie") or AUCUN)


def situation_de(nego):
    return {cle: nego.get(cle) for cle in ("secteur", "categorie_achat", "sous_categorie")}


def texte_situation(nego):
    cat = nego.get("categorie_achat")
    if cat and nego.get("sous_categorie"):
        cat = f"{cat} / {nego['sous_categorie']}"
    morceaux = [m for m in (nego.get("secteur"), cat) if m]
    return " · ".join(morceaux) if morceaux else "—"


def texte_contexte(nego):
    if not nego.get("forces"):
        return "—"
    notes = "-".join(str(nego["forces"][c]) for c in algo.COLONNES_FORCES)
    return f"{notes}  ({nego.get('raison') or '?'})"


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
        root.geometry("1280x880")
        root.minsize(1100, 760)
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
        self.remplir_historique()
        self.remplir_arbre()
        self.proposer_id()

    def maj_etat(self):
        m = self.modele
        if m is None:
            self.var_etat.set("Aucun modèle chargé.")
            return
        txt = (f"{len(m.negos)} négos  ·  {self.nb_feuilles(m.arbre)} groupes  ·  "
               f"{m.ajouts_depuis_construction} ajout(s) récent(s)  ·  embeddings : "
               f"{'modèle de phrases' if m.emb.mode == 'sentence-transformers' else 'TF-IDF (repli)'}  ·  "
               f"contexte : {'effets appris sur ' + str(m.effets['n']) + ' négos' if m.effets else 'non utilisé'}")
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
        self.onglets.add(f, text="Estimer une négo")
        self.sit_estimer = ContexteSituation(f)
        self.sit_estimer.pack(fill="x", pady=(0, 6))
        self.sit_estimer.definir({"secteur": "Grande distribution", "categorie_achat": "Emballage",
                                  "sous_categorie": "Verre"})
        haut = ttk.Frame(f)
        haut.pack(fill="x")
        gauche_h = ttk.Frame(haut)
        gauche_h.pack(side="left", fill="both", expand=True)
        ttk.Label(gauche_h, text="Arguments, du plus favorable (en haut) au moins favorable (en bas) :"
                  ).pack(anchor="w")
        self.ed_estimer = EditeurArguments(gauche_h)
        self.ed_estimer.pack(fill="x", pady=4)
        self.ctx_estimer = ContexteForces(haut)
        self.ctx_estimer.pack(side="left", fill="y", padx=(10, 0))
        self.ctx_estimer.definir({"force_fournisseur": 4, "force_acheteur": 3, "force_substituts": 2,
                                  "force_entrants": 3, "force_rivalite": 3}, "fin de contrat")
        self.ed_estimer.definir([("garder les volumes de l'an dernier", "consensus"),
                                 ("allonger les délais de paiement", "compromis"),
                                 ("baisse de prix de 5 %", "compromis"),
                                 ("exclusivité sur la catégorie", "conflit")])
        ttk.Button(f, text="Estimer le résultat", command=self.estimer).pack(anchor="w", pady=4)

        res = ttk.Frame(f)
        res.pack(fill="both", expand=True)
        gauche = ttk.Frame(res)
        gauche.pack(side="left", fill="both", expand=True)
        self.var_estimation = tk.StringVar(value="—")
        ttk.Label(gauche, textvariable=self.var_estimation, font=("TkDefaultFont", 15, "bold")).pack(anchor="w")
        self.txt_resultat = tk.Text(gauche, height=12, wrap="word", relief="flat", background="#f6f6f6")
        self.txt_resultat.pack(fill="both", expand=True, pady=4)
        for c in CATS:
            self.txt_resultat.tag_configure(c, foreground=COULEUR[c])
        self.txt_resultat.tag_configure("alerte", foreground="#b26a00")
        self.txt_resultat.tag_configure("gras", font=("TkDefaultFont", 10, "bold"))
        self.txt_resultat.tag_configure("plus", foreground="#2e7d32")
        self.txt_resultat.tag_configure("discret", foreground="#777777")
        self.txt_resultat.tag_configure("moins", foreground="#c62828")
        self.txt_resultat.configure(state="disabled")

        droite = ttk.Frame(res)
        droite.pack(side="left", fill="both", expand=True, padx=(8, 0))
        ttk.Label(droite, text="Négos passées les plus proches :").pack(anchor="w")
        self.tab_voisins = self.tableau(droite, [("nego", "Négo", 60), ("ress", "Ressemblance", 90),
                                                 ("res", "Résultat", 65), ("sit", "Situation", 260),
                                                 ("ctx", "Forces (raison)", 210), ("args", "Arguments", 400)],
                                        hauteur=10)

    def estimer(self):
        if self.modele is None or not self.ed_estimer.arguments:
            return
        forces, raison = self.ctx_estimer.valeurs()
        nego = {"id": "NOUVELLE", "arguments": list(self.ed_estimer.arguments), "resultat": None,
                "forces": forces, "raison": raison, **self.sit_estimer.valeurs()}
        self.en_fond("Recherche des négos comparables…",
                     lambda: algo.estimer_negociation(self.modele, nego),
                     lambda r: self.afficher_estimation(nego, r))

    def afficher_estimation(self, nego, r):
        self.derniere = (nego, r)
        n = len(nego["arguments"])
        self.var_estimation.set(f"Résultat estimé : {pct(r['estimation'])}   (marge {pct(r['marge'][0])} – "
                                f"{pct(r['marge'][1])})   ≈ {r['estimation'] * n:.1f} / {n} arguments")
        if not r["contexte_utilise"]:
            ctx_txt = "Contexte non renseigné (ni rapport de force, ni situation) : estimation à partir des arguments seuls."
        elif not self.modele.effets:
            ctx_txt = "Pas assez de négos avec contexte dans l'historique pour en tenir compte."
        else:
            ctx_txt = (f"Avec ces arguments et un contexte moyen : {pct(r['estimation_contexte_moyen'])}  →  "
                       f"avec votre contexte : {pct(r['estimation'])}")
        t = self.txt_resultat
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end", ctx_txt + "\n", "gras")
        if r["facteurs"]:
            t.insert("end", "Contributions du contexte (effets appris sur l'historique) :\n")
            for fct in r["facteurs"]:
                pts = fct["points"] * 100
                t.insert("end", f"   {pts:+5.1f} pts  ", "plus" if pts >= 0 else "moins")
                t.insert("end", f"{fct['element']} ({fct['valeur']})\n")
        if not r["fiable"]:
            t.insert("end", "⚠ Pas assez de précédents comparables : estimation peu fiable.\n", "alerte")
        if r["remonte_de_niveaux"]:
            t.insert("end", f"(Peu de négos très proches dans le groupe d'arrivée : recherche élargie de "
                            f"{r['remonte_de_niveaux']} niveau(x) dans l'arbre.)\n", "discret")
        for a in r["alertes"]:
            t.insert("end", "⚠ Classement inhabituel : " + a + "\n", "alerte")
        t.insert("end", "\nTa colonne :\n", "gras")
        bas, haut = r["marge"][0] * n, r["marge"][1] * n
        for i, (texte, cat) in enumerate(nego["arguments"], 1):
            etat = "probablement obtenu" if i <= bas else ("dans la marge" if i <= haut else "probablement hors d'atteinte")
            t.insert("end", f"{i}. {SYMBOLE[cat]} {texte}", cat)
            t.insert("end", f"  —  {etat}\n")
            if i == int(r["estimation"] * n) and i < n:
                t.insert("end", "- - - - - - - - - - ligne estimée - - - - - - - - - -\n")
        t.configure(state="disabled")

        par_id = {x["id"]: x for x in self.modele.negos}
        self.vider(self.tab_voisins)
        for pid, s, res in r["precedents"]:
            p = par_id[pid]
            self.tab_voisins.insert("", "end", values=(pid, f"{s:.2f}", pct(res), texte_situation(p),
                                                       texte_contexte(p), resume_arguments(p["arguments"])))

    # ----- onglet Ajouter -----
    def construire_onglet_ajouter(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Ajouter une négo terminée")
        ttk.Button(f, text="Reprendre les arguments et le contexte de la dernière estimation",
                   command=self.reprendre_estimation).pack(anchor="w")
        self.sit_ajout = ContexteSituation(f)
        self.sit_ajout.pack(fill="x", pady=(6, 0))
        haut = ttk.Frame(f)
        haut.pack(fill="x", pady=6)
        self.ed_ajout = EditeurArguments(haut, on_change=self.maj_spin)
        self.ed_ajout.pack(side="left", fill="both", expand=True)
        self.ctx_ajout = ContexteForces(haut)
        self.ctx_ajout.pack(side="left", fill="y", padx=(10, 0))

        grille = ttk.Frame(f)
        grille.pack(fill="x", pady=4)
        self.var_id = tk.StringVar()
        self.var_date = tk.StringVar(value=date.today().isoformat())
        self.var_obtenus = tk.IntVar(value=0)
        ttk.Label(grille, text="Identifiant").grid(row=0, column=0, sticky="w")
        ttk.Entry(grille, textvariable=self.var_id, width=12).grid(row=1, column=0, sticky="w", padx=(0, 12))
        ttk.Label(grille, text="Date (AAAA-MM-JJ)").grid(row=0, column=2, sticky="w")
        ttk.Entry(grille, textvariable=self.var_date, width=12).grid(row=1, column=2, sticky="w", padx=(0, 12))
        ttk.Label(grille, text="Arguments obtenus (dans l'ordre)").grid(row=0, column=3, sticky="w")
        self.spin = ttk.Spinbox(grille, from_=0, to=0, textvariable=self.var_obtenus, width=6, command=self.maj_spin)
        self.spin.grid(row=1, column=3, sticky="w")
        self.var_res_txt = tk.StringVar()
        ttk.Label(grille, textvariable=self.var_res_txt).grid(row=1, column=4, sticky="w", padx=8)
        ttk.Button(f, text="Ajouter à l'historique", command=self.ajouter).pack(anchor="w", pady=8)
        self.var_msg_ajout = tk.StringVar()
        ttk.Label(f, textvariable=self.var_msg_ajout, foreground="#2e7d32").pack(anchor="w")

    def maj_spin(self):
        n = len(self.ed_ajout.arguments)
        self.spin.configure(to=n)
        try:
            k = min(max(int(self.var_obtenus.get()), 0), n)
        except (tk.TclError, ValueError):
            k = 0
        self.var_obtenus.set(k)
        self.var_res_txt.set(f"→ résultat {k}/{n} = {pct(k / n)}" if n else "")

    def proposer_id(self):
        if self.modele is None:
            return
        nums = [int(x["id"][1:]) for x in self.modele.negos if x["id"][:1] == "N" and x["id"][1:].isdigit()]
        self.var_id.set(f"N{(max(nums) + 1) if nums else 1:04d}")

    def reprendre_estimation(self):
        if self.derniere:
            nego = self.derniere[0]
            self.ed_ajout.definir(nego["arguments"])
            self.ctx_ajout.definir(nego.get("forces"), nego.get("raison"))
            self.sit_ajout.definir(situation_de(nego))

    def ajouter(self):
        if self.modele is None:
            return
        args = list(self.ed_ajout.arguments)
        if not args:
            messagebox.showwarning("Ajout", "Ajoute au moins un argument.")
            return
        self.maj_spin()
        k = int(self.var_obtenus.get())
        try:
            jour = date.fromisoformat(self.var_date.get().strip()).isoformat()
        except ValueError:
            messagebox.showwarning("Ajout", "Date invalide : format attendu AAAA-MM-JJ.")
            return
        infos = {"date": jour}
        forces, raison = self.ctx_ajout.valeurs()
        nego = {"id": self.var_id.get().strip(), "arguments": args, "resultat": round(k / len(args), 3),
                "forces": forces, "raison": raison, **self.sit_ajout.valeurs(), "infos": infos}

        def calcul():
            algo.ajouter_negociation(self.modele, nego)
            self.modele.sauvegarder(FICHIER_MODELE)
            return nego

        def fini(n):
            self.var_msg_ajout.set(f"Négo {n['id']} ajoutée ({pct(n['resultat'])}) : arbre mis à jour et "
                                   f"écrite dans {os.path.basename(self.modele.chemin_csv)}.")
            self.ed_ajout.effacer()
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
        self.cb_filtre = ttk.Combobox(barre, textvariable=self.var_filtre, state="readonly", width=26)
        self.cb_filtre.pack(side="left", padx=4)
        self.cb_filtre.bind("<<ComboboxSelected>>", lambda e: self.remplir_historique())
        self.var_stats = tk.StringVar()
        ttk.Label(barre, textvariable=self.var_stats).pack(side="left", padx=12)
        self.tab_hist = self.tableau(f, [("nego", "Négo", 60), ("date", "Date", 90), ("sit", "Situation", 280),
                                         ("nb", "Nb args", 60), ("res", "Résultat", 70),
                                         ("ctx", "Forces (raison)", 220), ("args", "Arguments", 600)], hauteur=20)

    def remplir_historique(self):
        if self.modele is None:
            return
        negos = self.modele.negos
        categories = sorted({x.get("categorie_achat") for x in negos} - {None})
        self.cb_filtre.configure(values=["Tous"] + categories)
        choix = self.var_filtre.get()
        if choix != "Tous":
            negos = [x for x in negos if x.get("categorie_achat") == choix]
        self.vider(self.tab_hist)
        for x in sorted(negos, key=lambda x: x.get("infos", {}).get("date", ""), reverse=True):
            i = x.get("infos", {})
            self.tab_hist.insert("", "end", values=(x["id"], i.get("date", ""), texte_situation(x),
                                                    len(x["arguments"]), pct(x["resultat"]), texte_contexte(x),
                                                    resume_arguments(x["arguments"])))
        if negos:
            moy = sum(x["resultat"] for x in negos) / len(negos)
            self.var_stats.set(f"{len(negos)} négos · résultat moyen {pct(moy)}")

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
        self.vue_arbre.heading("moy", text="Résultat moyen")
        self.vue_arbre.heading("ecart", text="Écart-type")
        self.vue_arbre.column("#0", width=170)
        for c, larg in (("n", 70), ("moy", 115), ("ecart", 90)):
            self.vue_arbre.column(c, width=larg, anchor="e")
        self.vue_arbre.pack(fill="both", expand=True)
        self.vue_arbre.bind("<<TreeviewSelect>>", lambda e: self.afficher_noeud())
        panneau.add(cadre_arbre, weight=1)
        cadre_negos = ttk.Frame(panneau)
        self.tab_noeud = self.tableau(cadre_negos, [("nego", "Négo", 60), ("res", "Résultat", 70),
                                                    ("sit", "Situation", 260), ("ctx", "Forces (raison)", 200),
                                                    ("args", "Arguments", 500)], hauteur=20)
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
                                                     resume_arguments(x["arguments"])))

    # ----- onglet Évaluer -----
    def construire_onglet_evaluer(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Évaluer")
        ttk.Label(f, wraplength=900, justify="left", text=(
            "On construit l'arbre sur les négos les plus anciennes, puis on estime chacune des plus récentes AVANT de "
            "l'ajouter, comme en usage réel. On compare l'erreur moyenne de l'outil à celle d'une estimation naïve : "
            "le résultat moyen de toute la base. On le fait deux fois : avec les arguments seuls, puis avec "
            "le rapport de force, pour mesurer ce que le contexte apporte. Le calcul peut prendre une ou deux "
            "minutes.")).pack(anchor="w")
        ligne = ttk.Frame(f)
        ligne.pack(anchor="w", pady=8)
        ttk.Label(ligne, text="Part de l'historique utilisée pour le test :").pack(side="left")
        self.var_part = tk.StringVar(value="0.2")
        ttk.Spinbox(ligne, from_=0.1, to=0.4, increment=0.05, textvariable=self.var_part, width=6,
                    format="%.2f").pack(side="left", padx=4)
        ttk.Button(ligne, text="Lancer l'évaluation", command=self.evaluer).pack(side="left", padx=8)
        self.var_eval = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.var_eval, font=("TkDefaultFont", 13)).pack(anchor="w", pady=8)

    def evaluer(self):
        if self.modele is None:
            return
        try:
            part = min(max(float(self.var_part.get().replace(",", ".")), 0.05), 0.5)
        except ValueError:
            part = 0.2
        chemin = self.modele.chemin_csv

        def calcul():
            historique = algo.lire_negos_csv(chemin)
            seuls = algo.evaluer(historique, part_test=part, sans_contexte=True)
            avec = algo.evaluer(historique, part_test=part) if any(n.get("forces") for n in historique) else None
            return seuls, avec

        def fini(res):
            seuls, avec = res
            lignes = [f"Sur {seuls['n_test']} négos de test, erreur moyenne entre résultat estimé et réel :",
                      f"   {seuls['erreur_moyenne']:.1%}   moyenne simple de la base (référence naïve)",
                      f"   {seuls['erreur_outil']:.1%}   outil, arguments seuls"]
            if avec:
                lignes.append(f"   {avec['erreur_outil']:.1%}   outil, arguments + contexte (forces, raison, situation)")
                gain = (seuls["erreur_moyenne"] - avec["erreur_outil"]) * 100
                lignes.append(f"\n   gain par rapport à la moyenne simple : {gain:+.1f} points")
            else:
                lignes.append("\n   (l'historique ne contient pas de contexte : pas de comparaison possible)")
            self.var_eval.set("\n".join(lignes))
        self.en_fond("Évaluation en cours (deux passes)…", calcul, fini)

    # ----- onglet Assistant IA (Ollama, local) -----
    def construire_onglet_assistant(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Assistant IA")
        ttk.Label(f, wraplength=1150, justify="left", text=(
            "L'assistant utilise une IA qui tourne sur cet ordinateur (Ollama) : rien n'est envoyé à l'extérieur. "
            "Il ne calcule pas l'estimation, qui reste faite par l'algorithme. Il propose des notes de rapport de "
            "force à partir d'une description libre (que vous validez), et explique une estimation en français."
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
        self.btn_appliquer = ttk.Button(ligne, text="Appliquer à l'onglet Estimer", command=self.appliquer_contexte,
                                        state="disabled")
        self.btn_appliquer.pack(side="left", padx=6)
        self.var_raison_ia = tk.StringVar()
        ttk.Label(ligne, textvariable=self.var_raison_ia).pack(side="left", padx=8)
        self.tab_propositions = self.tableau(cadre1, [("force", "Force", 190), ("note", "Note proposée", 125),
                                                     ("args", "Justification de l'IA", 700)], hauteur=5)
        self.proposition = None

        cadre2 = ttk.LabelFrame(f, text="2. Expliquer la dernière estimation (onglet Estimer)", padding=8)
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
        self.ctx_estimer.definir(forces, self.proposition["raison"] or raison_actuelle)
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
            messagebox.showwarning("Assistant IA", "Fais d'abord une estimation dans l'onglet Estimer.")
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
    def tableau(self, parent, colonnes, hauteur=10):
        cadre = ttk.Frame(parent)
        cadre.pack(fill="both", expand=True)
        t = ttk.Treeview(cadre, columns=[c for c, _, _ in colonnes], show="headings", height=hauteur)
        for c, titre, larg in colonnes:
            t.heading(c, text=titre)
            t.column(c, width=larg, stretch=(c == "args"), anchor="w")
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
