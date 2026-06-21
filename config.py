# config.py — Gestion et configuration de WishAI
#
#   python config.py
#
# ============================================================

import os, sys, shutil, json, glob, time, subprocess

ROOT     = os.path.dirname(os.path.abspath(__file__))
SRC      = os.path.join(ROOT, "src")
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR  = os.path.join(ROOT, "data")
LOCK_FILE = os.path.join(ROOT, "deps.lock")

# ── Helpers ─────────────────────────────────────────────────────

def sep(titre=""):
    if titre:
        print(f"\n  {'='*56}")
        print(f"  {titre}")
        print(f"  {'='*56}")
    else:
        print(f"  {'─'*56}")

def ok(msg):   print(f"  OK  {msg}")
def err(msg):  print(f"  ERR {msg}")
def info(msg): print(f"  >>  {msg}")

def confirmer(question):
    r = input(f"  {question} [o/N] > ").strip().lower()
    return r == "o"

def choisir_modele(prompt="Choix"):
    modeles = lister_modeles_dispo()
    if not modeles:
        err("Aucun modele trouve dans model/")
        return None
    for i, (nom, meta) in enumerate(modeles.items(), 1):
        params = meta.get("params", "?")
        val    = meta.get("val_loss", "?")
        date   = meta.get("date", "?")
        taille = meta.get("taille_mo", 0)
        print(f"  [{i}] {nom:<20} {params}  val={val}  {date}  ({taille:.0f} Mo)")
    choix = input(f"  {prompt} > ").strip()
    try:
        idx = int(choix) - 1
        if 0 <= idx < len(modeles):
            return list(modeles.keys())[idx]
    except ValueError:
        pass
    err("Choix invalide")
    return None

def lister_modeles_dispo():
    """Retourne dict {nom: metadata} pour tous les modeles trouves."""
    modeles = {}
    for pt in sorted(glob.glob(os.path.join(MODEL_DIR, "*", "modele.pt"))):
        nom = os.path.basename(os.path.dirname(pt))
        meta = {}
        try:
            import torch
            data = torch.load(pt, map_location="cpu", weights_only=False)
            arch = data.get("architecture", data.get("hyperparams", {}))
            entr = data.get("entrainement", {})
            nb   = arch.get("nb_params", arch.get("n_embd", 0))
            meta["params"]   = f"{nb/1e6:.1f}M" if nb > 1000 else "?"
            meta["val_loss"] = entr.get("val_loss", data.get("val_loss", "?"))
            meta["date"]     = entr.get("date", "?")
        except Exception:
            meta["params"] = "?"
        meta["taille_mo"] = os.path.getsize(pt) / 1e6
        modeles[nom] = meta
    return modeles

def taille_dossier(path):
    """Taille totale d un dossier en Mo."""
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except Exception:
                pass
    return total / 1e6


# ════════════════════════════════════════════════════════════════
#  1 — LISTER LES MODELES
# ════════════════════════════════════════════════════════════════

def lister_modeles():
    sep("MODELES ENTRAINES")
    modeles = lister_modeles_dispo()
    if not modeles:
        info("Aucun modele trouve. Lance go.py ou quick.py pour en creer un.")
        return
    total_mo = 0
    for nom, meta in modeles.items():
        params   = meta.get("params", "?")
        val      = meta.get("val_loss", "?")
        date     = meta.get("date", "?")
        taille   = meta.get("taille_mo", 0)
        total_mo += taille
        # Verifier si safetensors existe aussi
        st = os.path.exists(os.path.join(MODEL_DIR, nom, "modele.safetensors"))
        formats = ".pt" + (" + .safetensors" if st else "")
        print(f"\n  Nom     : {nom}")
        print(f"  Params  : {params}   Val Loss : {val}   Date : {date}")
        print(f"  Taille  : {taille:.1f} Mo   Formats : {formats}")
    sep()
    print(f"  Total : {len(modeles)} modele(s)  |  {total_mo:.1f} Mo sur le disque")


# ════════════════════════════════════════════════════════════════
#  2 — SUPPRIMER UN MODELE
# ════════════════════════════════════════════════════════════════

def supprimer_modele():
    sep("SUPPRIMER UN MODELE")
    nom = choisir_modele("Modele a supprimer")
    if not nom:
        return
    dossier = os.path.join(MODEL_DIR, nom)
    taille  = taille_dossier(dossier)
    print(f"\n  Modele : {nom}  ({taille:.1f} Mo)")
    if confirmer("Confirmer la suppression ?"):
        shutil.rmtree(dossier)
        ok(f"{nom} supprime ({taille:.1f} Mo liberes)")
    else:
        info("Annule.")


# ════════════════════════════════════════════════════════════════
#  3 — SUPPRIMER TOUS LES MODELES
# ════════════════════════════════════════════════════════════════

def supprimer_tous_modeles():
    sep("SUPPRIMER TOUS LES MODELES")
    modeles = lister_modeles_dispo()
    if not modeles:
        info("Aucun modele a supprimer.")
        return
    total = taille_dossier(MODEL_DIR)
    print(f"\n  {len(modeles)} modele(s) trouves  |  {total:.1f} Mo")
    for nom in modeles:
        print(f"    - {nom}")
    print()
    if confirmer("Supprimer TOUS les modeles ? (irreversible)"):
        for nom in modeles:
            shutil.rmtree(os.path.join(MODEL_DIR, nom), ignore_errors=True)
        # Garder le dossier model/ mais vider active.json
        active = os.path.join(MODEL_DIR, "active.json")
        if os.path.exists(active):
            os.remove(active)
        ok(f"{len(modeles)} modele(s) supprimes  |  {total:.1f} Mo liberes")
    else:
        info("Annule.")


# ════════════════════════════════════════════════════════════════
#  4 — VOIR LES HYPERPARAMETRES D'UN MODELE
# ════════════════════════════════════════════════════════════════

def voir_hyperparams():
    sep("HYPERPARAMETRES D'UN MODELE")
    nom = choisir_modele()
    if not nom:
        return
    pt = os.path.join(MODEL_DIR, nom, "modele.pt")
    try:
        import torch
        data = torch.load(pt, map_location="cpu", weights_only=False)
        arch = data.get("architecture", data.get("hyperparams", {}))
        entr = data.get("entrainement", {})
        print(f"\n  Modele : {nom}")
        sep()
        print(f"  Architecture")
        print(f"    Parametres  : {arch.get('nb_params', '?'):,}  ({arch.get('nb_params', 0)/1e6:.1f}M)")
        print(f"    Couches     : {arch.get('n_layer', '?')}")
        print(f"    Tetes       : {arch.get('n_head', '?')}")
        print(f"    Embedding   : {arch.get('n_embd', '?')}")
        print(f"    Contexte    : {arch.get('block_size', '?')} tokens")
        print(f"    Vocab BPE   : {arch.get('vocab_size', '?')} tokens")
        print(f"    Dropout     : {arch.get('dropout', '?')}")
        sep()
        print(f"  Entrainement")
        print(f"    Preset      : {entr.get('preset', '?')}")
        print(f"    Val Loss    : {entr.get('val_loss', '?')}")
        print(f"    Iterations  : {entr.get('iterations', '?'):,}" if entr.get('iterations') else f"    Iterations  : ?")
        print(f"    Appareil    : {entr.get('device', '?').upper()}")
        print(f"    Date        : {entr.get('date', '?')}")
    except ImportError:
        err("torch non installe — lance d abord go.py pour installer les dependances")
    except Exception as e:
        err(f"Impossible de lire le modele : {e}")


# ════════════════════════════════════════════════════════════════
#  5 — RENOMMER UN MODELE
# ════════════════════════════════════════════════════════════════

def renommer_modele():
    sep("RENOMMER UN MODELE")
    nom = choisir_modele("Modele a renommer")
    if not nom:
        return
    nouveau = input(f"  Nouveau nom > ").strip()
    if not nouveau:
        err("Nom vide — annule.")
        return
    import re
    nouveau = re.sub(r'[^a-zA-Z0-9_\-]', '_', nouveau)
    dest = os.path.join(MODEL_DIR, nouveau)
    if os.path.exists(dest):
        err(f"Un modele '{nouveau}' existe deja.")
        return
    os.rename(os.path.join(MODEL_DIR, nom), dest)
    # Mettre a jour active.json si necessaire
    active = os.path.join(MODEL_DIR, "active.json")
    if os.path.exists(active):
        try:
            with open(active) as f:
                a = json.load(f)
            if a.get("model") == nom:
                a["model"] = nouveau
                with open(active, "w") as f:
                    json.dump(a, f)
        except Exception:
            pass
    ok(f"{nom}  ->  {nouveau}")


# ════════════════════════════════════════════════════════════════
#  6 — DUPLIQUER UN MODELE
# ════════════════════════════════════════════════════════════════

def dupliquer_modele():
    sep("DUPLIQUER UN MODELE")
    nom = choisir_modele("Modele a dupliquer")
    if not nom:
        return
    copie = input(f"  Nom de la copie [{nom}_copie] > ").strip()
    if not copie:
        copie = nom + "_copie"
    import re
    copie = re.sub(r'[^a-zA-Z0-9_\-]', '_', copie)
    dest = os.path.join(MODEL_DIR, copie)
    if os.path.exists(dest):
        err(f"Un modele '{copie}' existe deja.")
        return
    src = os.path.join(MODEL_DIR, nom)
    taille = taille_dossier(src)
    print(f"  Copie de {nom} -> {copie}  ({taille:.1f} Mo)...")
    shutil.copytree(src, dest)
    ok(f"Copie creee : {copie}")


# ════════════════════════════════════════════════════════════════
#  7 — EXPORTER UN MODELE
# ════════════════════════════════════════════════════════════════

def exporter_modele():
    sep("EXPORTER UN MODELE")
    nom = choisir_modele("Modele a exporter")
    if not nom:
        return
    dest_defaut = os.path.join(os.path.expanduser("~"), "Desktop", nom)
    dest = input(f"  Dossier destination [{dest_defaut}] > ").strip()
    if not dest:
        dest = dest_defaut
    if os.path.exists(dest):
        err(f"Le dossier '{dest}' existe deja.")
        return
    src = os.path.join(MODEL_DIR, nom)
    taille = taille_dossier(src)
    print(f"  Export de {nom} vers {dest}  ({taille:.1f} Mo)...")
    shutil.copytree(src, dest)
    # Copier aussi le tokenizer (necessaire pour generer du texte)
    tok = os.path.join(ROOT, "tokenizer.json")
    if os.path.exists(tok):
        shutil.copy2(tok, dest)
        ok("tokenizer.json copie avec le modele")
    ok(f"Modele exporte -> {dest}")


# ════════════════════════════════════════════════════════════════
#  8 — SUPPRIMER LE CACHE BPE
# ════════════════════════════════════════════════════════════════

def supprimer_cache_bpe():
    sep("SUPPRIMER LE CACHE BPE")
    caches = glob.glob(os.path.join(DATA_DIR, "**", "bpe_cache.pt"), recursive=True)
    caches += glob.glob(os.path.join(DATA_DIR, "bpe_cache.pt"))
    caches = list(set(caches))
    if not caches:
        info("Aucun cache BPE trouve.")
        return
    total = sum(os.path.getsize(c) / 1e6 for c in caches)
    for c in caches:
        print(f"    {c}  ({os.path.getsize(c)/1e6:.1f} Mo)")
    print()
    if confirmer(f"Supprimer {len(caches)} cache(s)  ({total:.1f} Mo) ?"):
        for c in caches:
            os.remove(c)
        ok(f"Cache(s) supprimes — sera recree au prochain lancement")
    else:
        info("Annule.")


# ════════════════════════════════════════════════════════════════
#  9 — VOIR LES DONNEES DISPONIBLES
# ════════════════════════════════════════════════════════════════

def voir_donnees():
    sep("DONNEES D'ENTRAINEMENT")
    if not os.path.exists(DATA_DIR):
        info("Dossier data/ inexistant.")
        return
    trouves = False
    for racine, dossiers, fichiers in os.walk(DATA_DIR):
        for f in fichiers:
            if f.endswith(".txt"):
                chemin = os.path.join(racine, f)
                taille = os.path.getsize(chemin) / 1e6
                rel    = os.path.relpath(chemin, ROOT)
                # Compter les lignes rapidement
                try:
                    with open(chemin, encoding="utf-8", errors="ignore") as fh:
                        lignes = sum(1 for _ in fh)
                except Exception:
                    lignes = 0
                print(f"\n  Fichier : {rel}")
                print(f"  Taille  : {taille:.2f} Mo  |  {lignes:,} lignes")
                trouves = True
    if not trouves:
        info("Aucun fichier .txt trouve dans data/")
        info("Lance go.py pour telecharger des donnees, ou quick.py pour creer un texte de demo.")


# ════════════════════════════════════════════════════════════════
#  10 — SUPPRIMER LES DONNEES DE DEMO
# ════════════════════════════════════════════════════════════════

def supprimer_donnees_demo():
    sep("SUPPRIMER LES DONNEES DE DEMO")
    demo = os.path.join(DATA_DIR, "data.txt")
    if not os.path.exists(demo):
        info("Pas de fichier de demo (data/data.txt).")
        return
    taille = os.path.getsize(demo) / 1e6
    print(f"  Fichier : data/data.txt  ({taille:.2f} Mo)")
    if confirmer("Supprimer le texte de demo ?"):
        os.remove(demo)
        # Supprimer aussi le cache BPE associe
        cache = os.path.join(DATA_DIR, "bpe_cache.pt")
        if os.path.exists(cache):
            os.remove(cache)
            ok("Cache BPE associe supprime aussi")
        ok("data/data.txt supprime")
    else:
        info("Annule.")


# ════════════════════════════════════════════════════════════════
#  11 — REGENERER LE TOKENIZER
# ════════════════════════════════════════════════════════════════

def regenerer_tokenizer():
    sep("REGENERER LE TOKENIZER")
    tok = os.path.join(ROOT, "tokenizer.json")
    if os.path.exists(tok):
        taille = os.path.getsize(tok) / 1e6
        print(f"  tokenizer.json existant  ({taille:.2f} Mo)")
        if not confirmer("Supprimer et regenerer depuis les donnees actuelles ?"):
            info("Annule.")
            return
        os.remove(tok)
        ok("tokenizer.json supprime")
    # Supprimer aussi tous les caches BPE (ils sont lies au tokenizer)
    caches = glob.glob(os.path.join(DATA_DIR, "**", "bpe_cache.pt"), recursive=True)
    for c in caches:
        os.remove(c)
        ok(f"Cache BPE supprime : {c}")
    print()
    print("  Regeneration du tokenizer...")
    subprocess.run([sys.executable, os.path.join(SRC, "tokenizer.py")], cwd=ROOT)


# ════════════════════════════════════════════════════════════════
#  12 — SUPPRIMER LES DEPENDANCES (reset deps.lock)
# ════════════════════════════════════════════════════════════════

def supprimer_dependances():
    sep("REINITIALISER LES DEPENDANCES")
    print("  Cette option supprime le fichier deps.lock.")
    print("  Au prochain lancement de go.py ou quick.py,")
    print("  toutes les dependances seront verifiees et reinstallees si besoin.")
    print()
    if os.path.exists(LOCK_FILE):
        if confirmer("Supprimer deps.lock et forcer la reinstallation ?"):
            os.remove(LOCK_FILE)
            ok("deps.lock supprime — reinstallation au prochain lancement")
        else:
            info("Annule.")
    else:
        info("Aucun deps.lock trouve (les dependances n ont pas encore ete installees).")


# ════════════════════════════════════════════════════════════════
#  13 — RESET COMPLET
# ════════════════════════════════════════════════════════════════

def reset_complet():
    sep("RESET COMPLET")
    print("  Ceci va supprimer :")
    print("    - Tous les modeles entraines  (model/*/)")
    print("    - Tous les caches BPE         (bpe_cache.pt)")
    print("    - Le tokenizer                (tokenizer.json)")
    print("    - Les donnees de demo         (data/data.txt)")
    print("    - Le fichier de lock          (deps.lock)")
    print("    - Les fichiers de session     (session.json, control.json)")
    print()
    if not confirmer("RESET COMPLET — tout supprimer ? (irreversible)"):
        info("Annule.")
        return
    if not confirmer("Vraiment ? Il n y a pas de retour en arriere."):
        info("Annule.")
        return

    # Modeles
    for dossier in glob.glob(os.path.join(MODEL_DIR, "*")):
        if os.path.isdir(dossier) and os.path.basename(dossier) != "active.json":
            shutil.rmtree(dossier, ignore_errors=True)
    active = os.path.join(MODEL_DIR, "active.json")
    if os.path.exists(active):
        os.remove(active)
    ok("Modeles supprimes")

    # Caches BPE
    for c in glob.glob(os.path.join(DATA_DIR, "**", "bpe_cache.pt"), recursive=True):
        os.remove(c)
    ok("Caches BPE supprimes")

    # Tokenizer
    tok = os.path.join(ROOT, "tokenizer.json")
    if os.path.exists(tok):
        os.remove(tok)
    ok("Tokenizer supprime")

    # Demo
    demo = os.path.join(DATA_DIR, "data.txt")
    if os.path.exists(demo):
        os.remove(demo)
    ok("Donnees de demo supprimees")

    # Lock
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    ok("deps.lock supprime")

    # Session / control
    for f in ["session.json", "control.json", "control.json.tmp"]:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            os.remove(p)
    ok("Fichiers de session supprimes")

    print()
    print("  WishAI est revenu a son etat initial.")
    print("  Lance go.py pour recommencer depuis zero.")


# ════════════════════════════════════════════════════════════════
#  14 — INFOS PC / GPU
# ════════════════════════════════════════════════════════════════

def infos_pc():
    sep("INFOS PC / GPU")
    # Python
    print(f"  Python     : {sys.version.split()[0]}  ({sys.executable})")

    # torch
    try:
        import torch
        print(f"  PyTorch    : {torch.__version__}")
        if torch.cuda.is_available():
            nom_gpu  = torch.cuda.get_device_name(0)
            vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_ver = torch.version.cuda
            print(f"  GPU        : {nom_gpu}")
            print(f"  VRAM       : {vram:.1f} Go")
            print(f"  CUDA       : {cuda_ver}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print(f"  GPU        : Apple Silicon (MPS)")
        else:
            print(f"  GPU        : aucun (CPU uniquement)")
    except ImportError:
        print(f"  PyTorch    : non installe")

    # RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  RAM        : {ram.total/1e9:.1f} Go total  |  {ram.available/1e9:.1f} Go libre")
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logic = psutil.cpu_count(logical=True)
        print(f"  CPU        : {cpu_count} coeurs physiques  ({cpu_logic} logiques)")
    except ImportError:
        print(f"  RAM/CPU    : psutil non installe")

    # Disque
    try:
        import shutil as _sh
        total, used, free = _sh.disk_usage(ROOT)
        print(f"  Disque     : {total/1e9:.0f} Go total  |  {free/1e9:.1f} Go libre")
    except Exception:
        pass

    # Espace modeles
    if os.path.exists(MODEL_DIR):
        taille_models = taille_dossier(MODEL_DIR)
        print(f"  Modeles    : {taille_models:.1f} Mo utilises dans model/")


# ════════════════════════════════════════════════════════════════
#  15 — TESTER PYTORCH + GPU
# ════════════════════════════════════════════════════════════════

def tester_pytorch():
    sep("TEST PYTORCH + GPU")
    try:
        import torch
        ok(f"torch importe  (version {torch.__version__})")
    except ImportError:
        err("torch non installe — lance go.py pour installer les dependances")
        return

    # Test CPU basique
    try:
        a = __import__("torch").randn(100, 100)
        b = __import__("torch").randn(100, 100)
        c = a @ b
        ok(f"Multiplication matricielle CPU  (100x100)")
    except Exception as e:
        err(f"Erreur CPU : {e}")
        return

    # Test GPU
    import torch
    if torch.cuda.is_available():
        try:
            t0 = time.time()
            a = torch.randn(1000, 1000, device="cuda")
            b = torch.randn(1000, 1000, device="cuda")
            c = a @ b
            torch.cuda.synchronize()
            duree = (time.time() - t0) * 1000
            ok(f"Multiplication matricielle GPU  (1000x1000)  en {duree:.1f} ms")
            vram_utilise = torch.cuda.memory_allocated() / 1e6
            ok(f"VRAM utilisee pour le test : {vram_utilise:.1f} Mo")
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as e:
            err(f"Erreur GPU : {e}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            a = torch.randn(500, 500, device="mps")
            b = torch.randn(500, 500, device="mps")
            c = a @ b
            ok("Multiplication matricielle MPS (Apple Silicon)")
        except Exception as e:
            err(f"Erreur MPS : {e}")
    else:
        info("Pas de GPU — entrainement possible sur CPU (plus lent)")

    # Test safetensors
    try:
        import safetensors
        ok(f"safetensors importe  (version {safetensors.__version__})")
    except ImportError:
        err("safetensors non installe")

    # Test psutil
    try:
        import psutil
        psutil.virtual_memory()
        ok("psutil OK")
    except Exception:
        err("psutil non installe")

    print()
    print("  Tout est pret pour l entrainement !")


# ════════════════════════════════════════════════════════════════
#  16 — LOGS DU DERNIER ENTRAINEMENT
# ════════════════════════════════════════════════════════════════

def voir_logs():
    sep("LOGS DU DERNIER ENTRAINEMENT")

    # Trouver le modele actif ou le plus recent
    log_file = None
    active = os.path.join(MODEL_DIR, "active.json")
    if os.path.exists(active):
        try:
            with open(active) as f:
                nom = json.load(f).get("model")
            if nom:
                candidate = os.path.join(MODEL_DIR, nom, "log_active.json")
                if os.path.exists(candidate):
                    log_file = candidate
        except Exception:
            pass

    # Fallback : log le plus recent
    if not log_file:
        logs = glob.glob(os.path.join(MODEL_DIR, "*", "log_active.json"))
        if logs:
            log_file = max(logs, key=os.path.getmtime)

    if not log_file:
        info("Aucun log trouve. Lance un entrainement d abord.")
        return

    try:
        with open(log_file, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        err(f"Impossible de lire le log : {e}")
        return

    print(f"  Modele   : {data.get('nom_modele', '?')}")
    print(f"  Statut   : {data.get('status', '?')}")
    print(f"  Debut    : {data.get('debut', '?')}")
    print(f"  Mode     : {data.get('mode', '?')}")
    hp = data.get("hyperparams", {})
    if hp:
        print(f"  Preset   : {data.get('preset', '?')}")
        print(f"  Params   : {hp.get('nb_params', 0)/1e6:.1f}M")
        print(f"  Device   : {hp.get('device', '?').upper()}")
    steps = data.get("steps", [])
    if steps:
        sep()
        print(f"  Etapes evaluees : {len(steps)}")
        # Afficher les 5 dernieres
        print(f"  {'Etape':<10} {'Train Loss':<14} {'Val Loss':<14} {'Temps'}")
        for s in steps[-5:]:
            t = s.get("temps", 0)
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            print(f"  {s.get('step', 0):<10,} {s.get('train_loss', 0):<14.4f} {s.get('val_loss', 0):<14.4f} {h}h{m:02d}m")
        best = min(steps, key=lambda s: s.get("val_loss", 999))
        print(f"\n  Meilleure val loss : {best.get('val_loss', '?'):.4f}  a l etape {best.get('step', '?'):,}")


# ════════════════════════════════════════════════════════════════
#  MENU PRINCIPAL
# ════════════════════════════════════════════════════════════════

MENU = [
    ("MODELES",      None),
    ("Lister les modeles",                 lister_modeles),
    ("Supprimer un modele",                supprimer_modele),
    ("Supprimer TOUS les modeles",         supprimer_tous_modeles),
    ("Voir les hyperparametres",           voir_hyperparams),
    ("Renommer un modele",                 renommer_modele),
    ("Dupliquer un modele",                dupliquer_modele),
    ("Exporter un modele",                 exporter_modele),
    ("DONNEES",      None),
    ("Voir les donnees disponibles",       voir_donnees),
    ("Supprimer les donnees de demo",      supprimer_donnees_demo),
    ("Supprimer le cache BPE",             supprimer_cache_bpe),
    ("Regenerer le tokenizer",             regenerer_tokenizer),
    ("SYSTEME",      None),
    ("Infos PC / GPU",                     infos_pc),
    ("Tester PyTorch + GPU",               tester_pytorch),
    ("Logs du dernier entrainement",       voir_logs),
    ("Reinitialiser les dependances",      supprimer_dependances),
    ("Reset complet (tout effacer)",       reset_complet),
]

def afficher_menu():
    print("\n")
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           WishAI  —  Configuration          ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    num = 0
    for titre, fn in MENU:
        if fn is None:
            print(f"\n  ── {titre} {'─'*(44-len(titre))}")
        else:
            num += 1
            print(f"  [{num:>2}]  {titre}")
    print()
    print("  [ 0]  Quitter")
    print()

def main():
    while True:
        afficher_menu()
        choix = input("  Choix > ").strip()
        if choix == "0" or choix.lower() in ("q", "quit", "exit"):
            print("\n  A bientot !\n")
            break
        try:
            idx = int(choix) - 1
        except ValueError:
            print("  Choix invalide.\n")
            continue

        # Construire la liste des fonctions dans l ordre
        fonctions = [fn for _, fn in MENU if fn is not None]
        if 0 <= idx < len(fonctions):
            print()
            try:
                fonctions[idx]()
            except KeyboardInterrupt:
                print("\n  Interruption.")
            input("\n  Appuie sur Entree pour revenir au menu...")
        else:
            print("  Choix invalide.\n")

if __name__ == "__main__":
    main()
