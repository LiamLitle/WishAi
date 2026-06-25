import os
import sys
import glob
import time
import json
import shutil
import subprocess

from .utils import (
    sep, ok, err, info, confirmer, taille_dossier,
    ROOT, MODEL_DIR, DATA_DIR, LOCK_FILE
)

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
#  13 — DESINSTALLER LES DEPENDANCES
# ════════════════════════════════════════════════════════════════

def desinstaller_dependances():
    sep("DESINSTALLER LES DEPENDANCES")
    req = os.path.join(ROOT, "requirements.txt")
    if not os.path.exists(req):
        err("requirements.txt introuvable.")
        return

    # Lire les noms de paquets (sans version ni commentaires)
    import re
    paquets = []
    with open(req, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#"):
                continue
            # Extraire le nom uniquement (avant ==, >=, <=, ~=, !=, [, ;)
            nom = re.split(r'[>=<!~\[\];]', ligne)[0].strip()
            if nom:
                paquets.append(nom)

    if not paquets:
        info("Aucun paquet trouve dans requirements.txt.")
        return

    print("  Paquets qui seront desinstalles :")
    for p in paquets:
        print(f"    - {p}")
    print()
    print("  Note : deps.lock n est PAS supprime.")
    print("         Pour reinstaller automatiquement, supprime-le ensuite (option 17).")
    print()

    if not confirmer(f"Desinstaller {len(paquets)} paquet(s) ?"):
        info("Annule.")
        return

    print()
    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y"] + paquets,
        cwd=ROOT
    )
    if result.returncode == 0:
        ok("Paquets desinstalles.")
    else:
        err("Certains paquets n ont pas pu etre desinstalles (peut-etre deja absents).")


# ════════════════════════════════════════════════════════════════
#  14 — RESET COMPLET
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
    tok = os.path.join(ROOT, "system", "tokenizer.json")
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
    for f in ["session.json", "control.json", "control.json.tmp", "TEMP"]:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
    ok("Fichiers et dossiers temporaires supprimes")

    print()
    print("  WishAI est revenu a son etat initial.")
    print("  Lance go.py pour recommencer depuis zero.")
