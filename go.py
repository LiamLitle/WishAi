# go.py -- lance tout WishAI en une commande
# python go.py

import os, sys, subprocess, time, json

ROOT = os.path.dirname(os.path.abspath(__file__))
# Centralisation du cache (__pycache__)
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(ROOT, "cache", "pycache")
sys.pycache_prefix = os.path.join(ROOT, "cache", "pycache")
SRC  = os.path.join(ROOT, "src")

CONTROL_FILE = os.path.join(ROOT, "control.json")

def run(script):
    return subprocess.run([sys.executable, os.path.join(SRC, script)], cwd=ROOT).returncode

def run_args(script, args):
    return subprocess.run([sys.executable, os.path.join(SRC, script)] + args, cwd=ROOT).returncode

def bg(script):
    return subprocess.Popen([sys.executable, os.path.join(SRC, script)], cwd=ROOT)

def donnees_existent():
    for langue in ["en", "fr", "multi"]:
        if os.path.exists(os.path.join(ROOT, "data", langue, "data.txt")):
            return True
    return os.path.exists(os.path.join(ROOT, "data", "data.txt"))

def lire_control():
    try:
        with open(CONTROL_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"commande": "run"}

def ecrire_control(commande, **extra):
    data = {"commande": commande, "timestamp": time.time()}
    data.update(extra)
    with open(CONTROL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


print("\n  WishAI\n")

# dependances en premier
run("require.py")

# -- Niveau de protection -----------------------------------------------
sys.path.insert(0, SRC)
from protection import NIVEAUX, DEFAUT

# Le système utilise par défaut le niveau STANDARD défini dans protection.py

# initialise control.json a "run"
ecrire_control("run")

# donnees d'entrainement
if not donnees_existent():
    print("\n  Aucune donnee d'entrainement trouvee.")
    print("  Sans donnees, l'IA n'a rien a lire.\n")
    print("  [O] Telecharger maintenant")
    print("  [n] Plus tard -- je lancerai python src/telecharger.py moi-meme")
    print()
    choix_dl = input("  Choix [O] > ").strip().lower()
    if choix_dl != "n":
        run_args("telecharger.py", [])
        if not donnees_existent():
            print("\n  Toujours pas de donnees -- arret.")
            sys.exit(1)
    else:
        print("\n  Lance python src/telecharger.py quand tu es pret, puis relance go.py.\n")
        sys.exit(0)
else:
    print("\n  [?] Des donnees existent deja.")
    print("  Veux-tu ouvrir la bibliotheque pour en rajouter ? [o/N]")
    choix_dl = input("  Choix [N] > ").strip().lower()
    if choix_dl == "o":
        run_args("telecharger.py", [])
        print("\n  Retour a go.py...")

# tokenizer
tok = os.path.join(ROOT, "tokenizer.json")
if not os.path.exists(tok):
    print("\n  tokenizer.json introuvable -- lancement du tokenizer...")
    print()
    run("tokenizer.py")

# monitor, dashboard, bouton -- en arriere-plan
bg("monitor.py")
bg("dashboard.py")
bg("btn_dashboard.py")
time.sleep(2)

# -- Signal nouvelle session pour le dashboard -------------------------
with open(os.path.join(ROOT, "session.json"), "w", encoding="utf-8") as _sf:
    json.dump({"id": str(int(time.time())), "status": "starting", "ts": time.time()}, _sf)

# -- Boucle d'entrainement (reprise auto phase 3) ----------------------
while True:
    ecrire_control("run")
    run("nanogpt_bpe.py")

    ctrl     = lire_control()
    commande = ctrl.get("commande", "run")

    if commande == "reprendre":
        print("\n" + "="*56)
        print("  CONDITIONS OK -- 