# go.py -- lance tout WishAI en une commande
# python go.py

import os, sys, subprocess, time, json, webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")

CONTROL_FILE = os.path.join(ROOT, "control.json")

def run(script):
    return subprocess.run([sys.executable, os.path.join(SRC, script)], cwd=ROOT).returncode

def run_args(script, args):
    return subprocess.run([sys.executable, os.path.join(SRC, script)] + args, cwd=ROOT).returncode

def bg(script):
    return subprocess.Popen([sys.executable, os.path.join(SRC, script)], cwd=ROOT)

def donnees_existent():
    def non_vide(p):
        return os.path.exists(p) and os.path.getsize(p) > 1024
    for langue in ["en", "fr", "multi"]:
        if non_vide(os.path.join(ROOT, "data", langue, "data.txt")):
            return True
    return non_vide(os.path.join(ROOT, "data", "data.txt"))

def lire_control():
    try:
        with open(CONTROL_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"commande": "run"}

def ecrire_control(commande, **extra):
    data = {"commande": commande, "timestamp": time.time()}
    data.update(extra)
    # Ecriture atomique : on ecrit dans un .tmp puis on remplace d'un coup,
    # pour qu'un lecteur ne tombe jamais sur un control.json a moitie ecrit.
    tmp = CONTROL_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, CONTROL_FILE)


print("\n  WishAI\n")

# dependances (installe uniquement ce qui manque, < 1s si deja installe)
run("require.py")

# -- Demarrage anticipe du serveur dashboard ----------------------------
_dash_url_file = os.path.join(ROOT, "dashboard_url.json")
if os.path.exists(_dash_url_file):
    try: os.remove(_dash_url_file)
    except Exception: pass
_dash_proc = bg("dashboard.py")

def _wait_dash_url(timeout=10):
    for _ in range(timeout * 4):
        if os.path.exists(_dash_url_file):
            try:
                with open(_dash_url_file, encoding="utf-8") as _f:
                    return json.load(_f)
            except Exception:
                pass
        time.sleep(0.25)
    return None

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
        print("\n  Demarrage du serveur en cours...")
        _dash_info = _wait_dash_url(timeout=10)
        if _dash_info and _dash_info.get("lib_url"):
            webbrowser.open(_dash_info["lib_url"])
        else:
            _lib = os.path.join(ROOT, "library.html")
            webbrowser.open("file:///" + _lib.replace("\\", "/"))
        print("\n  Bibliotheque ouverte dans le navigateur.")
        input("  Appuie sur Entree quand tu as fini... ")
        print("\n  Retour a go.py...")

# tokenizer -- se reentraine automatiquement si le dataset a change
print("\n  Verification du tokenizer...")
print()
run("tokenizer.py")

# monitor -- en arriere-plan (dashboard deja lance plus haut)
bg("monitor.py")
_wait_dash_url(timeout=8)
time.sleep(1)

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
        print("  CONDITIONS OK -- REPRISE DE L'ENTRAINEMENT")
        print("="*56 + "\n")
        time.sleep(2)
        continue

    break

print("\n  Termine. Pour generer du texte : python src/generate.py\n")
