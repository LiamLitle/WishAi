# require.py -- installe les dependances WishAI (une seule fois)
#
# Systeme de lock : si tout est deja installe, demarre en < 1 seconde.
# Reinstalle uniquement ce qui manque ou si les paquets ont change.

import subprocess, sys, os, json, hashlib, time

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_FILE = os.path.join(ROOT, "system", "deps.lock")

# -- Paquets requis --------------------------------------------------
PAQUETS_BASE = {
    "psutil"         : "psutil",
    "numpy"          : "numpy",
    "safetensors"    : "safetensors",
    "requests"       : "requests",
    "datasets"       : "datasets",
    "huggingface_hub": "huggingface_hub",
}

TORCH_PKGS = "torch torchvision torchaudio"
TORCH_CPU  = TORCH_PKGS
TORCH_CUDA = TORCH_PKGS + " --index-url https://download.pytorch.org/whl/cu121"

PAQUETS_GPU = {
    "GPUtil" : "gputil",
    "pynvml" : "pynvml",
}


def gpu_nvidia():
    try:
        return subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        ).returncode == 0
    except Exception:
        return False


def importer_ok(nom_import):
    """Retourne True si le paquet est importable et fonctionnel."""
    try:
        __import__(nom_import)
        return True
    except Exception:
        return False


def torch_status():
    """Retourne 'ok', 'absent' ou 'corrompu'."""
    try:
        __import__("torch")
        return "ok"
    except ImportError:
        return "absent"
    except OSError:
        return "corrompu"
    except Exception:
        return "corrompu"


def check_and_install_julia():
    """Verifie si Julia est installe, et l'installe via winget si besoin sur Windows."""
    try:
        subprocess.run(["julia", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        if sys.platform == "win32":
            print("\n  📥 Julia introuvable. Installation en arriere-plan (cela peut prendre quelques minutes)...")
            try:
                subprocess.run(["winget", "install", "--id", "Julialang.Juliaup", "-e", "--silent", "--accept-package-agreements", "--accept-source-agreements"])
                print("  ✅ Julia installe avec succes !")
                # Laisse un instant a Windows pour mettre a jour le PATH si possible
                time.sleep(2)
            except Exception as e:
                print(f"  ⚠️ Erreur lors de l'installation de Julia : {e}")
                print("     Veuillez installer Julia manuellement depuis julialang.org")
        else:
            print("\n  ⚠️ Julia introuvable. Veuillez l'installer manuellement pour utiliser les estimations avancees.")

    # Installe aussi les paquets Julia
    try:
        print("  Verification des paquets Julia (LsqFit, JSON)...")
        # Pkg.add installe s'il manque, met a jour ou ignore s'il est a jour
        subprocess.run(["julia", "-e", "using Pkg; Pkg.add([\"LsqFit\", \"JSON\"])"], capture_output=True, check=True)
    except Exception:
        pass


def pip_run(commande):
    """Lance une commande pip simple (pas de capture)."""
    subprocess.run(
        sys.executable + " -m pip " + commande,
        shell=True,
    )


def pip_install(commande):
    """Lance pip install avec progression visible. Gere Ctrl+C proprement."""
    proc = subprocess.Popen(
        sys.executable + " -m pip install " + commande,
        shell=True,
        stderr=subprocess.PIPE,
    )
    try:
        _, stderr = proc.communicate()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\n  Installation annulee (Ctrl+C).")
        sys.exit(0)
    if proc.returncode != 0:
        print("  [ERREUR pip] " + stderr.decode(errors="ignore").strip()[-300:])
        return False
    return True


def empreinte_env():
    gpu = gpu_nvidia()
    contenu = (
        sys.version + "|gpu=" + str(gpu)
        + "|" + str(sorted(PAQUETS_BASE.keys()))
        + "|" + str(sorted(PAQUETS_GPU.keys()))
    )
    return hashlib.sha256(contenu.encode()).hexdigest()[:16], gpu


def lire_lock():
    try:
        with open(LOCK_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def ecrire_lock(empreinte):
    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            json.dump({"empreinte": empreinte, "python": sys.version}, f)
    except Exception:
        pass


# -- Avertissement si hors virtualenv --------------------------------
def _dans_venv():
    if sys.prefix != sys.base_prefix:
        return True
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return True
    return False

if not _dans_venv():
    print()
    print("  /!\\ ATTENTION : pas de virtualenv detecte.")
    print("  Les paquets vont etre installes dans Python GLOBAL.")
    print()
    print("  Conseil : cree un venv d'abord :")
    print("    python -m venv .venv")
    print("    .venv\\Scripts\\activate    (Windows)")
    print("    source .venv/bin/activate   (Linux/Mac)")
    print()
    rep = input("  Continuer quand meme ? [o/N] > ").strip().lower()
    if rep != "o":
        print("  Annule.")
        sys.exit(0)
    print()

# -- Verification rapide via lock file -------------------------------
empreinte, gpu = empreinte_env()
lock = lire_lock()

if lock.get("empreinte") == empreinte:
    if importer_ok("torch") and importer_ok("safetensors") and importer_ok("psutil"):
        print("  dependances OK")
        sys.exit(0)

# -- Installation necessaire -----------------------------------------
print("\n  WishAI -- verification des dependances\n")

# Torch
_torch = torch_status()
if _torch == "ok":
    print("  torch : OK")
elif _torch == "corrompu":
    print("  torch corrompu (DLL invalide) -- desinstallation + reinstallation...")
    pip_run("uninstall torch torchvision torchaudio -y")
    if gpu:
        print("  GPU NVIDIA detecte (CUDA)")
        pip_install(TORCH_CUDA)
    else:
        print("  CPU uniquement")
        pip_install(TORCH_CPU)
else:
    print("  torch absent -- installation...")
    if gpu:
        print("  GPU NVIDIA detecte (CUDA)")
        pip_install(TORCH_CUDA)
    else:
        print("  CPU uniquement")
        pip_install(TORCH_CPU)

# Paquets de base
manquants = [pip for nom, pip in PAQUETS_BASE.items() if not importer_ok(nom)]
if manquants:
    print("  installation : " + " ".join(manquants))
    pip_install(" ".join(manquants))
else:
    print("  paquets de base : OK")

# Paquets GPU
if gpu:
    manquants_gpu = [pip for nom, pip in PAQUETS_GPU.items() if not importer_ok(nom)]
    if manquants_gpu:
        print("  installation GPU : " + " ".join(manquants_gpu))
        pip_install(" ".join(manquants_gpu))
    else:
        print("  paquets GPU : OK")

# Verification finale
erreurs = [n for n in PAQUETS_BASE if not importer_ok(n)]
_torch_final = torch_status()
if erreurs or _torch_final != "ok":
    print("\n  ATTENTION : paquets toujours manquants : " + str(erreurs))
    if _torch_final != "ok":
        print("  torch inaccessible -- essaie manuellement :")
        print("    pip uninstall torch torchvision torchaudio -y")
        if gpu:
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("    pip install torch torchvision torchaudio")
    sys.exit(1)

check_and_install_julia()

ecrire_lock(empreinte)
print("\n  tout est installe OK\n")
