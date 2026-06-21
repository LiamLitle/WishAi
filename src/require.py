# require.py — installe les dependances WishAI (une seule fois)
#
# Systeme de lock : si tout est deja installe, demarre en < 1 seconde.
# Reinstalle uniquement ce qui manque ou si les paquets ont change.

import subprocess, sys, os, json, hashlib

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_FILE = os.path.join(ROOT, "deps.lock")

# ── Paquets requis : { nom_import : paquet_pip } ────────────────
PAQUETS_BASE = {
    "psutil"        : "psutil",
    "numpy"         : "numpy",
    "safetensors"   : "safetensors",
    "requests"      : "requests",
    "datasets"      : "datasets",
    "huggingface_hub": "huggingface_hub",
}

# torch est verifie separement (version CPU vs CUDA)
TORCH_CPU  = "torch torchvision torchaudio"
TORCH_CUDA = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

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
    """Retourne True si le paquet est importable."""
    try:
        __import__(nom_import)
        return True
    except ImportError:
        return False


def pip_install(commande):
    """Lance pip install et retourne True si succes."""
    r = subprocess.run(
        f"{sys.executable} -m pip install {commande}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if r.returncode != 0:
        # Afficher l erreur seulement en cas d echec
        print(f"  [ERREUR pip] {r.stderr.decode(errors='ignore').strip()[-300:]}")
        return False
    return True


def empreinte_env():
    """Hash unique de l environnement : python + GPU + liste des paquets."""
    gpu = gpu_nvidia()
    contenu = f"{sys.version}|gpu={gpu}|{sorted(PAQUETS_BASE.keys())}|{sorted(PAQUETS_GPU.keys())}"
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


# ── Avertissement si hors virtualenv ────────────────────────────
def _dans_venv():
    """Retourne True si on tourne dans un virtualenv ou conda."""
    if sys.prefix != sys.base_prefix:
        return True
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return True
    return False

if not _dans_venv():
    print()
    print("  /!\\ ATTENTION : pas de virtualenv detecte.")
    print("  Les paquets vont etre installes dans Python GLOBAL.")
    print("  Ca peut casser d'autres projets sur ce PC.")
    print()
    print("  Conseil : cree un venv d'abord :")
    print("    python -m venv .venv")
    print("    .venv\\Scripts\\activate    (Windows)")
    print("    source .venv/bin/activate   (Linux/Mac)")
    print()
    rep = input("  Continuer quand meme ? [o/N] > ").strip().lower()
    if rep != "o":
        print("  Annule. Cree ton venv puis relance go.py ou quick.py.")
        sys.exit(0)
    print()

# ── Verification rapide via lock file ───────────────────────────
empreinte, gpu = empreinte_env()
lock = lire_lock()

if lock.get("empreinte") == empreinte:
    # Lock valide : on verifie juste torch + 2-3 paquets cles en import
    if importer_ok("torch") and importer_ok("safetensors") and importer_ok("psutil"):
        print("  dependances OK")
        sys.exit(0)

# ── Installation necessaire ──────────────────────────────────────
print("\n  WishAI — verification des dependances\n")

manquants = []

# Torch
if not importer_ok("torch"):
    print("  torch manquant — installation...")
    if gpu:
        print("  GPU NVIDIA detecte -> version CUDA")
        pip_install(TORCH_CUDA)
    else:
        print("  pas de GPU -> version CPU")
        pip_install(TORCH_CPU)
else:
    print("  torch : OK")

# Paquets de base
for nom_import, pip_name in PAQUETS_BASE.items():
    if not importer_ok(nom_import):
        manquants.append(pip_name)

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
if erreurs or not importer_ok("torch"):
    print("\n  ATTENTION : paquets toujours manquants : " + str(erreurs))
    print("  Lance manuellement : pip install torch safetensors psutil requests")
    sys.exit(1)

# Tout est la -> sauvegarder le lock
ecrire_lock(empreinte)
print("\n  tout est installe OK\n")
