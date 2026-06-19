# require.py — installe tout ce qu'il faut pour WishAI
# python require.py

import subprocess, sys, platform, os, urllib.request, tempfile


def pip(paquet):
    print(f"  installing {paquet}...")
    r = subprocess.run(f"pip install {paquet}", shell=True)
    if r.returncode != 0:
        print(f"\n  échec sur {paquet} — essaie de lancer la commande toi-même")
        sys.exit(1)

def gpu_nvidia():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


print("\n  WishAI — installation des dépendances\n")

gpu = gpu_nvidia()

if gpu:
    print("  GPU NVIDIA détecté → version CUDA\n")
    subprocess.run(
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121",
        shell=True
    )
else:
    print("  pas de GPU NVIDIA → version CPU (plus lent mais ça marche)\n")
    subprocess.run("pip install torch torchvision torchaudio", shell=True)

print()

for p in ["psutil", "requests", "numpy", "safetensors", "datasets", "huggingface_hub"]:
    pip(p)

if gpu:
    for p in ["gputil", "pynvml"]:
        pip(p)

print("\n  tout est installé ✅")
print("  pour démarrer : python go.py\n")
