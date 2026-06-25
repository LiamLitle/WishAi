import os
import glob
import subprocess
import sys

from .utils import (
    sep, ok, info, confirmer,
    DATA_DIR, ROOT, SRC
)

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
#  11 — REGENERER LE TOKENIZER
# ════════════════════════════════════════════════════════════════

def regenerer_tokenizer():
    sep("REGENERER LE TOKENIZER")
    tok = os.path.join(ROOT, "system", "tokenizer.json")
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
