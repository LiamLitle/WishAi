import os
import shutil
import json
import sys
import glob

from .utils import (
    sep, ok, err, info, confirmer,
    choisir_modele, lister_modeles_dispo, taille_dossier,
    MODEL_DIR, ROOT
)

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
    if not os.path.exists(pt):
        pt = os.path.join(MODEL_DIR, nom, "best_model.pt")
        if not os.path.exists(pt):
            pt = os.path.join(MODEL_DIR, nom, "checkpoint.pt")
            
    try:
        import torch
        data = torch.load(pt, map_location="cpu", weights_only=False)
        arch = data.get("architecture", data.get("hyperparams", {}))
        if not arch and "log_data" in data:
            arch = data["log_data"].get("hyperparams", {})
        entr = data.get("entrainement", {})
        print(f"\n  Modele : {nom}")
        sep()
        print(f"  Architecture")
        _nb = arch.get('nb_params')
        _nb_str = f"{_nb:,}" if isinstance(_nb, int) else "?"
        _nb_mo  = f"{_nb/1e6:.1f}M" if isinstance(_nb, int) else "?"
        print(f"    Parametres  : {_nb_str}  ({_nb_mo})")
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
        _iters = entr.get('iterations')
        print(f"    Iterations  : {_iters:,}" if isinstance(_iters, int) else f"    Iterations  : ?")
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
    sep("EXPORTER UN MODELE (GGUF / ONNX)")
    nom = choisir_modele("Modele a exporter")
    if not nom:
        return
        
    print("\n  [1/2] Generation des formats GGUF et ONNX...")
    import subprocess
    try:
        # Appelle le script d'export
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export.py"), nom], check=True)
    except subprocess.CalledProcessError:
        err("L'exportation GGUF/ONNX a echoue.")
        return
        
    print("\n  [2/2] Finalisation du dossier...")
    dest = os.path.join(MODEL_DIR, nom)
    
    # Copier aussi le tokenizer (necessaire pour generer du texte)
    tok = os.path.join(ROOT, "system", "tokenizer.json")
    if os.path.exists(tok):
        shutil.copy2(tok, dest)
        ok("tokenizer.json ajoute au dossier du modele")
        
    ok(f"Modele '{nom}' exporte (GGUF + ONNX + PT) dans : {dest}")
