import os
import glob
import time

ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC       = os.path.join(ROOT, "src")
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR  = os.path.join(ROOT, "data")
LOCK_FILE = os.path.join(ROOT, "system", "deps.lock")

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

def lister_modeles_dispo():
    """Retourne dict {nom: metadata} pour tous les modeles trouves."""
    modeles = {}
    for dossier in sorted(glob.glob(os.path.join(MODEL_DIR, "*/"))):
        nom = os.path.basename(os.path.dirname(dossier))
        if nom.startswith(".") or not nom:
            continue
            
        # Cherche un fichier PyTorch valide
        fichiers_pt = [
            os.path.join(dossier, "modele.pt"),
            os.path.join(dossier, "best_model.pt"),
            os.path.join(dossier, "checkpoint.pt")
        ]
        pt = next((f for f in fichiers_pt if os.path.exists(f)), None)
        
        if not pt:
            continue
            
        meta = {}
        try:
            import torch
            data = torch.load(pt, map_location="cpu", weights_only=False)
            arch = data.get("architecture", data.get("hyperparams", {}))
            if not arch and "log_data" in data:
                arch = data["log_data"].get("hyperparams", {})
            entr = data.get("entrainement", {})
            nb   = arch.get("nb_params", arch.get("n_embd", 0))
            meta["params"]   = f"{nb/1e6:.1f}M" if nb > 1000 else "?"
            
            # Essayer de lire la val_loss depuis le fichier, ou depuis le checkpoint
            if "log_data" in data and "steps" in data["log_data"] and len(data["log_data"]["steps"]) > 0:
                meta["val_loss"] = data["log_data"]["steps"][-1]["val_loss"]
            else:
                meta["val_loss"] = entr.get("val_loss", data.get("val_loss", "?"))
                
            meta["date"]     = entr.get("date", "?")
            if meta["date"] == "?" and "log_data" in data:
                meta["date"] = data["log_data"].get("last_update", "?")
                if isinstance(meta["date"], float):
                    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta["date"]))
        except Exception:
            meta["params"] = "?"
        meta["taille_mo"] = os.path.getsize(pt) / 1e6
        meta["chemin_pt"] = pt # On garde le chemin exact pour l'export
        modeles[nom] = meta
    return modeles

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
