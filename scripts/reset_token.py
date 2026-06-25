# token.py — Réinitialisation et réentraînement du tokenizer
#
#   ./wish token          → supprime tokenizer.json et réentraîne
#   ./wish token reset    → supprime uniquement (réentraînement au prochain go)

import os, sys, subprocess, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
SYS  = os.path.join(ROOT, "system")
sys.path.insert(0, SRC)

MODE_RESET_ONLY = "reset" in sys.argv

TOK_SYSTEME = os.path.join(SYS, "tokenizer.json")
TOK_MODELES = glob.glob(os.path.join(ROOT, "model", "*", "tokenizer.json"))

print()
print("  ══════════════════════════════════════════")
print("  RÉINITIALISATION DU TOKENIZER — WishAI")
print("  ══════════════════════════════════════════\n")

# ── Fichiers à supprimer ────────────────────────────────────────
fichiers = []
if os.path.exists(TOK_SYSTEME):
    fichiers.append(TOK_SYSTEME)
fichiers.extend(TOK_MODELES)

if not fichiers:
    print("  Aucun tokenizer.json trouvé — rien à réinitialiser.\n")
else:
    print("  Fichiers détectés :\n")
    for f in fichiers:
        taille = os.path.getsize(f) / 1000
        rel    = os.path.relpath(f, ROOT)
        print(f"    • {rel:<45} {taille:.0f} Ko")

    print()
    while True:
        rep = input("  Supprimer ces fichiers ? [o/n] > ").strip().lower()
        if rep in ("o", "n"):
            break

    if rep != "o":
        print("  Annulé.\n")
        sys.exit(0)

    for f in fichiers:
        try:
            os.remove(f)
            print(f"  Supprimé : {os.path.relpath(f, ROOT)}")
        except Exception as e:
            print(f"  Erreur : {f} — {e}")

    print()

# ── Réentraînement immédiat ou plus tard ────────────────────────
if MODE_RESET_ONLY:
    print("  Le tokenizer sera réentraîné automatiquement")
    print("  au prochain lancement de ./wish go ou ./wish quick\n")
    sys.exit(0)

# Cherche les données disponibles
DATA_FILE = None
for candidat in [
    os.path.join(ROOT, "data", "data.txt"),
    os.path.join(ROOT, "data", "fr", "data.txt"),
    os.path.join(ROOT, "data", "en", "data.txt"),
    os.path.join(ROOT, "data", "multi", "data.txt"),
]:
    if os.path.exists(candidat) and os.path.getsize(candidat) > 1000:
        DATA_FILE = candidat
        break

if DATA_FILE is None:
    print("  Aucune donnée d'entraînement trouvée.")
    print("  Lance d'abord : ./wish go  pour télécharger des données.\n")
    sys.exit(0)

taille = os.path.getsize(DATA_FILE) / 1_000_000
print(f"  Données : {os.path.relpath(DATA_FILE, ROOT)} ({taille:.0f} Mo)")
print("  Lancement du réentraînement BPE...\n")

result = subprocess.run(
    [sys.executable, os.path.join(SRC, "tokenizer.py")],
    cwd=ROOT
)

if result.returncode == 0:
    print("\n  Tokenizer réentraîné avec succès.")
    print(f"  Fichier : system/tokenizer.json\n")
else:
    print("\n  Le réentraînement a rencontré une erreur.")
    print("  Vérifie que les données existent dans data/\n")
