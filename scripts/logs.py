# logs.py — Affiche les logs du bot WishAI
#
#   ./wish logs           → dernières 30 lignes (INFO+)
#   ./wish logs erreurs   → erreurs uniquement
#   ./wish logs repair    → vérifie et répare les dépendances cassées

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from bot_logger import (
    lire_derniers_logs, afficher_resume,
    verifier_et_reparer_tous, LOG_BOT, LOG_ERREURS, LOGS_DIR
)

mode = sys.argv[1] if len(sys.argv) > 1 else ""

print()

if mode == "repair":
    print("  Vérification et réparation des dépendances...\n")
    ok, repares, echecs = verifier_et_reparer_tous()
    print(f"\n  OK : {ok}  |  Réparés : {repares}  |  Échecs : {echecs}")
    if echecs:
        print(f"  Détails : {LOG_ERREURS}")

elif mode == "erreurs":
    lignes = lire_derniers_logs(n=50, niveau_min="ERROR")
    if not lignes:
        print("  Aucune erreur dans les logs.")
    else:
        print(f"  Erreurs récentes ({len(lignes)}) :\n")
        for l in lignes:
            print(f"  {l}")

else:
    lignes = lire_derniers_logs(n=40, niveau_min="INFO")
    if not lignes:
        print("  Aucun log disponible.")
        print(f"  (dossier : {LOGS_DIR})")
    else:
        print(f"  Derniers logs ({len(lignes)}) :\n")
        for l in lignes:
            print(f"  {l}")
    print(f"\n  Fichiers : {LOGS_DIR}")
    print("  ./wish logs erreurs   → erreurs uniquement")
    print("  ./wish logs repair    → réparer les dépendances")

print()
