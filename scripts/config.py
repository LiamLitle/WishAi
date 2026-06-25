# config.py — Gestion et configuration de WishAI
#
#   ./wish config
#
# ============================================================

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from Config.modeles import (
    lister_modeles, supprimer_modele, supprimer_tous_modeles,
    voir_hyperparams, renommer_modele, dupliquer_modele, exporter_modele
)
from Config.donnees import (
    voir_donnees, supprimer_donnees_demo, supprimer_cache_bpe, regenerer_tokenizer
)
from Config.systeme import (
    infos_pc, tester_pytorch, voir_logs, supprimer_dependances,
    desinstaller_dependances, reset_complet
)

# ════════════════════════════════════════════════════════════════
#  MENU PRINCIPAL
# ════════════════════════════════════════════════════════════════

MENU = [
    ("MODELES",      None),
    ("Lister les modeles",                 lister_modeles),
    ("Supprimer un modele",                supprimer_modele),
    ("Supprimer TOUS les modeles",         supprimer_tous_modeles),
    ("Voir les hyperparametres",           voir_hyperparams),
    ("Renommer un modele",                 renommer_modele),
    ("Dupliquer un modele",                dupliquer_modele),
    ("Exporter un modele",                 exporter_modele),
    ("DONNEES",      None),
    ("Voir les donnees disponibles",       voir_donnees),
    ("Supprimer les donnees de demo",      supprimer_donnees_demo),
    ("Supprimer le cache BPE",             supprimer_cache_bpe),
    ("Regenerer le tokenizer",             regenerer_tokenizer),
    ("SYSTEME",      None),
    ("Infos PC / GPU",                     infos_pc),
    ("Tester PyTorch + GPU",               tester_pytorch),
    ("Logs du dernier entrainement",       voir_logs),
    ("Reinitialiser les dependances",      supprimer_dependances),
    ("Desinstaller les dependances",       desinstaller_dependances),
    ("Reset complet (tout effacer)",       reset_complet),
]

def afficher_menu():
    print("\n")
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           WishAI  —  Configuration          ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    num = 0
    for titre, fn in MENU:
        if fn is None:
            print(f"\n  ── {titre} {'─'*(44-len(titre))}")
        else:
            num += 1
            print(f"  [{num:>2}]  {titre}")
    print()
    print("  [ 0]  Quitter")
    print()

def main():
    while True:
        afficher_menu()
        choix = input("  Choix > ").strip()
        if choix == "0" or choix.lower() in ("q", "quit", "exit"):
            print("\n  A bientot !\n")
            break
        try:
            idx = int(choix) - 1
        except ValueError:
            print("  Choix invalide.\n")
            continue

        # Construire la liste des fonctions dans l ordre
        fonctions = [fn for _, fn in MENU if fn is not None]
        if 0 <= idx < len(fonctions):
            print()
            try:
                fonctions[idx]()
            except KeyboardInterrupt:
                print("\n  Interruption.")
            input("\n  Appuie sur Entree pour revenir au menu...")
        else:
            print("  Choix invalide.\n")

if __name__ == "__main__":
    main()
