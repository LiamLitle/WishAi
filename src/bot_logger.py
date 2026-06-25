"""
bot_logger.py — Système de logs pour WishAI

Fichiers générés dans system/logs/ :
  bot.log          Tout : info, warnings, erreurs
  erreurs.log      Erreurs et fatals uniquement (diagnostic rapide)

Rotation automatique : max 2 Mo par fichier, 3 fichiers conservés.
"""

import os, sys, json, time, datetime, threading, subprocess

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT, "system", "logs")

LOG_BOT     = os.path.join(LOGS_DIR, "bot.log")
LOG_ERREURS = os.path.join(LOGS_DIR, "erreurs.log")

MAX_SIZE    = 2 * 1024 * 1024   # 2 Mo
MAX_GARDER  = 3                  # fichiers conservés par rotation

_lock = threading.Lock()

NIVEAUX = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "FATAL": 4}


# ================================================================
# ROTATION
# ================================================================

def _rotation(chemin):
    """Renomme .log → .1.log → .2.log etc., supprime le plus vieux."""
    if not os.path.exists(chemin) or os.path.getsize(chemin) < MAX_SIZE:
        return
    base = chemin[:-4]  # enlève .log
    for i in range(MAX_GARDER - 1, 0, -1):
        src = f"{base}.{i}.log"
        dst = f"{base}.{i+1}.log"
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
    if os.path.exists(f"{base}.{MAX_GARDER}.log"):
        os.remove(f"{base}.{MAX_GARDER}.log")
    os.rename(chemin, f"{base}.1.log")


def _ecrire(chemin, ligne):
    _rotation(chemin)
    with open(chemin, "a", encoding="utf-8") as f:
        f.write(ligne + "\n")


# ================================================================
# API PUBLIQUE
# ================================================================

def log(niveau: str, message: str, source: str = "bot"):
    """
    Enregistre un événement dans les logs.

    Paramètres :
        niveau  : "INFO" | "WARNING" | "ERROR" | "FATAL"
        message : texte libre
        source  : origine (ex: "telecharger", "require", "auto_repair")
    """
    niveau = niveau.upper()
    ts     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ligne  = f"[{ts}] [{niveau:<7}] [{source}] {message}"

    os.makedirs(LOGS_DIR, exist_ok=True)

    with _lock:
        _ecrire(LOG_BOT, ligne)
        if NIVEAUX.get(niveau, 0) >= NIVEAUX["ERROR"]:
            _ecrire(LOG_ERREURS, ligne)

    # Affiche les erreurs/fatals dans le terminal
    if NIVEAUX.get(niveau, 0) >= NIVEAUX["WARNING"]:
        print(f"  [{niveau}] {message}")


def log_telechargement(source_key: str, nom: str, mo: float, ok: bool, detail: str = ""):
    """Log spécifique à un téléchargement (résumé structuré)."""
    statut = "OK" if ok else "ECHEC"
    msg    = f"[{statut}] {source_key} — {nom} — {mo:.1f} Mo"
    if detail:
        msg += f" — {detail}"
    log("INFO" if ok else "ERROR", msg, source="telecharger")

    # Écrit aussi dans downloads.log (JSON Lines)
    entry = {
        "ts":     datetime.datetime.now().isoformat(),
        "source": source_key,
        "nom":    nom,
        "mo":     round(mo, 2),
        "ok":     ok,
        "detail": detail,
    }
    dl_log = os.path.join(LOGS_DIR, "downloads.log")
    os.makedirs(LOGS_DIR, exist_ok=True)
    with _lock:
        _rotation(dl_log)
        with open(dl_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def lire_derniers_logs(n: int = 30, niveau_min: str = "INFO") -> list:
    """Retourne les n dernières lignes de bot.log filtrées par niveau."""
    seuil = NIVEAUX.get(niveau_min.upper(), 0)
    lignes = []
    if not os.path.exists(LOG_BOT):
        return []
    try:
        with open(LOG_BOT, encoding="utf-8") as f:
            for l in f:
                for niv, val in NIVEAUX.items():
                    if f"[{niv}" in l and val >= seuil:
                        lignes.append(l.rstrip())
                        break
    except Exception:
        pass
    return lignes[-n:]


def afficher_resume():
    """Affiche un résumé des erreurs récentes dans le terminal."""
    erreurs = lire_derniers_logs(n=10, niveau_min="ERROR")
    if not erreurs:
        print("  Aucune erreur récente dans les logs.")
        return
    print(f"\n  Dernières erreurs ({len(erreurs)}) :\n")
    for l in erreurs:
        print(f"  {l}")
    print(f"\n  Logs complets : {LOG_BOT}")


# ================================================================
# AUTO-RÉPARATION DES DÉPENDANCES
# ================================================================

PAQUETS_IMPORT = {
    "datasets":       "datasets",
    "huggingface_hub": "huggingface_hub",
    "torch":          "torch",
    "safetensors":    "safetensors",
    "psutil":         "psutil",
    "requests":       "requests",
    "numpy":          "numpy",
    "gguf":           "gguf",
    "onnx":           "onnx",
    "onnxscript":     "onnxscript",
}


def _tester_import(nom_import: str) -> tuple:
    """
    Tente d'importer un module.
    Retourne (ok: bool, statut: str) — statut: "ok" | "absent" | "corrompu"
    """
    try:
        __import__(nom_import)
        return True, "ok"
    except ImportError:
        return False, "absent"
    except (OSError, Exception):
        return False, "corrompu"


def auto_repair(paquet_import: str, pip_name: str = None) -> bool:
    """
    Détecte si un paquet est absent/corrompu et le réinstalle.

    Retourne True si le paquet est OK après réparation.
    """
    if pip_name is None:
        pip_name = paquet_import

    ok, statut = _tester_import(paquet_import)

    if ok:
        return True

    log("WARNING", f"Paquet '{paquet_import}' : {statut} — réinstallation...", source="auto_repair")
    print(f"  Réinstallation de '{pip_name}'...")

    try:
        if statut == "corrompu":
            # Désinstaller d'abord
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", pip_name],
                capture_output=True
            )

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", pip_name],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            log("ERROR",
                f"Réinstallation '{pip_name}' échouée : {result.stderr[-200:]}",
                source="auto_repair")
            return False

        # Vérifie que l'import marche maintenant
        ok2, _ = _tester_import(paquet_import)
        if ok2:
            log("INFO", f"Paquet '{paquet_import}' réparé avec succès.", source="auto_repair")
            print(f"  '{pip_name}' réinstallé avec succès.")
            return True
        else:
            log("ERROR", f"'{paquet_import}' toujours non importable après réinstallation.", source="auto_repair")
            return False

    except Exception as e:
        log("ERROR", f"Erreur lors de la réinstallation de '{pip_name}' : {e}", source="auto_repair")
        return False


def verifier_et_reparer_tous():
    """
    Vérifie tous les paquets critiques et répare ceux qui sont cassés.
    Retourne (nb_ok, nb_repares, nb_echecs).
    """
    nb_ok = nb_repares = nb_echecs = 0

    log("INFO", "Vérification complète des dépendances...", source="auto_repair")

    for imp, pip_n in PAQUETS_IMPORT.items():
        ok, statut = _tester_import(imp)
        if ok:
            nb_ok += 1
            continue
        log("WARNING", f"'{imp}' : {statut}", source="auto_repair")
        repare = auto_repair(imp, pip_n)
        if repare:
            nb_repares += 1
        else:
            nb_echecs += 1

    log("INFO",
        f"Bilan : {nb_ok} OK, {nb_repares} réparés, {nb_echecs} échecs",
        source="auto_repair")

    return nb_ok, nb_repares, nb_echecs


if __name__ == "__main__":
    print("\n  Vérification et réparation des dépendances WishAI\n")
    ok, repares, echecs = verifier_et_reparer_tous()
    print(f"\n  Résultat : {ok} OK  |  {repares} réparés  |  {echecs} échecs")
    if echecs:
        print("  Consulte les logs pour les détails :", LOG_ERREURS)
    print()
