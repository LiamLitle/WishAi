# protection.py — niveaux de sécurité pour l'entraînement WishAI
# Utilisé par nanogpt_bpe.py, monitor.py et go.py
#
# 3 phases par niveau :
#   Phase 1 — alerte_ram       → alerte console + ralentit l'entraînement
#   Phase 2 — pause_ram        → pause (nanogpt_bpe attend en boucle)
#   Phase 3 — critique_ram/temp → arrêt propre + reprise automatique via monitor

NIVEAUX = {
    "minim": {
        "label":         "Minim      — machines puissantes (>32 Go RAM)",
        # Phase 1
        "alerte_ram":    85,    # % RAM → alerte + ralentit
        "ralentir_ms":   80,    # ms de pause ajoutés par itération
        # Phase 2
        "pause_ram":     90,    # % RAM → pause temporaire
        # Phase 3
        "critique_ram":  95,    # % RAM → arrêt + reprise auto
        "critique_temp": 90,    # °C    → arrêt + reprise auto
        # Seuils de reprise (après pause ou critique)
        "resume_ram":    82,    # % RAM → OK pour reprendre
        "resume_temp":   78,    # °C    → OK pour reprendre
    },
    "standard": {
        "label":         "Standard   — recommandé (16-32 Go RAM)",
        "alerte_ram":    75,
        "ralentir_ms":   150,
        "pause_ram":     82,
        "critique_ram":  92,
        "critique_temp": 90,
        "resume_ram":    70,
        "resume_temp":   75,
    },
    "protection": {
        "label":         "Protection — PC moyen ou laptop (8-16 Go RAM)",
        "alerte_ram":    70,
        "ralentir_ms":   200,
        "pause_ram":     78,
        "critique_ram":  85,
        "critique_temp": 90,
        "resume_ram":    65,
        "resume_temp":   72,
    },
    "max": {
        "label":         "Max        — PC ancien ou très limité (<8 Go RAM)",
        "alerte_ram":    60,
        "ralentir_ms":   300,
        "pause_ram":     70,
        "critique_ram":  80,
        "critique_temp": 89,
        "resume_ram":    55,
        "resume_temp":   70,
    },
}

DEFAUT = "standard"


def charger_seuils(config_path=""):
    """
    Retourne le dict de seuils du niveau Standard par défaut.
    Le système config.json a été supprimé pour simplifier l'utilisation.
    """
    return NIVEAUX[DEFAUT], DEFAUT
