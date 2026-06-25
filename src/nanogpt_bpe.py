"""
================================================================
  WISHAI BPE — Entraîne ta propre IA de zéro
  Auteur  : Liam
  Version : 1.0.0 — Open Source
================================================================

COMMENT UTILISER CE PROJET (dans l'ordre) :

  Étape 1 — Télécharge des données d'entraînement
    python telecharger.py

  Étape 2 — Entraîne le tokenizer (une seule fois !)
    python tokenizer.py

  Étape 3 — Lance l'entraînement de ton IA
    python nanogpt_bpe.py

  Étape 4 — Surveille l'entraînement en direct
    Ouvre dashboard.html dans ton navigateur
    (sous VS Code : clic droit → "Open with Live Server")

  Étape 5 — Parle avec ton IA !
    ./wish chat --terminal

----------------------------------------------------------------
QU'EST-CE QU'UN TOKENIZER BPE ?

  L'IA ne lit pas les lettres directement — elle travaille
  avec des morceaux de mots appelés "tokens".
  Le BPE (Byte Pair Encoding) fusionne les séquences les plus
  fréquentes pour former des tokens plus grands :
    'h','e' → 'he' → 'the' → ' the' → ' the '  etc.

  Résultat avec un bloc de 256 tokens :
    Char-level : ~50 mots de contexte
    BPE 4000   : ~180 mots de contexte  (3× plus !)

----------------------------------------------------------------
QU'EST-CE QU'UN TRANSFORMER ?

  C'est l'architecture de base de GPT, Llama, Claude...
  Elle fonctionne avec de l'"attention" : chaque mot peut
  regarder tous les autres pour mieux comprendre le contexte.

  Structure d'un bloc :
    Texte encodé
      → Multi-Head Attention (chaque token regarde les autres)
      → + connexion résiduelle + LayerNorm
      → Feed-Forward Network (raisonnement local)
      → + connexion résiduelle + LayerNorm
    Représentation enrichie

  On empile N blocs → le modèle "réfléchit" de plus en plus
  profondément à chaque couche.

----------------------------------------------------------------
PRÉREQUIS (à installer une seule fois) :

  python require.py
================================================================
"""

import math
import torch
import os, sys, time, json, gc, glob
from safetensors.torch import save_file as safe_save_file

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import re as _re
import psutil
import subprocess as _sp
import signal as _sig

from tokenizer import TokenizerBPE, TOKENIZER_FILE, VOCAB_SIZE as _VOCAB_SIZE
from model import WishAI_BPE, ConfigModele

# ================================================================
#  CONFIGURATIONS PRÊTES À L'EMPLOI (PRESETS)
# ================================================================
#
#  Plutôt que de configurer chaque paramètre à la main, tu peux
#  choisir un preset adapté à ton matériel.
#
#  GLOSSAIRE DES PARAMÈTRES :
#
#  batch_size   → Nombre d'exemples traités EN PARALLÈLE à chaque étape
#                 ↑ plus grand = plus rapide, mais utilise plus de VRAM
#                 Exemple : 16 = l'IA apprend sur 16 textes à la fois
#
#  block_size   → Longueur du contexte en tokens BPE
#                 C'est la "fenêtre de lecture" de l'IA
#                 128 tokens ≈  90 mots  |  256 ≈ 180 mots  |  512 ≈ 360 mots
#
#  n_embd       → Dimension des vecteurs internes (l'"espace" de l'IA)
#                 ↑ plus grand = l'IA est plus expressive, mais plus lente
#                 Exemple : 512 = chaque token est représenté par 512 nombres
#
#  n_head       → Nombre de "têtes" d'attention en parallèle
#                 Chaque tête apprend à regarder le texte sous un angle différent
#                 OBLIGATOIRE : doit diviser n_embd exactement (ex: 512/8 = 64 ✅)
#
#  n_layer      → Nombre de blocs Transformer empilés
#                 ↑ plus de couches = l'IA "réfléchit" plus profondément
#                 Exemple : 12 couches = architecture similaire à GPT-2 small
#
#  dropout      → Régularisation : éteint X% des neurones aléatoirement
#                 Empêche l'IA de mémoriser par cœur (overfitting)
#                 Valeur typique : 0.1 à 0.3 — 0.2 est un bon choix
#
#  learning_rate → Vitesse d'apprentissage ("la taille des pas")
#                  Trop grand → instable  |  Trop petit → très lent
#                  Valeur typique : 3e-4 (= 0.0003)
# ================================================================

PRESETS = {
    "NANO": {
        "emoji"           : "🐢",
        "description"     : "CPU ou GPU < 4 Go — petit modèle, entraînement rapide",
        "vram_min_go"     : 0,
        "batch_size"      : 4,
        "grad_accum_steps": 2,
        "block_size"      : 128,
        "n_embd"          : 128,
        "n_head"          : 4,
        "n_layer"         : 4,
        "dropout"         : 0.1,
        "learning_rate"   : 3e-4,
        "nb_params_approx": "~2M",
    },
    "SMALL": {
        "emoji"           : "🚀",
        "description"     : "GPU 4-6 Go — bon équilibre vitesse et qualité",
        "vram_min_go"     : 4,
        "batch_size"      : 4,
        "grad_accum_steps": 4,
        "block_size"      : 256,
        "n_embd"          : 256,
        "n_head"          : 4,
        "n_layer"         : 6,
        "dropout"         : 0.15,
        "learning_rate"   : 3e-4,
        "nb_params_approx": "~10M",
    },
    "MINI": {
        "emoji"           : "⚡",
        "description"     : "GPU 4-8 Go — 20M params, bon point de départ pour tester",
        "vram_min_go"     : 4,
        "batch_size"      : 4,
        "grad_accum_steps": 4,
        "block_size"      : 256,
        "n_embd"          : 384,
        "n_head"          : 6,
        "n_layer"         : 10,
        "dropout"         : 0.15,
        "learning_rate"   : 3e-4,
        "nb_params_approx": "~20M",
    },
    "MEDIUM": {
        "emoji"           : "⚡",
        "description"     : "GPU 6-8 Go — meilleur rapport qualité / ressources",
        "vram_min_go"     : 6,
        "batch_size"      : 4,
        "grad_accum_steps": 4,
        "block_size"      : 256,
        "n_embd"          : 512,
        "n_head"          : 8,
        "n_layer"         : 12,
        "dropout"         : 0.2,
        "learning_rate"   : 3e-4,
        "nb_params_approx": "~40M",
    },
    "LARGE": {
        "emoji"           : "🧠",
        "description"     : "GPU 12+ Go — modèle puissant, entraînement long",
        "vram_min_go"     : 12,
        "batch_size"      : 4,
        "grad_accum_steps": 8,
        "block_size"      : 512,
        "n_embd"          : 768,
        "n_head"          : 12,
        "n_layer"         : 16,
        "dropout"         : 0.2,
        "learning_rate"   : 1e-4,
        "nb_params_approx": "~85M",
    },
}

# ================================================================
#  VÉRIFICATION DU PC
# ================================================================

def verifier_pc():
    """
    Analyse le matériel disponible et recommande un preset.
    Retourne : (device, gpu_nom, vram_go, ram_go, preset_recommande)
    """
    print("\n" + "="*62)
    print("  🖥️  VÉRIFICATION DE TON PC")
    print("="*62)

    # ── GPU et VRAM ──────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_nom = torch.cuda.get_device_name(0)
        vram_go = torch.cuda.get_device_properties(0).total_memory / 1e9
        device  = "cuda"
        print(f"  ✅ GPU      : {gpu_nom}")
        print(f"  ✅ VRAM     : {vram_go:.1f} Go")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Apple Silicon : memoire unifiee (pas de VRAM dediee).
        # On estime la part utilisable depuis la RAM systeme pour la reco de preset.
        gpu_nom = "Apple Silicon (MPS)"
        ram_totale = psutil.virtual_memory().total / 1e9
        vram_go = ram_totale * 0.6
        device  = "mps"
        print(f"  ✅ GPU      : {gpu_nom}")
        print(f"  ✅ Mémoire  : unifiée (~{vram_go:.1f} Go utilisables sur {ram_totale:.0f} Go)")
    else:
        gpu_nom = "Pas de GPU (CPU uniquement)"
        vram_go = 0.0
        device  = "cpu"
        print("  ⚠️  Aucun GPU détecté — entraînement sur CPU")
        print("     L'entraînement sera lent.")
        print("     → Utilise le preset NANO pour rester raisonnable.")

    # ── RAM système ──────────────────────────────────────────────
    ram_go = psutil.virtual_memory().total / 1e9
    print(f"  ✅ RAM      : {ram_go:.1f} Go")

    # ── Recommandation automatique ───────────────────────────────
    if   vram_go >= 12: recommande = "LARGE"
    elif vram_go >= 6 : recommande = "MEDIUM"
    elif vram_go >= 3 : recommande = "SMALL"
    else              : recommande = "NANO"

    p = PRESETS[recommande]
    print(f"\n  💡 Config recommandée pour ton PC : {p['emoji']} {recommande}")
    print(f"     {p['description']}")
    print(f"     {p['nb_params_approx']} paramètres")
    print("="*62)

    return device, gpu_nom, vram_go, ram_go, recommande

# ================================================================
#  LIMITES DE SÉCURITÉ MÉMOIRE
#  Pilotées par le niveau de protection choisi dans go.py.
# ================================================================

LIMITE_VRAM_GO = 5.0    # recalculée selon la VRAM réelle
LIMITE_RAM_GO  = 14.0   # recalculée selon la RAM réelle

# ── Chargement du niveau de protection ───────────────────────────
_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_SRC_DIR)
_SYS_DIR      = os.path.join(_ROOT_DIR, "system")
CONTROL_FILE  = os.path.join(_SYS_DIR, "control.json")
CONFIG_FILE   = os.path.join(_ROOT_DIR, "config.json")

sys.path.insert(0, _SRC_DIR)
from protection import charger_seuils as _charger_seuils
_PROT_SEUILS, _PROT_NIVEAU = _charger_seuils(CONFIG_FILE)

# Flag phase 1 — activé par check_securite(), lu dans la boucle principale
_ralentir = False

def _ecrire_control(commande, **extra):
    data = {"commande": commande, "timestamp": time.time()}
    data.update(extra)
    # Ecriture atomique (.tmp + os.replace) pour eviter un fichier corrompu.
    try:
        tmp = CONTROL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, CONTROL_FILE)
    except Exception:
        pass

def _lire_control():
    try:
        with open(CONTROL_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"commande": "run"}

def configurer_limites_memoire(vram_total_go, ram_total_go):
    """Calcule la limite VRAM à 85% du total."""
    global LIMITE_VRAM_GO, LIMITE_RAM_GO
    if vram_total_go > 0:
        LIMITE_VRAM_GO = round(vram_total_go * 0.85, 1)
    # RAM gérée via pourcentage dans check_securite() — LIMITE_RAM_GO garde
    # son rôle de seuil absolu de dernier recours
    LIMITE_RAM_GO = round(ram_total_go * 0.96, 1)

def lire_vram_systeme():
    """Lit la VRAM utilisée (en Go) via nvidia-smi."""
    try:
        r = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        return float(r.stdout.strip()) / 1024
    except Exception:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

def lire_temp_gpu():
    """Lit la température du GPU (en °C) via nvidia-smi."""
    try:
        r = _sp.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        return int(r.stdout.strip())
    except Exception:
        return 0

def get_memoire():
    """Retourne un dict avec toutes les infos mémoire et CPU."""
    vram_utilise = lire_vram_systeme()
    vram_total   = (torch.cuda.get_device_properties(0).total_memory / 1e9
                    if torch.cuda.is_available() else 6.0)
    ram          = psutil.virtual_memory()
    return {
        "vram_go"      : round(vram_utilise, 2),
        "vram_total_go": round(vram_total, 2),
        "ram_go"       : round(ram.used / 1e9, 2),
        "ram_total_go" : round(ram.total / 1e9, 2),
        "vram_pct"     : round(vram_utilise / vram_total * 100, 1) if vram_total > 0 else 0,
        "ram_pct"      : round(ram.percent, 1),
        "cpu_pct"      : round(psutil.cpu_percent(interval=None), 1),
    }

def check_securite():
    """
    Vérifie RAM, VRAM et température selon le niveau de protection choisi.

    3 phases :
      Phase 1 — alerte    : met _ralentir=True, la boucle principale ajoute un sleep
      Phase 2 — pause     : reste en vie, attend signal "resume" de monitor.py
      Phase 3 — critique  : sauvegarde checkpoint, écrit "arret_critique", retourne False
                            → go.py relance automatiquement depuis le checkpoint

    Retourne True (continuer) ou False (arrêt, go.py prend le relais).
    """
    global _ralentir
    s    = _PROT_SEUILS
    mem  = get_memoire()
    temp = lire_temp_gpu()
    ram_pct = mem["ram_pct"]

    # ── Phase 3 : critique — arrêt + reprise automatique ─────────
    crit_ram  = ram_pct >= s["critique_ram"]
    crit_temp = temp > 0 and temp >= s["critique_temp"]
    if crit_ram or crit_temp:
        raison = []
        if crit_ram:  raison.append(f"RAM {ram_pct:.1f}% ≥ {s['critique_ram']}%")
        if crit_temp: raison.append(f"GPU {temp}°C ≥ {s['critique_temp']}°C")
        raison_str = " + ".join(raison)
        print(f"\n🔴 CRITIQUE : {raison_str}")
        print("   Arrêt propre — reprise automatique dès que les conditions s'améliorent")
        # Écriture dans log_active.json (log_data est global)
        log_data["status"]      = "arret_critique"
        log_data["last_update"] = time.time()
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False)
        except Exception:
            pass
        _ecrire_control("arret_critique", raison=raison_str)
        return False  # go.py attend le signal "reprendre" de monitor.py

    # ── Phase 2 : pause — reste en vie, attend reprise ───────────
    pause_ram  = ram_pct >= s["pause_ram"]
    if pause_ram:
        raison = f"RAM {ram_pct:.1f}% ≥ {s['pause_ram']}%"
        print(f"\n⏸️  PAUSE : {raison}")
        print("   En attente de libération mémoire (monitor.py surveille)...")
        log_data["status"]      = "pause"
        log_data["last_update"] = time.time()
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False)
        except Exception:
            pass
        _ecrire_control("pause", raison=raison)

        # Boucle d'attente — monitor.py écrira "resume" quand c'est OK
        while True:
            time.sleep(4)
            ctrl = _lire_control()
            if ctrl.get("commande") == "resume":
                print(f"   ▶️  Reprise — RAM {psutil.virtual_memory().percent:.1f}%  GPU {lire_temp_gpu()}°C")
                log_data["status"]      = "entrainement"
                log_data["last_update"] = time.time()
                try:
                    with open(LOG_FILE, "w", encoding="utf-8") as f:
                        json.dump(log_data, f, ensure_ascii=False)
                except Exception:
                    pass
                _ralentir = False
                break

    # ── Phase 1 : alerte — ralentit via _ralentir ────────────────
    if ram_pct >= s["alerte_ram"]:
        if not _ralentir:
            print(f"\n🟡 ALERTE : RAM {ram_pct:.1f}% ≥ {s['alerte_ram']}% "
                  f"— ralentissement activé (ferme des apps !)")
        _ralentir = True
    else:
        if _ralentir:
            print(f"\n🟢 RAM revenue à {ram_pct:.1f}% — vitesse normale")
        _ralentir = False

    # ── VRAM ─────────────────────────────────────────────────────
    if mem["vram_go"] > LIMITE_VRAM_GO:
        print(f"\n🚨 VRAM {mem['vram_go']:.2f} Go > {LIMITE_VRAM_GO} Go — arrêt propre")
        return False
    elif mem["vram_go"] > LIMITE_VRAM_GO - 0.5:
        torch.cuda.empty_cache()
        time.sleep(3)
    elif mem["vram_go"] > LIMITE_VRAM_GO - 1.0:
        torch.cuda.empty_cache()
        time.sleep(1)

    return True

# ================================================================
#  FONCTIONS UTILITAIRES DE SAISIE
# ================================================================

def demander(question, defaut, type_=str, mini=None, maxi=None):
    """
    Pose une question à l'utilisateur avec une valeur par défaut.
    Appuyer sur Entrée sans rien taper → valeur par défaut utilisée.
    """
    valeur_str = input(f"  {question} [{defaut}] > ").strip()
    if not valeur_str:
        return defaut
    try:
        valeur = type_(valeur_str)
        if mini is not None and valeur < mini:
            print(f"  ⚠️  Minimum = {mini} → valeur ajustée")
            return type_(mini)
        if maxi is not None and valeur > maxi:
            print(f"  ⚠️  Maximum = {maxi} → valeur ajustée")
            return type_(maxi)
        return valeur
    except (ValueError, TypeError):
        print(f"  ⚠️  Valeur invalide → {defaut} conservé")
        return defaut

# ================================================================
#  ASSISTANT DE CONFIGURATION GUIDÉ
#  Pose toutes les questions nécessaires avant de commencer.
# ================================================================

def configurer():
    """
    Guide l'utilisateur à travers toute la configuration.
    Retourne un dict avec tous les paramètres d'entraînement.
    """

    # ────────────────────────────────────────────────────────────
    #  DÉTECTION DU PC
    # ────────────────────────────────────────────────────────────
    device, gpu_nom, vram_go, ram_go, preset_recommande = verifier_pc()
    configurer_limites_memoire(vram_go, ram_go)

    # ────────────────────────────────────────────────────────────
    #  ÉTAPE 1 — NOM DU MODÈLE
    # ────────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  📝  ÉTAPE 1 / 4 — NOM DE TON MODÈLE")
    print("="*62)
    print("  Donne un nom à ton IA.")
    print("  Ce nom sera utilisé pour les fichiers sauvegardés.")
    print("  Exemples : MonIA, GPT_v1, swift2, Jarvis, Zephyr...")
    print()
    nom_saisi  = input("  Nom > ").strip()
    if not nom_saisi:
        nom_saisi = "monIA"
    # Retire les caractères spéciaux pour un nom de fichier propre
    nom_modele = _re.sub(r'[^a-zA-Z0-9_\-]', '_', nom_saisi)
    print(f"  → Nom retenu : {nom_modele}")

    # ────────────────────────────────────────────────────────────
    #  ÉTAPE 2 — TAILLE DU MODÈLE (preset ou custom)
    # ────────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  ⚙️   ÉTAPE 2 / 4 — TAILLE DU MODÈLE")
    print("="*62)
    print("  Choisis une configuration adaptée à ton matériel.")
    print()

    noms_presets = list(PRESETS.keys())
    for i, (nom_p, p) in enumerate(PRESETS.items(), 1):
        recomm = "  ← RECOMMANDÉ POUR TON PC" if nom_p == preset_recommande else ""
        print(f"  [{i}] {p['emoji']} {nom_p:<8} {p['nb_params_approx']:<8} "
              f"{p['description']}{recomm}")
    print("  [6] 🔧 CUSTOM   Je configure chaque paramètre moi-même")
    print()
    print("  Appuie sur Entrée pour utiliser la config recommandée.")

    idx_defaut = noms_presets.index(preset_recommande) + 1
    choix_str  = input(f"  Choix [{idx_defaut}] > ").strip()

    try:
        choix_num = int(choix_str) if choix_str else idx_defaut
    except ValueError:
        choix_num = idx_defaut

    checkpoint_interval = None  # sera défini dans ÉTAPE 3 ou dans le mode Simple custom

    if 1 <= choix_num <= 5:
        # ── Preset sélectionné ───────────────────────────────────
        nom_preset       = noms_presets[choix_num - 1]
        p                = PRESETS[nom_preset]
        batch_size       = p["batch_size"]
        grad_accum_steps = p.get("grad_accum_steps", 1)
        block_size       = p["block_size"]
        n_embd        = p["n_embd"]
        n_head        = p["n_head"]
        n_layer       = p["n_layer"]
        dropout       = p["dropout"]
        learning_rate = p["learning_rate"]
        print(f"\n  ✅ {p['emoji']} Config {nom_preset} sélectionnée")
        print(f"     {p['description']}")
        print(f"     → Environ {p['nb_params_approx']} paramètres")

    else:
        # ── Mode CUSTOM ──────────────────────────────────────────
        nom_preset = "CUSTOM"
        print("\n  🔧 MODE CUSTOM")
        print()
        print("  Combien de paramètres veux-tu pour ton modèle ?")
        print("  (C'est la taille de l'IA — plus c'est grand, plus elle est capable.)")
        print()
        print("     2M   → très rapide, petit modèle (CPU ok)")
        print("    10M   → bon départ, quelques heures")
        print("    40M   → niveau MEDIUM, bonne qualité (6 Go VRAM)")
        print("   100M   → niveau GPT-2 small (8+ Go VRAM)")
        print("   200M   → ambitieux, GPU solide requis")
        print()

        def _parser_params(s):
            s = s.strip().lower().replace(" ", "")
            if s.endswith("m"):
                return int(float(s[:-1]) * 1_000_000)
            if s.endswith("k"):
                return int(float(s[:-1]) * 1_000)
            if s.endswith("b"):
                return int(float(s[:-1]) * 1_000_000_000)
            return int(float(s))

        while True:
            cible_str = input("  Params cibles [40M] > ").strip()
            if not cible_str:
                cible_str = "40M"
            try:
                cible = _parser_params(cible_str)
                if cible < 500_000:
                    print("  ⚠️  Minimum 500k params recommandé.")
                    continue
                if cible > 1_500_000_000:
                    print("  ⚠️  Plus d'1.5 milliard — hors de portée sur machine locale.")
                    continue
                break
            except Exception:
                print("  ❌ Format invalide. Exemples : 2M, 10M, 40M, 100M")

        def _calculer_config(cible):
            import math as _math
            VOCAB = _VOCAB_SIZE
            BLOCK = 256
            best = None
            best_ecart = float("inf")
            for n_layer in [2, 4, 6, 8, 12, 16, 24, 32]:
                a = 12 * n_layer
                b_coef = VOCAB + BLOCK
                disc = b_coef * b_coef + 4 * a * cible
                n_embd_f = (-b_coef + _math.sqrt(disc)) / (2 * a)
                for candidate in sorted(set([
                    max(64, int(n_embd_f // 64) * 64),
                    max(64, (int(n_embd_f // 64) + 1) * 64),
                ])):
                    if candidate > 2048:
                        continue
                    n_head = max([d for d in [1, 2, 4, 8, 12, 16] if candidate % d == 0], default=1)
                    params = 12 * n_layer * candidate ** 2 + (VOCAB + BLOCK) * candidate
                    ecart = abs(params - cible)
                    if ecart < best_ecart:
                        best_ecart = ecart
                        best = (candidate, n_head, n_layer, params)
            return best

        n_embd_calc, n_head_calc, n_layer_calc, params_reel = _calculer_config(cible)

        if params_reel < 5_000_000:
            bs_c, ga_c, bl_c, lr_c, do_c = 4, 2, 128, 3e-4, 0.1
        elif params_reel < 20_000_000:
            bs_c, ga_c, bl_c, lr_c, do_c = 4, 4, 256, 3e-4, 0.1
        elif params_reel < 60_000_000:
            bs_c, ga_c, bl_c, lr_c, do_c = 4, 4, 256, 3e-4, 0.2
        elif params_reel < 150_000_000:
            bs_c, ga_c, bl_c, lr_c, do_c = 4, 8, 512, 1e-4, 0.2
        else:
            bs_c, ga_c, bl_c, lr_c, do_c = 4, 16, 512, 1e-4, 0.2

        p_m = params_reel / 1_000_000
        params_label = f"~{p_m:.1f}M" if p_m >= 1 else f"~{params_reel // 1000}k"

        print()
        print("  " + "-" * 50)
        print(f"  Config calculée pour {params_label} paramètres :")
        print(f"     n_embd  = {n_embd_calc:<6}  (richesse interne)")
        print(f"     n_head  = {n_head_calc:<6}  (têtes d'attention)")
        print(f"     n_layer = {n_layer_calc:<6}  (profondeur du réseau)")
        print("  " + "-" * 50)
        print()
        print("  Mode :")
        print("  [1] Simple  — utiliser cette config directement")
        print("  [2] Expert  — modifier les paramètres un par un")
        print()
        mode_c = input("  Choix [1] > ").strip()

        if mode_c == "2":
            # ── Expert : valeurs calculées pré-remplies, recommandations dynamiques ──
            VOCAB_E = _VOCAB_SIZE; BLOCK_E = 256

            def _conseil(msg):
                print("  \U0001f4a1 " + msg)

            def _params_reels(ne, nl):
                return 12 * nl * ne * ne + (VOCAB_E + BLOCK_E) * ne

            print("\n  ⚠️  Les valeurs calculées sont pré-remplies.")
            print("  Appuie sur Entrée pour les garder.\n")

            # ── BATCH SIZE ──────────────────────────────────────────────────────
            print("  ─── BATCH SIZE PHYSIQUE (VRAM) ───────────────────────────────────")
            print("  Nombre d'exemples traités en parallèle — limité par ta VRAM.")
            print("  4 = sûr pour presque toutes les configs (1-2 Go VRAM).")
            batch_size = demander("Batch size", bs_c, int, mini=1, maxi=512)
            _batch_rec_ga = max(1, 32 // batch_size)
            _conseil(f"Pour batch_size={batch_size}, vise accumulation={_batch_rec_ga} → batch effectif {batch_size * _batch_rec_ga}.")

            # ── ACCUMULATION ────────────────────────────────────────────────────
            print("\n  ─── ACCUMULATION DE GRADIENTS ────────────────────────────────")
            print("  Simule un plus grand batch sans consommer plus de VRAM.")
            print(f"  Recommandé : {_batch_rec_ga} (pour atteindre batch effectif ~32).")
            grad_accum_steps = demander("Accumulation", ga_c, int, mini=1, maxi=128)
            _eff = batch_size * grad_accum_steps
            if _eff < 8:
                _conseil(f"Batch effectif {_eff} est faible — apprentissage potentiellement instable. Monte l'accumulation.")
            elif _eff > 128:
                _conseil(f"Batch effectif {_eff} est très élevé — peut ralentir sans gain. 16-64 suffit généralement.")
            else:
                _conseil(f"Batch effectif {_eff} — bon équilibre stabilité/vitesse.")

            # ── BLOCK SIZE ──────────────────────────────────────────────────────
            print("\n  ─── LONGUEUR DU CONTEXTE (block_size) ──────────────────────────")
            print("  128 ≈ 90 mots  |  256 ≈ 180 mots  |  512 ≈ 360 mots")
            block_size = demander("Block size", bl_c, int, mini=32, maxi=1024)
            _bl_words = block_size * 0.7
            if block_size > 512 and params_reel < 20_000_000:
                _conseil(f"Block size {block_size} est grand pour un modèle {params_label} — préfère 256 pour éviter le surapprentissage.")
            elif block_size < 128:
                _conseil(f"Block size {block_size} = ~{int(_bl_words)} mots de contexte — très court, le modèle verra peu de structure.")
            else:
                _conseil(f"Block size {block_size} = ~{int(_bl_words)} mots de contexte — bien adapté.")

            # ── N_EMBD ──────────────────────────────────────────────────────────
            print("\n  ─── TAILLE INTERNE (n_embd) ─────────────────────────────────────")
            print(f"  Calculé automatiquement : {n_embd_calc}. Doit être multiple de 64.")
            n_embd = demander("N_embd", n_embd_calc, int, mini=64, maxi=2048)
            if n_embd % 64 != 0:
                _conseil(f"{n_embd} n'est pas multiple de 64 — moins efficace sur GPU. Essaie {(n_embd//64)*64} ou {(n_embd//64+1)*64}.")
            else:
                _p_new = _params_reels(n_embd, n_layer_calc)
                _delta_pct = (_p_new - cible) / cible * 100
                _sign = "+" if _delta_pct >= 0 else ""
                _conseil(f"n_embd={n_embd} donne ~{_p_new/1e6:.1f}M params ({_sign}{_delta_pct:.0f}% vs cible {params_label}).")

            # ── N_HEAD ──────────────────────────────────────────────────────────
            diviseurs_valides = [d for d in [1, 2, 4, 8, 12, 16] if n_embd % d == 0]
            print("\n  ─── TÊTES D'ATTENTION (n_head) ─────────────────────────────────────")
            print(f"  Valides pour n_embd={n_embd} : {diviseurs_valides}")
            print(f"  Head size cible ≈ 64 (stable) — recommandé : {diviseurs_valides[-1]} (head size = {n_embd // diviseurs_valides[-1]}).")
            n_head_def = n_head_calc if n_embd % n_head_calc == 0 else diviseurs_valides[-1]
            n_head = demander("N_head", n_head_def, int, mini=1, maxi=32)
            while n_embd % n_head != 0:
                print(f"  ⚠️  {n_head} ne divise pas {n_embd} — valides : {diviseurs_valides}")
                n_head = demander("N_head (corriger)", diviseurs_valides[-1], int, mini=1, maxi=32)
            _hs = n_embd // n_head
            if _hs < 32:
                _conseil(f"Head size {_hs} est petit — peut limiter la capacité d'attention. Essaie {diviseurs_valides[max(0,len(diviseurs_valides)-2)]} têtes.")
            elif _hs > 128:
                _conseil(f"Head size {_hs} est grand — envisage plus de têtes pour mieux capturer les patterns.")
            else:
                _conseil(f"Head size {_hs} par tête — bon équilibre.")

            # ── N_LAYER ─────────────────────────────────────────────────────────
            print("\n  ─── PROFONDEUR (n_layer) ──────────────────────────────────────────")
            print("  4 (rapide) · 6 · 8 · 12 (MEDIUM) · 16 (puissant)")
            n_layer = demander("N_layer", n_layer_calc, int, mini=1, maxi=48)
            _p_final = _params_reels(n_embd, n_layer)
            _delta2 = (_p_final - cible) / cible * 100
            _sign2 = "+" if _delta2 >= 0 else ""
            _conseil(f"Config finale : {n_embd} embd × {n_layer} layers = ~{_p_final/1e6:.1f}M params ({_sign2}{_delta2:.0f}% vs cible).")
            if abs(_delta2) > 50:
                print(f"  ⚠️  Écart important avec la cible {params_label} — tu peux ajuster n_embd ou n_layer.")

            # ── DROPOUT ─────────────────────────────────────────────────────────
            print("\n  ─── ORIGINALITÉ (dropout) ─────────────────────────────────────────")
            print("  0.0 → aucune régularisation · 0.2 → équilibré · 0.4 → original")
            _do_rec = 0.1 if _p_final < 10_000_000 else 0.2 if _p_final < 60_000_000 else 0.25
            print(f"  Pour {_p_final/1e6:.1f}M params, recommandé : {_do_rec}")
            dropout = demander("Dropout", do_c, float, mini=0.0, maxi=0.5)
            if dropout > 0.35 and _p_final < 20_000_000:
                _conseil(f"Dropout {dropout} est élevé pour un petit modèle — risque de sous-apprentissage. Essaie {_do_rec}.")
            elif dropout == 0.0:
                _conseil("Dropout 0.0 = pas de régularisation — risque d'overfitting si le dataset est petit.")

            # ── LEARNING RATE ────────────────────────────────────────────────────
            print("\n  ─── VITESSE D'APPRENTISSAGE (learning_rate) ───────────────────────")
            _lr_rec = 3e-4 if _p_final < 60_000_000 else 1e-4
            print(f"  Recommandé pour {_p_final/1e6:.1f}M params : {_lr_rec:.0e}")
            print("  Trop grand → instable (loss explose). Trop petit → très lent.")
            learning_rate = demander("Learning rate", lr_c, float, mini=1e-6, maxi=1e-2)
            if learning_rate > 5e-4 and _p_final > 50_000_000:
                _conseil(f"{learning_rate:.0e} est élevé pour un grand modèle — risque d'instabilité. Essaie {_lr_rec:.0e}.")
            elif learning_rate < 1e-5:
                _conseil(f"{learning_rate:.0e} est très lent — l'entraînement prendra très longtemps.")
            else:
                _conseil(f"Learning rate {learning_rate:.0e} — dans la plage recommandée.")

        else:
            # ── Simple : config calculée directement ──────────────────
            batch_size       = bs_c
            grad_accum_steps = ga_c
            block_size       = bl_c
            n_embd           = n_embd_calc
            n_head           = n_head_calc
            n_layer          = n_layer_calc
            learning_rate    = lr_c

            print()
            print("  ─── ORIGINALITÉ ──────────────────────────────────────────────────────────")
            print("  0.0 → texte cohérent et prévisible")
            print("  0.2 → bon équilibre  ← recommandé")
            print("  0.4 → texte plus original, moins répétitif")
            dropout = demander("Originalité", do_c, float, mini=0.0, maxi=0.5)

            print("\n  ─── SAUVEGARDE AUTOMATIQUE (checkpoint) ────────────────────────")
            print("  Toutes les  100 étapes → très fréquent")
            print("  Toutes les  500 étapes → bon équilibre  ← recommandé")
            print("  Toutes les 1000 étapes → moins fréquent")
            checkpoint_interval = demander("Sauvegarder toutes les N étapes", 500, int, mini=10, maxi=10000)
            print(f"\n  ✅ Config : {n_embd} embd / {n_head} heads / {n_layer} layers — {params_label} — originalité {dropout} — checkpoint /{checkpoint_interval}")

    # ────────────────────────────────────────────────────────────
    #  ÉTAPE 3 — SAUVEGARDE AUTOMATIQUE (CHECKPOINT)
    # ────────────────────────────────────────────────────────────
    if checkpoint_interval is None:
        print("\n" + "="*62)
        print("  💾  ÉTAPE 3 / 4 — SAUVEGARDE AUTOMATIQUE")
        print("="*62)
        print("  Un checkpoint = une sauvegarde complète de l'entraînement.")
        print("  Si ton PC s'éteint ou plante, tu peux reprendre EXACTEMENT")
        print("  là où tu t'es arrêté au prochain lancement.")
        print()
        print("  Toutes les  100 étapes → très fréquent (prend un peu plus de place)")
        print("  Toutes les  500 étapes → bon équilibre  ← recommandé")
        print("  Toutes les 1000 étapes → moins fréquent (risque de perdre + de travail)")
        print()
        checkpoint_interval = demander(
            "Sauvegarder toutes les N étapes", 500, int, mini=10, maxi=10000
        )
        print(f"  → Checkpoint toutes les {checkpoint_interval} étapes")

    # ────────────────────────────────────────────────────────────
    #  ÉTAPE 4 — PARAMÈTRES AVANCÉS (optionnel)
    # ────────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  🔬  ÉTAPE 4 / 4 — PARAMÈTRES AVANCÉS (optionnel)")
    print("="*62)
    print("  Ces paramètres ont déjà de bonnes valeurs par défaut.")
    print("  Appuie sur Entrée pour les garder sans les modifier.")
    print()

    print("  ─── EVAL INTERVAL ────────────────────────────────────────")
    print("  Tous les combien d'étapes calcule-t-on la val_loss ?")
    print("  Évaluer trop souvent ralentit l'entraînement (mais c'est plus précis).")
    eval_interval = demander("Évaluer toutes les N étapes", 500, int, mini=50, maxi=5000)

    print("\n  ─── EVAL ITERS ───────────────────────────────────────────")
    print("  Combien de batches utiliser pour calculer la val_loss ?")
    print("  ↑ = mesure plus précise, mais légèrement plus lente")
    eval_iters = demander("Batches pour l'évaluation", 100, int, mini=10, maxi=500)

    # ────────────────────────────────────────────────────────────
    #  RÉCAPITULATIF
    # ────────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  📋  RÉCAPITULATIF DE TA CONFIGURATION")
    print("="*62)
    print(f"  Nom du modèle    : {nom_modele}")
    print(f"  Appareil         : {device.upper()}  ({gpu_nom})")
    print(f"  Batch size       : {batch_size} (x{grad_accum_steps} accum = {batch_size * grad_accum_steps} effectif)")
    print(f"  Block size       : {block_size} tokens  (≈ {block_size * 3 // 4} mots)")
    print(f"  Architecture     : {n_layer} couches · {n_head} têtes · {n_embd} dim")
    print(f"  Dropout          : {dropout}")
    print(f"  Learning rate    : {learning_rate:.2e}")
    print(f"  Checkpoint       : toutes les {checkpoint_interval} étapes")
    print(f"  Évaluation       : toutes les {eval_interval} étapes")
    print(f"  Limite VRAM      : {LIMITE_VRAM_GO} Go  (arrêt auto au-dessus)")
    print(f"  Limite RAM       : {LIMITE_RAM_GO} Go  (arrêt auto au-dessus)")
    print("="*62)

    return {
        "nom_modele"         : nom_modele,
        "device"             : device,
        "gpu_nom"            : gpu_nom,
        "vram_total_go"      : vram_go,
        "ram_total_go"       : ram_go,
        "batch_size"         : batch_size,
        "grad_accum_steps"   : grad_accum_steps,
        "block_size"         : block_size,
        "n_embd"             : n_embd,
        "n_head"             : n_head,
        "n_layer"            : n_layer,
        "dropout"            : dropout,
        "learning_rate"      : learning_rate,
        "checkpoint_interval": checkpoint_interval,
        "eval_interval"      : eval_interval,
        "eval_iters"         : eval_iters,
        "preset"             : nom_preset,
    }

# ================================================================
#  MODE RAPIDE (--quick) : zéro question, preset MINI
# ================================================================

def configurer_quick():
    """
    Configuration automatique pour python go.py --quick (ou quick.py).
    Aucune interaction : preset MINI, nom 'test_20M', mode auto.
    """
    print("\n" + "="*62)
    print("  ⚡  MODE RAPIDE — WishAI Quick")
    print("="*62)
    device, gpu_nom, vram_go, ram_go, _ = verifier_pc()
    configurer_limites_memoire(vram_go, ram_go)
    p = PRESETS["MINI"]
    print(f"\n  Preset sélectionné : {p['emoji']} MINI — {p['nb_params_approx']} paramètres")
    print(f"  {p['description']}")
    print("  (nom du modèle : quick)")
    print("  (durée : 30 minutes max)")
    print("="*62)
    return {
        "nom_modele"         : "quick",
        "device"             : device,
        "gpu_nom"            : gpu_nom,
        "vram_total_go"      : vram_go,
        "ram_total_go"       : ram_go,
        "batch_size"         : p["batch_size"],
        "grad_accum_steps"   : p.get("grad_accum_steps", 2),
        "block_size"         : p["block_size"],
        "n_embd"             : p["n_embd"],
        "n_head"             : p["n_head"],
        "n_layer"            : p["n_layer"],
        "dropout"            : p["dropout"],
        "learning_rate"      : p["learning_rate"],
        "checkpoint_interval": 500,
        "eval_interval"      : 500,
        "eval_iters"         : 100,
        "preset"             : "MINI",
    }

# ================================================================
#  LANCEMENT DE LA CONFIGURATION
# ================================================================

_QUICK_MODE = "--quick" in sys.argv
cfg = configurer_quick() if _QUICK_MODE else configurer()

# Extraction des variables
nom_modele          = cfg["nom_modele"]
device              = cfg["device"]
batch_size          = cfg["batch_size"]
grad_accum_steps    = cfg.get("grad_accum_steps", 1)
block_size          = cfg["block_size"]
n_embd              = cfg["n_embd"]
n_head              = cfg["n_head"]
n_layer             = cfg["n_layer"]
dropout             = cfg["dropout"]
learning_rate       = cfg["learning_rate"]
checkpoint_interval = cfg["checkpoint_interval"]
eval_interval       = cfg["eval_interval"]
eval_iters          = cfg["eval_iters"]
max_iters           = 999_999_999   # sera ajusté par choisir_duree()

# ================================================================
#  CHARGEMENT DU TOKENIZER BPE
# ================================================================
print("\n" + "="*62)
print("  🔤  TOKENIZER BPE")
print("="*62)

if not os.path.exists(TOKENIZER_FILE):
    print("❌ tokenizer.json introuvable !")
    print("   Lance d'abord : python tokenizer.py")
    raise SystemExit(1)

tok          = TokenizerBPE()
tok.charger(TOKENIZER_FILE)
taille_vocab = tok.taille_vocab
print(f"  ✅ Vocab BPE : {taille_vocab} tokens chargés")
print("     (vs ~100 tokens en char-level — contexte 3× plus riche !)")

# ================================================================
#  SÉLECTION ET CHARGEMENT DES DONNÉES
# ================================================================
DATA_FILE     = os.path.join(_ROOT, 'data', 'data.txt')
DATA_MANIFEST = None   # chemin du manifest.json si mode manifest
DATA_LANG     = None   # langue du dataset sélectionné
FLAG          = {
    "fr"   : "🇫🇷 Français",
    "en"   : "🇬🇧 Anglais",
    "multi": "🌍 Multilingue",
}

def _lire_depuis_manifest(manifest_path):
    """Concatène toutes les sources listées dans manifest.json."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    langue      = manifest["langue"]
    sources_dir = os.path.join(_ROOT, "data", langue, "sources")
    texte = ""
    for src in manifest.get("sources", []):
        fichier = os.path.join(sources_dir, src["fichier"])
        if os.path.exists(fichier):
            mo = src.get("mo", "?")
            print(f"     Lecture : {src['fichier']} ({mo} Mo)")
            with open(fichier, 'r', encoding='utf-8', errors='ignore') as f:
                texte += f.read()
    return texte

def choisir_donnees():
    """
    Cherche manifests puis data.txt disponibles, demande lequel utiliser si plusieurs.
    Priorité : manifest.json > data.txt (manifest = pas de doublon de stockage).
    """
    global DATA_FILE, DATA_MANIFEST, DATA_LANG

    # ── Manifests disponibles ──────────────────────────────────────
    manifests_dispo = []
    for langue in ["fr", "en", "multi"]:
        manifest = os.path.join(_ROOT, "data", langue, "manifest.json")
        if os.path.exists(manifest):
            try:
                with open(manifest, 'r', encoding='utf-8') as f:
                    m = json.load(f)
                total_mo = m.get("total_mo", 0)
                manifests_dispo.append((langue, manifest, total_mo))
            except Exception:
                pass

    # ── data.txt legacy ───────────────────────────────────────────
    legacy_dispo = []
    for langue in ["fr", "en", "multi"]:
        chemin = os.path.join(_ROOT, "data", langue, "data.txt")
        if os.path.exists(chemin) and os.path.getsize(chemin) > 1024:
            taille = os.path.getsize(chemin) / 1_000_000
            legacy_dispo.append((langue, chemin, taille))

    if not manifests_dispo and not legacy_dispo:
        if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 1024:
            print(f"  Données : {DATA_FILE} ({os.path.getsize(DATA_FILE)/1e6:.1f} Mo)")
            return
        print("\n❌ Aucun fichier de données trouvé !")
        print("   Lance d'abord : python telecharger.py")
        raise SystemExit(1)

    toutes = ([(l, m, s, "manifest") for l, m, s in manifests_dispo] +
              [(l, c, s, "txt")      for l, c, s in legacy_dispo])

    if len(toutes) == 1:
        langue, chemin, taille, typ = toutes[0]
        if typ == "manifest":
            DATA_MANIFEST = chemin
            DATA_LANG     = langue
            DATA_FILE     = None
            print(f"  Données : {FLAG.get(langue, langue)} — manifest ({taille:.1f} Mo total)")
        else:
            DATA_FILE = chemin
            DATA_LANG = langue
            print(f"  Données : {FLAG.get(langue, langue)} — {chemin} ({taille:.1f} Mo)")
        return

    print("\n" + "="*62)
    print("  📚  PLUSIEURS JEUX DE DONNÉES DISPONIBLES")
    print("="*62)
    for i, (langue, chemin, taille, typ) in enumerate(toutes, 1):
        label = "manifest" if typ == "manifest" else "data.txt"
        print(f"  [{i}]  {FLAG.get(langue, langue):<24} {taille:.1f} Mo  ({label})")
    print("="*62)

    if _QUICK_MODE:
        toutes.sort(key=lambda x: x[2], reverse=True)
        langue, chemin, taille, typ = toutes[0]
    else:
        choix = input("  Choix > ").strip()
        try:
            idx = int(choix) - 1
            if 0 <= idx < len(toutes):
                langue, chemin, taille, typ = toutes[idx]
            else:
                raise ValueError
        except (ValueError, IndexError):
            langue, chemin, taille, typ = toutes[0]
            print(f"  → {FLAG.get(langue, langue)} sélectionné par défaut")

    if typ == "manifest":
        DATA_MANIFEST = chemin
        DATA_LANG     = langue
        DATA_FILE     = None
        print(f"  → {FLAG.get(langue, langue)} — manifest ({taille:.1f} Mo)")
    else:
        DATA_FILE = chemin
        DATA_LANG = langue
        print(f"  → {FLAG.get(langue, langue)} — {chemin} ({taille:.1f} Mo)")

print("\n" + "="*62)
print("  📚  DONNÉES D'ENTRAÎNEMENT")
print("="*62)
choisir_donnees()

# ── Encodage BPE + cache ─────────────────────────────────────────
#
#  L'encodage BPE peut prendre 5-10 minutes la première fois.
#  Ensuite, le résultat est mis en cache (bpe_cache.pt) :
#  les lancements suivants sont instantanés.
#
if DATA_MANIFEST:
    _data_dir  = os.path.dirname(DATA_MANIFEST)
else:
    _data_dir  = os.path.dirname(DATA_FILE) if DATA_FILE else "."
CACHE_FILE = os.path.join(_data_dir, "bpe_cache.pt")
RELOAD_FLAG = os.path.join(_ROOT, "data", DATA_LANG, "reload_requested.flag") if DATA_LANG else None

def _cache_valide():
    """Cache valide ssi il existe ET est plus récent que tokenizer.json et les sources."""
    if not os.path.exists(CACHE_FILE):
        return False
    tok_file = os.path.join(_ROOT, "data", "tokenizer.json")
    if os.path.exists(tok_file):
        if os.path.getmtime(tok_file) > os.path.getmtime(CACHE_FILE):
            return False
    # Si manifest : vérifier si une source est plus récente que le cache
    if DATA_MANIFEST and os.path.exists(DATA_MANIFEST):
        try:
            with open(DATA_MANIFEST, 'r', encoding='utf-8') as f:
                m = json.load(f)
            sources_dir = os.path.join(_ROOT, "data", m["langue"], "sources")
            for src in m.get("sources", []):
                fp = os.path.join(sources_dir, src["fichier"])
                if os.path.exists(fp) and os.path.getmtime(fp) > os.path.getmtime(CACHE_FILE):
                    return False  # nouvelle source → cache périmé
        except Exception:
            pass
    return True

def recharger_donnees():
    """Re-tokenise depuis le manifest (appelé en hot-reload)."""
    print("\n  ♻️  Rechargement des données depuis manifest...")
    texte     = _lire_depuis_manifest(DATA_MANIFEST)
    ids_liste = tok.encoder_dataset(texte)
    del texte
    gc.collect()
    new_data = torch.tensor(ids_liste, dtype=torch.int32)
    del ids_liste
    gc.collect()
    torch.save(new_data, CACHE_FILE)
    print(f"  ✅ {len(new_data):,} tokens — cache mis à jour")
    return new_data

if _cache_valide():
    print("  ✅ Cache BPE trouvé — chargement instantané...")
    data = torch.load(CACHE_FILE, weights_only=True)
    print(f"     {len(data):,} tokens chargés")
else:
    if os.path.exists(CACHE_FILE):
        print("  ↻ Données mises à jour — réencodage...")
        try: os.remove(CACHE_FILE)
        except Exception: pass
    print("  ⏳ Encodage BPE en cours...")
    print("     (première fois uniquement — résultat mis en cache)")
    if DATA_MANIFEST:
        texte = _lire_depuis_manifest(DATA_MANIFEST)
    else:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            texte = f.read()
    ids_liste = tok.encoder_dataset(texte)
    del texte
    gc.collect()
    data = torch.tensor(ids_liste, dtype=torch.int32)
    del ids_liste
    gc.collect()
    torch.save(data, CACHE_FILE)
    print(f"  ✅ {len(data):,} tokens encodés et mis en cache → {CACHE_FILE}")
    # libère le cache interne du tokenizer — inutile pendant l'entraînement
    tok._cache.clear()
    gc.collect()

# Division train/validation (90% / 10%)
#   Train : données sur lesquelles l'IA apprend
#   Val   : données JAMAIS vues pendant l'entraînement → mesure la vraie performance
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"\n  Train : {len(train_data):,} tokens (90%)")
print(f"  Val   : {len(val_data):,}   tokens (10%) — jamais vus pendant l'entraînement")
print(f"  RAM   : {data.element_size() * data.nelement() / 1e9:.2f} Go")

def get_batch(split):
    """
    Extrait un mini-batch aléatoire du dataset.

    x : séquences d'entrée    [token_0, token_1, ..., token_T-1]
    y : séquences cibles      [token_1, token_2, ..., token_T  ]

    y est décalé d'un pas par rapport à x : l'IA doit prédire
    le token suivant à partir des tokens précédents.
    """
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size]     for i in ix]).long()
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix]).long()
    return x.to(device), y.to(device)

# ================================================================
#  ARCHITECTURE DU MODÈLE GPT
# ================================================================
#
#  On implémente un GPT Decoder-Only — la même architecture
#  que GPT-2 (OpenAI), à plus petite échelle.
#
#  Flux de données :
#    [tokens] → Token Embedding + Position Embedding
#             → N × BlocTransformer
#             → LayerNorm finale
#             → Linear → probabilités sur le vocabulaire
# ================================================================

# Les classes du modèle (CoucheAttention, MultiTetesAttention, CoucheFFN,
# BlocTransformer, WishAI_BPE) vivent désormais dans src/model.py (importées
# plus haut). Paramétrées par ConfigModele → testables sans entraînement.


# ================================================================
#  CRÉATION DU MODÈLE
# ================================================================
print("\n" + "="*62)
print("  🧠  CRÉATION DU MODÈLE")
print("="*62)

_cfg_modele = ConfigModele(
    vocab_size=taille_vocab, n_embd=n_embd, n_head=n_head,
    n_layer=n_layer, block_size=block_size, dropout=dropout,
)
modele    = WishAI_BPE(_cfg_modele).to(device)
nb_params = sum(p.numel() for p in modele.parameters())
print(f"  Paramètres  : {nb_params:,}  ({nb_params/1e6:.1f}M)")
print(f"  Vocab       : {taille_vocab} tokens BPE")
print(f"  Contexte    : {block_size} tokens  ≈ {block_size * 3 // 4} mots")
print(f"  Architecture: {n_layer} couches × {n_head} têtes × {n_embd} dim")
print(f"  Device      : {device.upper()}")

# ── Précision mixte (AMP) — accélère et réduit la mémoire sur GPU NVIDIA ──
import contextlib as _contextlib
_amp_on = (device == "cuda")
if _amp_on:
    _amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _amp_ctx   = torch.amp.autocast(device_type="cuda", dtype=_amp_dtype)
    print(f"  AMP         : activée ({'bfloat16' if _amp_dtype == torch.bfloat16 else 'float16'})")
else:
    _amp_dtype = torch.float32
    _amp_ctx   = _contextlib.nullcontext()
    print(f"  AMP         : désactivée (float32 sur {device.upper()})")
# GradScaler : indispensable en fp16 pour éviter les NaN ; inutile en bf16/float32.
scaler = torch.amp.GradScaler("cuda", enabled=(_amp_on and _amp_dtype == torch.float16))

# ── Compilation (PyTorch 2.x) — activée seulement sur Linux+CUDA ──
# (instable / non supportée sur Windows et MPS → on saute proprement)
try:
    if hasattr(torch, "compile") and device == "cuda" and sys.platform.startswith("linux"):
        modele = torch.compile(modele)
        print("  torch.compile : activé")
    else:
        print("  torch.compile : ignoré (Linux+CUDA uniquement)")
except Exception as _e:
    print(f"  torch.compile : indisponible ({_e}) — on continue sans")

# Module brut sous-jacent : pour sauvegarder/charger les poids SANS le préfixe
# '_orig_mod.' ajouté par torch.compile → checkpoints toujours compatibles.
def _modele_brut():
    return getattr(modele, "_orig_mod", modele)

@torch.no_grad()
def estimer_perte():
    """
    Calcule la perte moyenne sur train et val SANS modifier les poids.
    (torch.no_grad() désactive le calcul de gradient → plus rapide)

    Utilise eval_iters batches aléatoires pour une estimation stable.
    Retourne un dict : {"train": float, "val": float}
    """
    out = {}
    modele.eval()  # désactive le dropout pendant l'évaluation
    for split in ['train', 'val']:
        pertes = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y      = get_batch(split)
            _, p      = modele(X, Y)
            pertes[k] = p.item()
        out[split] = pertes.mean()
    modele.train()  # réactive le dropout pour l'entraînement
    return out

optimiseur = torch.optim.AdamW(modele.parameters(), lr=learning_rate)
#
#  AdamW = Adam + weight decay
#  C'est l'optimiseur standard pour les Transformers.
#  Il adapte le learning rate de chaque paramètre séparément.

# ── Learning Rate Scheduler (warmup + cosine decay) ─────────────
WARMUP_ITERS = 100      # étapes de montée progressive
LR_MIN       = learning_rate * 0.1   # LR minimum = 10% du LR initial

def get_lr(iteration):
    """Calcule le learning rate avec warmup et cosine decay."""
    if iteration < WARMUP_ITERS:
        return learning_rate * (iteration + 1) / WARMUP_ITERS
    
    decay_iters = 10_000 if (max_iters == 999_999_999) else (max_iters - WARMUP_ITERS)
    progress = (iteration - WARMUP_ITERS) / max(decay_iters, 1)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + coeff * (learning_rate - LR_MIN)

# tout va dans model/<nom>/ — propre et rangé (chemins absolus pour eviter
# tout bug de CWD entre la creation du dossier et l'ecriture du log)
MODEL_DIR = os.path.join(_ROOT, "model", nom_modele)
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(MODEL_DIR, "checkpoint.pt")
LOG_FILE        = os.path.join(MODEL_DIR, "log_active.json")

# le dashboard lit ce fichier pour savoir quel modèle surveiller
os.makedirs(os.path.join(_ROOT, "model"), exist_ok=True)
with open(os.path.join(_ROOT, "model", "active.json"), "w", encoding="utf-8") as _f:
    json.dump({"model": nom_modele}, _f)

start_iter = 0

log_data = {
    "hyperparams": {
        "batch_size"   : batch_size,
        "block_size"   : block_size,
        "max_iters"    : max_iters,
        "learning_rate": learning_rate,
        "n_embd"       : n_embd,
        "n_head"       : n_head,
        "n_layer"      : n_layer,
        "dropout"      : dropout,
        "nb_params"    : nb_params,
        "taille_vocab" : taille_vocab,
        "tokenizer"    : "BPE",
        "device"       : device,
        "gpu"          : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_total_go": round(cfg["vram_total_go"], 1),
        "ram_total_go" : round(cfg["ram_total_go"], 1),
        "nb_tokens_train": len(train_data),
    },
    "nom_modele" : nom_modele,
    "preset"     : cfg.get("preset", "CUSTOM"),
    "mode"       : "auto",
    "steps"      : [],
    "status"     : "en cours",
    "debut"      : time.strftime("%H:%M:%S"),
    "last_update": time.time(),
}

# ── Reprise depuis checkpoint ────────────────────────────────────
if _QUICK_MODE:
    # Mode quick : toujours repartir de zéro, pas de question posée
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

elif os.path.exists(CHECKPOINT_FILE):
    print("\n" + "="*62)
    print("  💾  CHECKPOINT DÉTECTÉ !")
    print("="*62)
    _ckpt_tmp = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
    ckpt_iter = _ckpt_tmp["iteration"]
    print(f"  Sauvegarde trouvée à l'étape {ckpt_iter:,}")
    print()
    print("  [O] Continuer l'entraînement depuis cette étape")
    print("  [n] Recommencer depuis zéro  (efface le checkpoint)")
    choix = input("  Choix [O] > ").strip().lower()
    if choix != 'n':
        _modele_brut().load_state_dict(_ckpt_tmp["modele_state"])
        optimiseur.load_state_dict(_ckpt_tmp["optimiseur_state"])
        start_iter = ckpt_iter + 1
        log_data   = _ckpt_tmp.get("log_data", log_data)
        log_data["status"] = "en cours"
        print(f"  → Reprise à l'étape {start_iter:,}")
    else:
        os.remove(CHECKPOINT_FILE)
        print("  → Nouveau départ depuis zéro")

else:
    # Mode normal sans checkpoint : proposer fine-tuning
    modeles_dispo = sorted(glob.glob(os.path.join(_ROOT, "model", "*", "modele.pt")))
    if modeles_dispo:
        print("\n" + "="*62)
        print("  🔄  FINE-TUNING — partir d'un modèle existant ?")
        print("="*62)
        print("  Le fine-tuning charge les poids d'un modèle déjà entraîné")
        print("  et continue sur de nouvelles données.")
        print("  Utile pour spécialiser un modèle généraliste.")
        print()
        for i, f in enumerate(modeles_dispo):
            nom_base = os.path.basename(os.path.dirname(f))
            taille = os.path.getsize(f) / 1_000_000
            print(f"    [{i+1}] {nom_base}  ({taille:.0f} Mo)")
        print("    [0] Nouveau modèle depuis zéro  ← par défaut")
        choix_base = input("  Choix [0] > ").strip()
        try:
            idx_base = int(choix_base)
            if 1 <= idx_base <= len(modeles_dispo):
                base_file = modeles_dispo[idx_base - 1]
                base_ckpt = torch.load(base_file, map_location=device, weights_only=False)
                _modele_brut().load_state_dict(base_ckpt["modele_state"])
                print(f"  → Poids chargés depuis {base_file}")
                print("     Fine-tuning activé !")
        except Exception:
            print("  → Nouveau modèle depuis zéro")

# ================================================================
#  SÉLECTEUR DE DURÉE D'ENTRAÎNEMENT
# ================================================================

def choisir_duree():
    """
    Demande combien de temps doit durer l'entraînement.

    Mode automatique (Entrée) :
      Mesure la val_loss toutes les eval_interval étapes.
      Si elle ne s'améliore plus pendant 5 évaluations → arrêt.
      C'est le mode recommandé pour obtenir le meilleur résultat.

    Mode minuté (ex: 60) :
      Lance un warmup de 100 étapes pour mesurer la vitesse,
      puis calcule exactement combien d'étapes rentrent
      dans le temps imparti.
    """
    global max_iters
    print("\n" + "="*62)
    print("  ⏱️   DURÉE D'ENTRAÎNEMENT")
    print("="*62)
    print("  Combien de minutes veux-tu entraîner ton IA ?")
    print()
    print("  Exemples : 30, 60, 120, 240, 480 (en minutes)")
    print("  [Entrée] = Mode automatique — s'arrête tout seul quand")
    print("             l'IA ne progresse plus (recommandé !)")
    print()
    saisie = input("  Minutes [auto] > ").strip()

    if not saisie:
        max_iters = 999_999_999
        log_data["hyperparams"]["max_iters"] = max_iters
        log_data["mode"] = "auto"
        print("  → Mode automatique — arrêt à la convergence (Ctrl+C pour forcer l'arrêt)")
        return start_iter

    try:
        minutes = float(saisie)
    except Exception:
        minutes = 60.0

    duree_s  = int(minutes * 60)
    hm       = (f"{int(minutes)//60}h{int(minutes)%60:02d}m"
                if minutes >= 60 else f"{int(minutes)}min")

    print(f"\n  Durée cible : {hm}")
    print("  Calcul de la vitesse sur 100 étapes de warmup...")

    debut_w = time.time()
    for _ in range(100):
        xb, yb    = get_batch('train')
        optimiseur.zero_grad(set_to_none=True)
        with _amp_ctx:
            _, loss_w = modele(xb, yb)
        scaler.scale(loss_w).backward()
        scaler.step(optimiseur)
        scaler.update()
    vitesse = 100 / max(time.time() - debut_w, 1e-6)

    new_start      = start_iter + 100
    iters_restants = int(vitesse * duree_s)
    new_max        = new_start + iters_restants
    heure_fin      = time.strftime("%H:%M", time.localtime(time.time() + duree_s))

    print(f"\n  Vitesse mesurée : {vitesse:.2f} it/s")
    print(f"  Étapes planifiées : {new_max:,}  ({iters_restants:,} après warmup)")
    print(f"  Fin prévue à      : {heure_fin}")

    confirm = input("\n  Confirmer ? [O/n] > ").strip().lower()
    if confirm != 'n':
        max_iters = new_max
        log_data["hyperparams"]["max_iters"] = max_iters
        log_data["mode"] = f"minuté {int(minutes)}min"
        print(f"  ✅ Entraînement jusqu'à l'étape {max_iters:,}")
    return new_start

if _QUICK_MODE:
    # Mode rapide : 30 minutes max, puis arrêt propre
    _QUICK_MAX_MINUTES = 30
    _QUICK_MAX_SEC     = _QUICK_MAX_MINUTES * 60
    print(f"\n  ⚡ Mode rapide — mesure de la vitesse (50 étapes)...")
    _t0_warmup = time.time()
    for _wi in range(50):
        _xw, _yw = get_batch('train')
        optimiseur.zero_grad(set_to_none=True)
        with _amp_ctx:
            _, _lw = modele(_xw, _yw)
            _lw = _lw / grad_accum_steps
        scaler.scale(_lw).backward()
        scaler.step(optimiseur)
        scaler.update()
    _vitesse_quick = 50 / max(time.time() - _t0_warmup, 1e-6)
    _iters_30min   = int(_vitesse_quick * _QUICK_MAX_SEC)
    max_iters      = start_iter + 50 + _iters_30min
    log_data["hyperparams"]["max_iters"] = max_iters
    log_data["mode"] = f"quick {_QUICK_MAX_MINUTES}min"
    new_start_iter  = start_iter + 50
    _heure_fin = time.strftime("%H:%M", time.localtime(time.time() + _QUICK_MAX_SEC))
    print(f"  Vitesse : {_vitesse_quick:.1f} it/s  →  ~{_iters_30min:,} étapes en {_QUICK_MAX_MINUTES} min")
    print(f"  Fin prévue à {_heure_fin}")
    _QUICK_DEADLINE = time.time() + _QUICK_MAX_SEC
else:
    _QUICK_DEADLINE = None
    new_start_iter = choisir_duree()

print("\n" + "="*62)
print("  🚀  ENTRAÎNEMENT EN COURS")
print("="*62)
print(f"  Étapes  : {new_start_iter:,} → {max_iters:,}")
print(f"  Vocab   : {taille_vocab} tokens BPE")
print(f"  Contexte: {block_size} tokens  ≈ {block_size * 3 // 4} mots")
print()
print("  Légende des métriques :")
print("  train_loss = erreur sur les données d'entraînement")
print("  val_loss   = erreur sur données jamais vues (plus importante !)")
print("  perp       = perplexité = e^(val_loss)  |  GPT-2 small ≈ 30")
print("  Gap        = val_loss − train_loss  |  idéal ≈ 0  (si > 0.3 → overfitting)")
print("="*62 + "\n")

# ── Initialisation des logs ──────────────────────────────────────
debut        = time.time()
LOG_INTERVAL = 50      # fréquence de mise à jour du log pour le dashboard
_dernier_log = time.time()

with open(LOG_FILE, "w", encoding="utf-8") as f:
    json.dump(log_data, f, ensure_ascii=False)

# ── Early stopping ───────────────────────────────────────────────
_meilleur_val = float('inf')
_patience     = 5       # nb d'évaluations sans amélioration avant d'arrêter
_sans_progres = 0
_mode_auto    = (max_iters == 999_999_999)

# ── Arrêt propre au Ctrl+C ───────────────────────────────────────
_stop_demande = False
def _stop_handler(sig, frame):
    global _stop_demande
    _stop_demande = True
    print("\n⚠️  Ctrl+C détecté — arrêt propre en cours...")
_sig.signal(_sig.SIGINT, _stop_handler)

# ================================================================
#  BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# ================================================================

for iteration in range(new_start_iter, max_iters):

    # ── Application du Learning Rate ─────────────────────────────
    lr_actuel = get_lr(iteration)
    for param_group in optimiseur.param_groups:
        param_group['lr'] = lr_actuel

    # ── Mise à jour du log toutes les LOG_INTERVAL étapes ────────
    #    (pour que le dashboard se rafraîchisse en temps réel)
    if iteration % LOG_INTERVAL == 0 and iteration > 0:
        maintenant    = time.time()
        duree_segment = max(maintenant - _dernier_log, 1e-6)
        vitesse       = round(LOG_INTERVAL / duree_segment, 2)
        restant       = (max_iters - iteration) / vitesse if vitesse > 0 else 0
        _dernier_log  = maintenant

        log_data["iteration_courante"] = iteration
        log_data["vitesse_its"]        = vitesse
        log_data["eta_secondes"]       = round(restant)
        log_data["progression"]        = round(iteration / max_iters * 100, 1)
        log_data["temp_gpu"]           = lire_temp_gpu()
        log_data["memoire"]            = get_memoire()
        log_data["lr_actuel"]          = lr_actuel
        log_data["last_update"]        = time.time()

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False)

    # ── Évaluation (toutes les eval_interval étapes) ─────────────
    if iteration % eval_interval == 0 or iteration == max_iters - 1:
        pertes = estimer_perte()
        ecoul  = time.time() - debut
        perp   = round(math.exp(pertes['val'].item()), 2)
        gap    = round(pertes['val'].item() - pertes['train'].item(), 4)

        print(f"Étape {iteration:6,}  │  train: {pertes['train']:.4f}  "
              f"val: {pertes['val']:.4f}  "
              f"perp: {perp:.1f}  "
              f"gap: {gap:+.4f}  │  {ecoul:.0f}s")

        # Génère un exemple de texte pour voir la progression
        exemple_texte = ""
        if iteration > 0:
            modele.eval()
            with torch.no_grad():
                contexte  = torch.zeros((1, 1), dtype=torch.long, device=device)
                ids_gen   = modele.generer(contexte, max_nouveaux_tokens=80)[0].tolist()
                exemple_texte = tok.decoder(ids_gen)
                print(f"  Exemple : {exemple_texte[:120].strip()!r}")
            modele.train()

        mem = get_memoire()
        print(f"  Mémoire : VRAM {mem['vram_go']:.2f}/{mem['vram_total_go']:.1f} Go  │  "
              f"RAM {mem['ram_go']:.1f} Go  │  GPU {lire_temp_gpu()}°C\n")

        # Sauvegarde dans le log
        log_data["steps"].append({
            "step"      : iteration,
            "train_loss": round(pertes['train'].item(), 4),
            "val_loss"  : round(pertes['val'].item(), 4),
            "temps"     : round(ecoul, 1),
            "exemple"   : exemple_texte[:120].strip(),
            "memoire"   : mem,
        })
        log_data["status"]      = "termine" if iteration == max_iters - 1 else "en cours"
        log_data["progression"] = round(iteration / max_iters * 100, 1)
        log_data["memoire"]     = mem
        log_data["temp_gpu"]    = lire_temp_gpu()
        log_data["last_update"] = time.time()

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False)

        # ── Export embeddings pour le visualiseur (visual/index.html) ──
        try:
            import numpy as _np
            _w = _modele_brut().token_embedding.weight.detach().cpu().float().numpy()
            # PCA 3D rapide (< 5ms pour 4000×64)
            _X   = _w - _w.mean(axis=0)
            _cov = _np.cov(_X.T)
            _val, _vec = _np.linalg.eigh(_cov)
            _top3 = _np.argsort(_val)[::-1][:3]
            _proj = (_X @ _vec[:, _top3]).tolist()
            _vocab = [tok.id_vers_token.get(i, f"<{i}>") for i in range(len(_proj))]
            _emb_json = {
                "step"      : iteration,
                "val_loss"  : round(pertes['val'].item(), 4),
                "n_embd"    : n_embd,
                "vocab_size": len(_vocab),
                "updated"   : time.strftime("%Y-%m-%d %H:%M:%S"),
                "tokens"    : _vocab,
                "x"         : [round(p[0], 5) for p in _proj],
                "y"         : [round(p[1], 5) for p in _proj],
                "z"         : [round(p[2], 5) for p in _proj],
            }
            with open(os.path.join(MODEL_DIR, "embeddings.json"), "w", encoding="utf-8") as _ef:
                json.dump(_emb_json, _ef, ensure_ascii=False)
        except Exception:
            pass  # silencieux — la visualisation est un bonus

        # Vérification sécurité mémoire/température
        if not check_securite():
            print("\n🛑 Limite mémoire ou température atteinte — sauvegarde et arrêt")
            break

        # Early stopping (mode automatique uniquement)
        if _mode_auto:
            val_actuel = pertes['val'].item()
            if val_actuel < _meilleur_val - 0.005:
                _meilleur_val = val_actuel
                _sans_progres = 0
                
                # 🏆 Sauvegarde du meilleur modele
                try:
                    best_ckpt = {
                        "modele_state": _modele_brut().state_dict(),
                        "optimiseur_state": optimiseur.state_dict(),
                        "iteration": iteration,
                        "val_loss": val_actuel
                    }
                    torch.save(best_ckpt, os.path.join(MODEL_DIR, "best_model.pt"))
                    print(f"  🏆 Nouveau meilleur modele sauvegarde (val_loss: {val_actuel:.4f})")
                except Exception:
                    pass
            else:
                _sans_progres += 1
                print(f"  ⏳ Pas d'amélioration {_sans_progres}/{_patience} "
                      f"(meilleure val_loss : {_meilleur_val:.4f})")
                if _sans_progres >= _patience:
                    print(f"\n✅ Convergence atteinte — val_loss stable à {_meilleur_val:.4f}")
                    print("   L'IA a atteint son meilleur niveau sur ces données.")
                    break

    # ── Checkpoint + hot-reload (toutes les checkpoint_interval étapes) ──
    if iteration % checkpoint_interval == 0 and iteration > 0:
        # Sauvegarde du checkpoint
        try:
            ckpt = {
                "iteration"       : iteration,
                "modele_state"    : _modele_brut().state_dict(),
                "optimiseur_state": optimiseur.state_dict(),
                "log_data"        : log_data,
            }
            torch.save(ckpt, CHECKPOINT_FILE)
        except Exception as _e:
            print(f"  ⚠️  Checkpoint échoué : {_e}")

        # Rechargement à chaud si nouvelles sources ajoutées
        if DATA_MANIFEST and RELOAD_FLAG and os.path.exists(RELOAD_FLAG):
            print(f"\n  ♻️  Nouvelles données détectées (étape {iteration}) — "
                  "ré-tokenisation en cours...")
            os.remove(RELOAD_FLAG)
            data = recharger_donnees()
            n          = int(0.9 * len(data))
            train_data = data[:n]
            val_data   = data[n:]
            print(f"  ✅ {len(data):,} tokens disponibles — reprise de l'entraînement\n")

    # ── Etape d'entrainement ──────────────────────────────────────
    optimiseur.zero_grad(set_to_none=True)
    for micro_step in range(grad_accum_steps):
        xb, yb        = get_batch('train')
        with _amp_ctx:
            logits, perte = modele(xb, yb)
            perte = perte / grad_accum_steps
        scaler.scale(perte).backward()
    scaler.step(optimiseur)
    scaler.update()

    # Phase 1 protection : ralentit si RAM en alerte
    if _ralentir:
        time.sleep(_PROT_SEUILS["ralentir_ms"] / 1000)

    if _stop_demande:
        print("\n  Arrêt propre en cours — sauvegarde du modèle...")
        break

    if _QUICK_DEADLINE and time.time() >= _QUICK_DEADLINE:
        print(f"\n  ⏱️  30 minutes écoulées — arrêt du mode rapide.")
        break

# ================================================================
#  SAUVEGARDE FINALE DU MODÈLE
#  Deux formats complémentaires :
#    modele.pt          → poids + architecture complète (PyTorch natif)
#    modele.safetensors → poids seuls + métadonnées (format universel)
# ================================================================

print("\n" + "="*62)
print("  💾  SAUVEGARDE DU MODÈLE")
print("="*62)

MODELE_FILE      = os.path.join(MODEL_DIR, "modele.pt")
SAFETENSORS_FILE = os.path.join(MODEL_DIR, "modele.safetensors")

_date_iso = time.strftime("%Y-%m-%d %H:%M:%S")
_val_finale = (log_data["steps"][-1]["val_loss"] if log_data.get("steps") else None)

# ── modele.pt — tout ce qu'il faut pour recharger et utiliser le modèle ──
modele_final = {
    "modele_state": _modele_brut().state_dict(),
    "architecture": {
        "n_embd"    : n_embd,
        "n_head"    : n_head,
        "n_layer"   : n_layer,
        "block_size": block_size,
        "dropout"   : dropout,
        "vocab_size": taille_vocab,
        "nb_params" : nb_params,
    },
    "entrainement": {
        "tokenizer"   : "BPE",
        "preset"      : cfg.get("preset", "CUSTOM"),
        "nom_modele"  : nom_modele,
        "device"      : device,
        "iterations"  : log_data.get("iteration_courante", 0),
        "val_loss"    : round(_val_finale, 6) if _val_finale else None,
        "date"        : _date_iso,
    },
}
torch.save(modele_final, MODELE_FILE)
print(f"  OK modele.pt          -> {MODELE_FILE}")

# modele.safetensors -- format universel avec metadonnees
try:
    _meta = {
        "format"     : "wishai-bpe",
        "nom_modele" : nom_modele,
        "preset"     : cfg.get("preset", "CUSTOM"),
        "n_embd"     : str(n_embd),
        "n_head"     : str(n_head),
        "n_layer"    : str(n_layer),
        "block_size" : str(block_size),
        "dropout"    : str(dropout),
        "vocab_size" : str(taille_vocab),
        "nb_params"  : str(nb_params),
        "tokenizer"  : "BPE",
        "val_loss"   : str(round(_val_finale, 6)) if _val_finale else "n/a",
        "date"       : _date_iso,
        "device"     : device,
    }
    safe_save_file(_modele_brut().state_dict(), SAFETENSORS_FILE, metadata=_meta)
    print(f"  OK modele.safetensors -> {SAFETENSORS_FILE}")
except Exception as e:
    print(f"  ATTENTION : Impossible de sauvegarder en .safetensors : {e}")

# Nettoyage : supprime checkpoint.pt residuel s il existe
if os.path.exists(CHECKPOINT_FILE):
    try:
        os.remove(CHECKPOINT_FILE)
        print("  checkpoint.pt supprime (remplace par le modele final)")
    except Exception:
        pass

_taille_pt = os.path.getsize(MODELE_FILE) / 1e6 if os.path.exists(MODELE_FILE) else 0
_taille_st = os.path.getsize(SAFETENSORS_FILE) / 1e6 if os.path.exists(SAFETENSORS_FILE) else 0
print()
print(f"  Modele : {nb_params/1e6:.1f}M params  |  "
      f".pt {_taille_pt:.1f} Mo  |  .safetensors {_taille_st:.1f} Mo")
if _val_finale:
    print(f"  Val Loss finale : {_val_finale:.4f}")
print("="*62)
