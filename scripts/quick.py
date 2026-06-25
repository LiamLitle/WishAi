# quick.py — Lance WishAI en mode rapide, zéro configuration
#
#   python quick.py
#
# Ce script :
#   1. Télécharge TinyStories (anglais, 50 Mo) si aucune donnée n'existe
#   2. Entraîne le tokenizer BPE automatiquement
#   3. Démarre le dashboard de surveillance
#   4. Lance l'entraînement avec le preset MINI (~20M paramètres)
#
# Aucune question posée. Ctrl+C pour arrêter.
# ============================================================

import os, sys, subprocess, time, json, webbrowser, shutil

# ── Utilitaires de base (définis en premier car utilisés partout) ────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
SYS  = os.path.join(ROOT, "system")

def run(script, args=None):
    cmd = [sys.executable, os.path.join(SRC, script)]
    if args:
        cmd += args
    try:
        return subprocess.run(cmd, cwd=ROOT).returncode
    except KeyboardInterrupt:
        return 1

def bg(script):
    return subprocess.Popen([sys.executable, os.path.join(SRC, script)], cwd=ROOT)

# ── Dataset téléchargé automatiquement en mode quick ────────────────
_QUICK_DATASET      = "inst_tinystories"   # TinyStories : idéal petits modèles
_QUICK_DATASET_SLUG = "tinystories"        # slug du fichier généré par telecharger.py
_QUICK_DL_MO        = 50                   # 50 Mo = assez pour 20M params, peu d'overfitting

DATA_DIR   = os.path.join(ROOT, "data")
DATA_QUICK = os.path.join(DATA_DIR, "data.txt")

# ============================================================
#  TEXTE DE DÉMONSTRATION
#  Utilisé uniquement si aucun fichier de données n'existe.
#  Contient des phrases françaises et anglaises variées
#  — suffisant pour voir le modèle apprendre en quelques minutes.
# ============================================================

_PHRASES = [
    # Langue & apprentissage
    "Le langage est le fondement de toute compréhension humaine.",
    "Apprendre une nouvelle langue ouvre une fenêtre sur le monde.",
    "Les mots sont les briques avec lesquelles on construit la pensée.",
    "L'écriture permet de fixer les idées dans le temps.",
    "Lire chaque jour enrichit le vocabulaire et affine l'esprit.",
    "Language is the road map of a culture.",
    "Words are the tools of thought.",
    "Reading is to the mind what exercise is to the body.",
    "Every language is a different vision of life.",
    "Writing is thinking on paper.",
    # Nature
    "Le soleil se lève chaque matin sur les collines dorées.",
    "La pluie efface les traces sur le sable de la plage.",
    "Les étoiles brillent dans le ciel d'une nuit sans lune.",
    "Le vent souffle doucement entre les feuilles des chênes.",
    "La montagne garde le silence comme un vieux sage.",
    "The river flows quietly through the valley.",
    "Clouds gather above the ancient forest.",
    "Snow covers the fields in a white silence.",
    "The sea stretches endlessly under the setting sun.",
    "Flowers bloom in the morning light.",
    # Science & technologie
    "L'intelligence artificielle apprend à partir de données massives.",
    "Un réseau de neurones imite le fonctionnement du cerveau humain.",
    "Le gradient descend vers le minimum de la fonction de perte.",
    "L'attention permet au modèle de relier des mots distants.",
    "Chaque paramètre est ajusté par rétropropagation du gradient.",
    "A neural network learns by adjusting millions of parameters.",
    "The transformer architecture revolutionized natural language processing.",
    "Backpropagation computes gradients layer by layer.",
    "Training loss measures how well the model fits the data.",
    "Validation loss tells us how well the model generalizes.",
    # Philosophie
    "Penser, c'est déjà agir sur le monde.",
    "La curiosité est le moteur de toute découverte.",
    "Le doute est le début de la sagesse.",
    "Comprendre le passé aide à construire l'avenir.",
    "La liberté commence là où l'ignorance finit.",
    "To think is to act upon the world.",
    "Curiosity is the engine of all discovery.",
    "Doubt is the beginning of wisdom.",
    "Understanding the past helps build the future.",
    "Knowledge is the only treasure that grows when shared.",
    # Histoires courtes
    "Il était une fois un jeune garçon qui voulait comprendre les étoiles.",
    "Elle marchait lentement sur le chemin de pierre, pensant à demain.",
    "Le vieux professeur ouvrit son livre et commença à lire à voix haute.",
    "Une lumière brillait à la fenêtre du grenier chaque nuit.",
    "L'enfant tendit la main vers le papillon qui s'envola aussitôt.",
    "A young girl sat by the window watching the rain fall.",
    "The old man smiled as he turned the last page of the book.",
    "A dog barked in the distance as night fell over the village.",
    "She wrote a letter and sealed it with care.",
    "The clock on the wall ticked quietly in the empty room.",
    # Descriptions
    "La ville s'éveille dans un concert de klaxons et de voix.",
    "Les toits rouges du village brillaient sous la pluie d'automne.",
    "Un café chaud sur une table en bois un matin d'hiver.",
    "Le marché sentait la lavande, le pain frais et la menthe.",
    "La bibliothèque était silencieuse, remplie de milliers de livres.",
    "The city wakes with the sound of traffic and voices.",
    "Autumn leaves drift slowly from the trees along the street.",
    "The smell of fresh bread fills the small bakery.",
    "Candles flicker on the wooden tables of the old inn.",
    "The library holds centuries of knowledge between its walls.",
    # Motivations
    "Chaque erreur est une occasion d'apprendre quelque chose de nouveau.",
    "La persévérance vient à bout de tous les obstacles.",
    "Petit à petit, l'oiseau fait son nid.",
    "Ce qui ne nous tue pas nous rend plus forts.",
    "Commence par faire ce qui est nécessaire, puis ce qui est possible.",
    "Every mistake is a chance to learn something new.",
    "Perseverance overcomes all obstacles in time.",
    "Step by step, great things are accomplished.",
    "What does not destroy us makes us stronger.",
    "Start with what is necessary, then what is possible.",
    # Dialogue simple
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Je vais très bien, merci. Et vous ?",
    "Quel temps fait-il dehors ce matin ?",
    "Il fait beau, le ciel est bleu et il n'y a pas de vent.",
    "Avez-vous lu un bon livre récemment ?",
    "Good morning, how are you today?",
    "I am doing well, thank you. And yourself?",
    "What is the weather like outside this morning?",
    "It is sunny and the sky is clear.",
    "Have you read a good book recently?",
    # Faits simples
    "La Terre tourne autour du Soleil en trois cent soixante-cinq jours.",
    "L'eau se compose de deux atomes d'hydrogène et d'un atome d'oxygène.",
    "La lumière voyage à trois cent mille kilomètres par seconde.",
    "Le cerveau humain contient environ quatre-vingt-six milliards de neurones.",
    "Paris est la capitale de la France depuis le dixième siècle.",
    "The Earth orbits the Sun in three hundred and sixty-five days.",
    "Water is made of two hydrogen atoms and one oxygen atom.",
    "Light travels at three hundred thousand kilometers per second.",
    "The human brain contains around eighty-six billion neurons.",
    "Paris has been the capital of France since the tenth century.",
    # Répétitions pour densité
    "L'apprentissage prend du temps mais chaque étape compte.",
    "Le modèle apprend à prédire le prochain token.",
    "La loss diminue progressivement au fil des itérations.",
    "Le vocabulaire BPE représente le texte de façon efficace.",
    "L'entraînement continue jusqu'à convergence.",
    "The model learns to predict the next token in the sequence.",
    "The loss decreases gradually over training iterations.",
    "BPE vocabulary represents text efficiently with fewer tokens.",
    "Training continues until the model converges.",
    "Each iteration brings the model closer to understanding language.",
]


def donnees_existent():
    def non_vide(p):
        return os.path.exists(p) and os.path.getsize(p) > 1024
    if non_vide(DATA_QUICK):
        return True
    for langue in ["fr", "en", "multi"]:
        if non_vide(os.path.join(DATA_DIR, langue, "data.txt")):
            return True
    return False


def creer_texte_demo():
    """Fallback : génère ~400Ko de texte de démo si le téléchargement échoue."""
    import random
    random.seed(42)
    os.makedirs(DATA_DIR, exist_ok=True)
    lignes = []
    while len("\n".join(lignes)) < 400_000:
        lignes.append(random.choice(_PHRASES))
    texte = "\n".join(lignes)
    with open(DATA_QUICK, "w", encoding="utf-8") as f:
        f.write(texte)
    taille_ko = len(texte) // 1000
    print(f"  ✅ Texte de démo créé : data/data.txt ({taille_ko} Ko)")


def telecharger_donnees_quick():
    """
    Télécharge TinyStories (anglais, 50 Mo) via telecharger.py — zéro interaction.
    TinyStories est spécifiquement conçu pour les petits modèles de langage (~20M params).
    Résultat : histoires courtes en anglais, vocabulaire varié, peu d'overfitting.
    Fallback vers le texte de démo si le réseau est indisponible.
    """
    print("  ⬇️  Téléchargement : TinyStories (anglais, 50 Mo)")
    print("     → Histoires courtes conçues pour les petits modèles (~20M params)")
    print()

    code = run("telecharger.py", ["--download", _QUICK_DATASET, "--mo", str(_QUICK_DL_MO)])

    # Le fichier atterrit dans data/en/sources/tinystories.txt
    _src = os.path.join(DATA_DIR, "en", "sources", _QUICK_DATASET_SLUG + ".txt")

    if code == 0 and os.path.exists(_src) and os.path.getsize(_src) > 10_000:
        # Copie vers data/data.txt (utilisé par nanogpt_bpe en mode quick)
        os.makedirs(DATA_DIR, exist_ok=True)
        shutil.copy2(_src, DATA_QUICK)
        # Copie aussi dans data/en/data.txt (détecté par choisir_donnees())
        _en_dir = os.path.join(DATA_DIR, "en")
        os.makedirs(_en_dir, exist_ok=True)
        shutil.copy2(_src, os.path.join(_en_dir, "data.txt"))
        # Supprime le cache BPE si présent (tokenizer doit être re-entraîné)
        for _cache in [
            os.path.join(DATA_DIR, "bpe_cache.pt"),
            os.path.join(DATA_DIR, "en", "bpe_cache.pt"),
        ]:
            if os.path.exists(_cache):
                try: os.remove(_cache)
                except Exception: pass
        _taille = os.path.getsize(DATA_QUICK) // 1_000_000
        print(f"  ✅ TinyStories prêt : {_taille} Mo")
    else:
        print("  ⚠️  Téléchargement échoué — texte de démo utilisé à la place.")
        creer_texte_demo()


# ============================================================
#  DÉMARRAGE
# ============================================================

print()
print("  ╔══════════════════════════════════════╗")
print("  ║   WishAI Quick  ⚡  Mode 20M params  ║")
print("  ╚══════════════════════════════════════╝")
print()

# Dependances (installe uniquement ce qui manque, < 1s si deja installe)
run("require.py")

# Données
print("\n  📂  DONNÉES")
print("  " + "-" * 40)
if not donnees_existent():
    telecharger_donnees_quick()
    print("  💡 Pour encore plus de données : ouvre library.html depuis le dashboard.")
else:
    print("  ✅ Données existantes détectées — on les utilise.")

# Tokenizer
print("\n  🔤  TOKENIZER")
print("  " + "-" * 40)
run("tokenizer.py")

# Dashboard (arrière-plan)
print("\n  📊  DASHBOARD")
print("  " + "-" * 40)
_dash_url_file = os.path.join(SYS, "dashboard_url.json")
if os.path.exists(_dash_url_file):
    try:
        os.remove(_dash_url_file)
    except Exception:
        pass
_dash_proc = bg("dashboard.py")

# Attente URL dashboard
_dash_url = None
for _ in range(40):
    if os.path.exists(_dash_url_file):
        try:
            with open(_dash_url_file, encoding="utf-8") as _f:
                _info = json.load(_f)
                _dash_url = _info.get("url") or _info.get("dash_url")
            break
        except Exception:
            pass
    time.sleep(0.25)

# Moniteur mémoire
bg("monitor.py")

# Session
with open(os.path.join(SYS, "session.json"), "w", encoding="utf-8") as _sf:
    json.dump({"id": str(int(time.time())), "status": "starting", "ts": time.time()}, _sf)

# Ouvrir dashboard
if _dash_url:
    print(f"  ✅ Dashboard : {_dash_url}")
    webbrowser.open(_dash_url)
else:
    _html = os.path.join(ROOT, "dashboard.html")
    print(f"  Dashboard : ouvre dashboard.html dans ton navigateur")

time.sleep(1)

# ── Écriture control.json ─────────────────────────────────────
_ctrl = os.path.join(SYS, "control.json")
_tmp  = _ctrl + ".tmp"

# ============================================================
#  ENTRAÎNEMENT — PRESET MINI (~20M params), ZÉRO QUESTION
# ============================================================
print()
print("  🚀  ENTRAÎNEMENT")
print("  " + "-" * 40)
print("  Modèle   : quick (~20M paramètres)")
print("  Durée    : automatique (arrêt à la convergence)")
print("  Ctrl+C   : arrêt propre à tout moment")
print()

def _ecrire_ctrl(commande):
    """Ecriture atomique de control.json (fallback direct sur Windows si verrouille)."""
    data = {"commande": commande, "timestamp": time.time()}
    try:
        with open(_tmp, "w", encoding="utf-8") as _f:
            json.dump(data, _f)
        os.replace(_tmp, _ctrl)
    except PermissionError:
        with open(_ctrl, "w", encoding="utf-8") as _f:
            json.dump(data, _f)

try:
    while True:
        _ecrire_ctrl("run")

        ret = run("nanogpt_bpe.py", ["--quick"])

        # Vérifier si go.py doit relancer (arrêt critique RAM/temp)
        try:
            with open(_ctrl, encoding="utf-8") as _f:
                _cmd = json.load(_f).get("commande", "stop")
        except Exception:
            _cmd = "stop"

        if _cmd == "reprendre":
            print("\n  Conditions OK — reprise de l’entraînement...\n")
            time.sleep(2)
            continue

        break
except KeyboardInterrupt:
    pass

print("\n  Terminé. Pour discuter avec ton IA :")
print("  ./wish chat --terminal\n")
