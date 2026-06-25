# chat.py — Interface de chat WishAI
#
#   ./wish chat              → interface web (navigateur)
#   ./wish chat --terminal   → mode terminal interactif
#
# ============================================================

import os, sys, glob, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

URL_FILE = os.path.join(ROOT, "system", "chat_url.json")

TERMINAL_MODE = "--terminal" in sys.argv or "-t" in sys.argv


# ════════════════════════════════════════════════════════════════
#  HELPERS COMMUNS
# ════════════════════════════════════════════════════════════════

def lister_modeles():
    """Retourne la liste des modèles disponibles dans model/."""
    modeles = []
    for f in sorted(glob.glob(os.path.join(ROOT, "model", "*", "modele.pt"))):
        nom = os.path.basename(os.path.dirname(f))
        taille = os.path.getsize(f) / 1_000_000
        modeles.append({"fichier": f, "nom": nom, "taille": taille})
    return modeles


# ════════════════════════════════════════════════════════════════
#  MODE WEB (défaut)
# ════════════════════════════════════════════════════════════════

def mode_web():
    modeles = lister_modeles()

    print("\n  ╔══════════════════════════════════════╗")
    print("  ║         WishAI  —  Chat Web          ║")
    print("  ╚══════════════════════════════════════╝\n")

    if not modeles:
        print("  ⚠️  Aucun modèle trouvé dans model/")
        print("  Lance d'abord un entraînement : ./wish go\n")
    else:
        print(f"  {len(modeles)} modèle(s) disponible(s) — sélection dans l'interface\n")

    print("  Démarrage du serveur...\n")

    if os.path.exists(URL_FILE):
        try:
            os.remove(URL_FILE)
        except Exception:
            pass

    proc = subprocess.Popen(
        [sys.executable, os.path.join(SRC, "chat_server.py")],
        cwd=ROOT
    )

    # Attendre que chat_server.py écrive chat_url.json
    for _ in range(40):
        if os.path.exists(URL_FILE):
            break
        time.sleep(0.25)

    if not os.path.exists(URL_FILE):
        print("  ⚠️  Le serveur n'a pas démarré correctement.")
        print("  Vérifie les logs ci-dessus.\n")
    else:
        try:
            import json
            with open(URL_FILE, encoding="utf-8") as f:
                info = json.load(f)
            url = info.get("url") or info.get("chat_url")
            if url:
                print(f"  Chat ouvert : {url}")
        except Exception:
            pass

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n  Serveur arrêté.")


# ════════════════════════════════════════════════════════════════
#  MODE TERMINAL
# ════════════════════════════════════════════════════════════════

def mode_terminal():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n  ╔══════════════════════════════════════╗")
    print("  ║       WishAI  —  Chat Terminal       ║")
    print("  ╚══════════════════════════════════════╝\n")

    # ── Liste des modèles ────────────────────────────────────────
    modeles = lister_modeles()

    if not modeles:
        print("  ❌ Aucun modèle trouvé dans model/")
        print("  Lance d'abord un entraînement : ./wish go\n")
        return

    print("  Modèles disponibles :\n")
    for i, m in enumerate(modeles):
        print(f"  [{i + 1}]  {m['nom']:<22} {m['taille']:.0f} Mo")

    print()
    choix = input("  Choix [1] > ").strip()
    try:
        idx = int(choix) - 1
        if not (0 <= idx < len(modeles)):
            raise ValueError
    except Exception:
        idx = 0

    modele_info = modeles[idx]
    nom = modele_info["nom"]
    print(f"\n  → Chargement de {nom}...\n")

    # ── Chargement ───────────────────────────────────────────────
    from model import WishAI_BPE, ConfigModele

    ckpt = torch.load(modele_info["fichier"], map_location=device, weights_only=False)

    if "char_vers_int" in ckpt:
        # Char-level (legacy)
        c2i = ckpt["char_vers_int"]
        i2c = ckpt["int_vers_char"]
        encoder = lambda t: [c2i.get(c, 0) for c in t]
        decoder = lambda ids: "".join([i2c.get(i, "?") for i in ids])
        taille_vocab = ckpt["taille_vocab"]
        hp = ckpt["hyperparams"]
    else:
        # BPE
        from tokenizer import TokenizerBPE
        tok_path = os.path.join(ROOT, "model", nom, "tokenizer.json")
        if not os.path.exists(tok_path):
            tok_path = os.path.join(ROOT, "system", "tokenizer.json")
        tok = TokenizerBPE()
        tok.charger(tok_path)
        encoder = tok.encoder
        decoder = tok.decoder
        hp = ckpt.get("architecture") or ckpt["hyperparams"]
        taille_vocab = hp.get("vocab_size") or ckpt.get("taille_vocab") or hp.get("taille_vocab")

    cfg = ConfigModele(
        vocab_size  = taille_vocab,
        n_embd      = hp["n_embd"],
        n_head      = hp["n_head"],
        n_layer     = hp["n_layer"],
        block_size  = hp["block_size"],
        dropout     = hp.get("dropout", 0.0),
    )
    modele = WishAI_BPE(cfg).to(device)
    modele.load_state_dict(ckpt["modele_state"])
    modele.eval()

    nb_params = sum(p.numel() for p in modele.parameters())
    print(f"  ✅ {nom}  —  {nb_params / 1e6:.1f}M params  —  {device.upper()}\n")

    # ── Boucle interactive ───────────────────────────────────────
    print("  ─────────────────────────────────────────────────")
    print("  Tape un début de phrase → l'IA continue")
    print("  t=0.8   → température    n=300 → longueur")
    print("  q       → quitter")
    print("  ─────────────────────────────────────────────────\n")

    temperature = 0.8
    nb_tokens   = 300

    while True:
        try:
            entree = input("  Toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Au revoir !")
            break

        if entree.lower() == "q":
            print("  Au revoir !")
            break

        if entree.startswith("t="):
            try:
                temperature = float(entree[2:])
                print(f"  → Température : {temperature}")
            except Exception:
                print("  Format : t=0.8")
            continue

        if entree.startswith("n="):
            try:
                nb_tokens = int(entree[2:])
                print(f"  → Longueur : {nb_tokens} tokens")
            except Exception:
                print("  Format : n=300")
            continue

        with torch.no_grad():
            ids = encoder(entree) if entree else []
            idx_t = (
                torch.tensor([ids], dtype=torch.long, device=device)
                if ids
                else torch.zeros((1, 1), dtype=torch.long, device=device)
            )
            sortie = modele.generer(idx_t, max_nouveaux_tokens=nb_tokens, temperature=temperature)
            texte  = decoder(sortie[0].tolist())

        # Coupe à la dernière phrase complète
        for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            pos = texte.rfind(sep)
            if pos != -1 and pos > len(texte) // 2:
                texte = texte[: pos + 1]
                break

        print(f"\n  IA  > {texte}\n")


# ════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════════

if TERMINAL_MODE:
    mode_terminal()
else:
    mode_web()
