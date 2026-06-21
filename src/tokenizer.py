"""
=================================================================
  TOKENIZER BPE (Byte Pair Encoding) — From Scratch
  Construit par Liam - appris de zéro
=================================================================

COMMENT ÇA MARCHE — EN 4 ÉTAPES :

  Étape 1 : vocab initial = tous les caractères uniques
    'a', 'b', 'c', ..., 'z', ' ', '.', ...  (~100 tokens)

  Étape 2 : on compte toutes les PAIRES adjacentes dans le texte
    "th"  → 500 000 fois
    "he"  → 400 000 fois
    "the" → 300 000 fois
    ...

  Étape 3 : on FUSIONNE la paire la plus fréquente → nouveau token
    't' + 'h' → 'th'   (token #101)
    On remplace toutes les occurrences de ['t','h'] par ['th']
    On recommence le comptage...

  Étape 4 : on répète 3900 fois pour atteindre vocab_size=4000
    Résultat final : 'the', 'ing', 'tion', ' the', 'Washington'...

AVANTAGE vs caractères :
  block_size=256 tokens :
  - Caractères : ~50 mots de contexte
  - BPE 4000   : ~180 mots de contexte  (3-4× plus !)

UTILISATION :
  python tokenizer.py           → entraîne et sauvegarde tokenizer.json
  python nanogpt_bpe.py         → entraîne le modèle avec BPE
=================================================================
"""

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import time
import hashlib
from collections import Counter, defaultdict

# ================================================================
# CONFIGURATION
# ================================================================
VOCAB_SIZE     = 4000          # tokens dans le vocabulaire final
SAMPLE_SIZE    = 5_000_000     # 5 Mo pour entraîner le tokenizer (assez représentatif)
TOKENIZER_FILE = "tokenizer.json"

# ================================================================
# CLASSE TOKENIZER BPE
# ================================================================

class TokenizerBPE:
    """
    Tokenizer BPE from scratch.

    Fonctionnement :
      • entrainer(texte)  → construit vocab + règles de fusion
      • encoder(texte)    → texte (str) → liste d'entiers
      • decoder(ids)      → liste d'entiers → texte (str)
      • sauvegarder()     → écrit tokenizer.json
      • charger()         → lit tokenizer.json

    Attributs internes :
      token_vers_id : { "the" : 412, "ing" : 87, ... }
      id_vers_token : { 412 : "the", 87 : "ing", ... }
      merges        : [ ("t","h"), ("th","e"), ... ]  ← règles dans l'ordre
    """

    def __init__(self):
        self.token_vers_id = {}
        self.id_vers_token = {}
        self.merges        = []
        self._cache        = {}          # cache mot → [ids] pour encoder vite
        self._pair_rang    = {}          # paire → rang (pour encoder vite)

    # ─────────────────────────────────────────────────────────────
    # ENTRAÎNEMENT
    # ─────────────────────────────────────────────────────────────

    def entrainer(self, texte, vocab_size=VOCAB_SIZE):
        """
        Entraîne le BPE sur le texte fourni.

        Algorithme :
        1. Découpe en mots (pré-tokenisation)
        2. Compte les fréquences de chaque mot
        3. Représente chaque mot comme (c1, c2, ..., cn, '</w>')
           '</w>' = marqueur de fin de mot
        4. Boucle : trouve la paire la plus fréquente → fusionne → répète
        """
        print(f"\n{'='*55}")
        print("  ENTRAÎNEMENT TOKENIZER BPE")
        print(f"{'='*55}")
        print(f"  Texte          : {len(texte):,} caractères ({len(texte)/1e6:.1f} Mo)")
        print(f"  Vocab cible    : {vocab_size} tokens")

        # ── 1. Comptage des mots ─────────────────────────────────
        print("\n  [1/4] Comptage des mots...")
        freq_mots = Counter(texte.split())

        # Représentation : "hello" → ('h','e','l','l','o','</w>')
        # '</w>' marque la fin du mot → distingue "the" en fin de mot vs milieu
        vocab_mots = {}
        for mot, freq in freq_mots.items():
            if len(mot) == 0:
                continue
            cle = tuple(list(mot) + ['</w>'])
            vocab_mots[cle] = vocab_mots.get(cle, 0) + freq

        print(f"  Mots uniques   : {len(vocab_mots):,}")

        # ── 2. Vocab initial = tous les caractères ───────────────
        print("\n  [2/4] Construction du vocab de base...")
        tous_chars = set()
        for mot in vocab_mots:
            for c in mot:
                tous_chars.add(c)
        tous_chars.add('</w>')

        vocab_base = sorted(tous_chars)
        n_base     = len(vocab_base)
        n_fusions  = vocab_size - n_base

        print(f"  Caractères uniques : {n_base}")
        print(f"  Fusions à faire    : {n_fusions}")

        # ── 3. Boucle BPE ────────────────────────────────────────
        print("\n  [3/4] Fusions BPE (cette étape prend 1-5 min)...")
        merges = []
        debut  = time.time()

        for i in range(n_fusions):

            # Compter toutes les paires adjacentes
            paires = defaultdict(int)
            for mot, freq in vocab_mots.items():
                for j in range(len(mot) - 1):
                    paires[(mot[j], mot[j+1])] += freq

            if not paires:
                print(f"  Plus de paires à fusionner à l'étape {i}.")
                break

            # Paire la plus fréquente
            meilleure  = max(paires, key=paires.get)
            freq_max   = paires[meilleure]

            # Si la fréquence max tombe à 1, pas la peine de continuer
            if freq_max < 2:
                print(f"  Fréquence max = 1 — arrêt à {i} fusions.")
                break

            # Fusion dans tout le vocabulaire de mots
            vocab_mots = self._fusionner(meilleure, vocab_mots)
            merges.append(meilleure)

            # Affiche la progression tous les 200 pas
            if (i + 1) % 200 == 0:
                ecoul = time.time() - debut
                reste = ecoul / (i + 1) * (n_fusions - i - 1)
                token_fusionne = ''.join(meilleure)
                print(f"    {i+1:4d}/{n_fusions}  {token_fusionne!r:25s}  "
                      f"freq={freq_max:>7,}   {ecoul:.0f}s écoulé / ~{reste:.0f}s restant")

        # ── 4. Construire le vocabulaire final ───────────────────
        print("\n  [4/4] Construction du vocabulaire final...")
        self.token_vers_id = {}
        self.id_vers_token = {}

        # Tokens de base en premier
        for idx, token in enumerate(vocab_base):
            self.token_vers_id[token] = idx
            self.id_vers_token[idx]   = token

        # Tokens fusionnés (dans l'ordre des fusions)
        idx         = n_base
        deja_vus    = set(vocab_base)
        for paire in merges:
            nouveau = ''.join(paire)
            if nouveau not in deja_vus:
                self.token_vers_id[nouveau] = idx
                self.id_vers_token[idx]     = nouveau
                deja_vus.add(nouveau)
                idx += 1

        self.merges     = merges
        self._pair_rang = {paire: rang for rang, paire in enumerate(merges)}
        self._cache     = {}

        # Résumé
        exemples_bpe = [t for t in list(self.token_vers_id.keys()) if len(t) > 2][:10]
        print(f"\n  ✅ Entraînement terminé en {time.time()-debut:.1f}s")
        print(f"  Taille vocab final : {len(self.token_vers_id)} tokens")
        print(f"  Exemples de tokens : {exemples_bpe}")

    def _fusionner(self, paire, vocab_mots):
        """
        Applique une fusion dans tout le vocabulaire de mots.
        Exemple : paire=('t','h') → remplace ('t','h') par ('th',) partout.
        """
        a, b   = paire
        fusion = a + b
        nouveau_vocab = {}

        for mot, freq in vocab_mots.items():
            nouveau_mot = []
            i = 0
            while i < len(mot):
                # Si on trouve la paire → fusion
                if i < len(mot) - 1 and mot[i] == a and mot[i+1] == b:
                    nouveau_mot.append(fusion)
                    i += 2
                else:
                    nouveau_mot.append(mot[i])
                    i += 1
            nouveau_vocab[tuple(nouveau_mot)] = freq

        return nouveau_vocab

    # ─────────────────────────────────────────────────────────────
    # ENCODAGE
    # ─────────────────────────────────────────────────────────────

    def _encoder_mot(self, mot):
        """
        Encode un seul mot en liste d'IDs (avec cache).

        Algorithme : applique les fusions dans l'ordre de leur rang.
        Exemple : "the" → ['t','h','e','</w>'] → ['th','e','</w>'] → ['the','</w>'] → [412, 87]
        """
        if mot in self._cache:
            return self._cache[mot]

        # Représentation initiale
        tokens = list(mot) + ['</w>']

        # Applique les fusions dans l'ordre de rang (greedy)
        while len(tokens) > 1:
            # Trouve la paire applicable de rang le plus bas
            meilleur_rang = len(self.merges)
            meilleur_i    = -1

            for i in range(len(tokens) - 1):
                paire = (tokens[i], tokens[i+1])
                rang  = self._pair_rang.get(paire, len(self.merges))
                if rang < meilleur_rang:
                    meilleur_rang = rang
                    meilleur_i    = i

            if meilleur_i == -1:
                break   # plus aucune fusion applicable

            # Applique la fusion
            fusion  = tokens[meilleur_i] + tokens[meilleur_i + 1]
            tokens  = tokens[:meilleur_i] + [fusion] + tokens[meilleur_i + 2:]

        ids = [self.token_vers_id.get(t, 0) for t in tokens]
        self._cache[mot] = ids
        return ids

    def encoder(self, texte):
        """
        Convertit un texte en liste d'IDs.

        Exemple :
          "The cat"  →  [412, 87, 203, 5, 99, 201]
        """
        ids  = []
        mots = texte.split(' ')
        for mot in mots:
            if mot:
                ids.extend(self._encoder_mot(mot))
        return ids

    def encoder_dataset(self, texte):
        """
        Encode le dataset complet efficacement.
        Utilise le cache : chaque mot unique n'est encodé qu'une seule fois.
        """
        from itertools import chain as _chain

        mots  = texte.split()          # split() gère espaces + newlines
        total = len(mots)
        jalon = max(1, total // 40)    # affiche tous les ~2.5%

        # ── passe 1 : encoder les mots uniques + barre de progression ──
        cache_local = {}
        for i, mot in enumerate(mots):
            if mot not in cache_local:
                cache_local[mot] = self._encoder_mot(mot)
            if i % jalon == 0:
                pct    = i / total * 100
                barlen = 30
                filled = int(barlen * i / total)
                bar    = "█" * filled + "░" * (barlen - filled)
                print(f"\r     [{bar}] {pct:5.1f}%  ({i/1_000_000:.1f}/{total/1_000_000:.1f}M mots)",
                      end="", flush=True)

        # ── passe 2 : assemblage rapide ──
        print(f"\r     [{'█'*30}] 100.0%  assemblage...", end="", flush=True)
        ids = list(_chain.from_iterable(cache_local[m] for m in mots))
        print(f"\r     [{'█'*30}] 100.0%  {total/1_000_000:.1f}M mots → {len(ids):,} tokens  ")
        return ids

    def decoder(self, ids):
        """
        Convertit une liste d'IDs en texte.

        '</w>' marque la fin d'un mot → on le remplace par un espace.
        """
        texte = ''
        for id in ids:
            token = self.id_vers_token.get(id, '')
            if token.endswith('</w>'):
                texte += token[:-4] + ' '   # retire </w>, ajoute espace
            else:
                texte += token
        return texte.strip()

    # ─────────────────────────────────────────────────────────────
    # SAUVEGARDE / CHARGEMENT
    # ─────────────────────────────────────────────────────────────

    def sauvegarder(self, chemin=TOKENIZER_FILE, dataset_sig=None, dataset_file=None):
        """Sauvegarde le tokenizer dans un fichier JSON.

        dataset_sig / dataset_file : empreinte des données d'entraînement,
        utilisée pour détecter un changement de dataset et réentraîner
        automatiquement (voir tokenizer_a_jour())."""
        data = {
            "vocab_size":    len(self.token_vers_id),
            "dataset_sig":   dataset_sig,
            "dataset_file":  dataset_file,
            "token_vers_id": self.token_vers_id,
            "id_vers_token": {str(k): v for k, v in self.id_vers_token.items()},
            "merges":        [list(m) for m in self.merges],
        }
        with open(chemin, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        taille = os.path.getsize(chemin) / 1024
        print(f"  💾 Tokenizer sauvegardé : {chemin} ({taille:.0f} Ko)")

    def charger(self, chemin=TOKENIZER_FILE):
        """Charge le tokenizer depuis un fichier JSON."""
        with open(chemin, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token_vers_id = data["token_vers_id"]
        self.id_vers_token = {int(k): v for k, v in data["id_vers_token"].items()}
        self.merges        = [tuple(m) for m in data["merges"]]
        self._pair_rang    = {tuple(m): rang for rang, m in enumerate(self.merges)}
        self._cache        = {}
        print(f"  Tokenizer chargé : {len(self.token_vers_id)} tokens, {len(self.merges)} fusions")

    @property
    def taille_vocab(self):
        return len(self.token_vers_id)

# ================================================================
# DÉTECTION DE CHANGEMENT DE DATASET (réentraînement automatique)
# ================================================================
CANDIDATS_DATA = [
    "data/en/data.txt",
    "data/fr/data.txt",
    "data/multi/data.txt",
    "data/data.txt",
]

def trouver_data_file(candidats=CANDIDATS_DATA):
    """Retourne le premier fichier de données non-vide, ou None."""
    for c in candidats:
        if os.path.exists(c) and os.path.getsize(c) > 1024:
            return c
    return None

def signature_dataset(data_file, sample_size=SAMPLE_SIZE, vocab_size=None):
    """Empreinte du dataset réellement utilisé pour entraîner le BPE.

    Le tokenizer n'apprend que sur les premiers `sample_size` octets : on hash
    donc exactement cette tranche (+ nom du fichier, taille totale, vocab cible).
    Si le dataset change, la signature change → réentraînement automatique.
    """
    if vocab_size is None:          # lu à l'exécution (évite le piège du défaut figé)
        vocab_size = VOCAB_SIZE
    h = hashlib.sha256()
    h.update(os.path.basename(data_file).encode("utf-8"))
    h.update(str(os.path.getsize(data_file)).encode("utf-8"))
    h.update(str(vocab_size).encode("utf-8"))
    with open(data_file, "rb") as f:
        h.update(f.read(sample_size))
    return h.hexdigest()

def tokenizer_a_jour(chemin_tok=TOKENIZER_FILE, data_file=None):
    """Indique si tokenizer.json existe ET correspond au dataset actuel.

    Retourne (a_jour: bool, raison: str).
    """
    if not os.path.exists(chemin_tok):
        return False, "tokenizer.json introuvable"
    if data_file is None:
        data_file = trouver_data_file()
    if data_file is None:
        # Pas de données pour vérifier : on conserve le tokenizer existant.
        return True, "aucune donnée pour vérifier — tokenizer conservé"
    try:
        with open(chemin_tok, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False, "tokenizer.json illisible — réentraînement"
    # Le hash encode déjà le fichier + sa taille + le vocab_size cible :
    # tout changement de dataset OU de vocab cible modifie la signature.
    if meta.get("dataset_sig") != signature_dataset(data_file):
        ancien = meta.get("dataset_file", "?")
        return False, f"dataset ou vocab modifié ({ancien} → {data_file})"
    return True, "tokenizer à jour"

# ================================================================
# SCRIPT PRINCIPAL — entraîne et sauvegarde
# ================================================================
if __name__ == "__main__":

    # Cherche le fichier de données
    DATA_FILE = trouver_data_file()

    if DATA_FILE is None:
        print("❌ Aucun fichier de données trouvé !")
        print("   Lance d'abord telecharger.py")
        raise SystemExit(1)

    # ── Réentraînement automatique uniquement si nécessaire ──────
    # (évite d'avoir à supprimer tokenizer.json à la main quand on
    #  change de dataset)
    a_jour, raison = tokenizer_a_jour(TOKENIZER_FILE, DATA_FILE)
    if a_jour:
        print(f"  ✅ Tokenizer déjà à jour ({raison}) — pas de réentraînement.")
        raise SystemExit(0)
    print(f"  ↻ Réentraînement du tokenizer : {raison}")

    taille_totale = os.path.getsize(DATA_FILE) / 1_000_000
    print(f"Données : {DATA_FILE} ({taille_totale:.0f} Mo)")

    # Lit un échantillon représentatif (5 Mo suffit pour BPE)
    print(f"Lecture de {SAMPLE_SIZE/1e6:.0f} Mo pour l'entraînement...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        echantillon = f.read(SAMPLE_SIZE)

    # Entraîne le tokenizer (et mémorise la signature du dataset)
    sig = signature_dataset(DATA_FILE)
    tok = TokenizerBPE()
    tok.entrainer(echantillon, vocab_size=VOCAB_SIZE)
    tok.sauvegarder(TOKENIZER_FILE, dataset_sig=sig, dataset_file=DATA_FILE)

    # ── Test de cohérence ────────────────────────────────────────
    print("\n" + "="*55)
    print("  TEST DU TOKENIZER")
    print("="*55)

    phrases_test = [
        "The development of artificial intelligence",
        "Washington announced a new policy",
        "hello world",
    ]

    for phrase in phrases_test:
        ids    = tok.encoder(phrase)
        tokens = [tok.id_vers_token[i] for i in ids]
        decode = tok.decoder(ids)
        print(f"\n  Texte   : {phrase!r}")
        print(f"  Tokens  : {tokens}")
        print(f"  IDs     : {ids}")
        print(f"  Décodé  : {decode!r}")
        ok = "✅" if decode.strip() == phrase.strip() else "⚠️"
        print(f"  {ok} Cohérence encodage/décodage")

    print("\n" + "="*55)
    print("  Tokenizer prêt !")
    print("  Lance maintenant : python nanogpt_bpe.py")
    print("="*55)
