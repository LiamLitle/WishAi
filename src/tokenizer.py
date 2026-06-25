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
_SAMPLE_MIN    = 5_000_000     # 5 Mo minimum
_SAMPLE_MAX    = 50_000_000    # 50 Mo maximum (au-delà, qualité BPE plafonne)
_SAMPLE_RATIO  = 0.15          # 15 % du fichier par défaut
TOKENIZER_FILE = os.path.join("system", "tokenizer.json")

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

        # ── 3. Boucle BPE (Optimisée par Index Inversé) ──────────
        print("\n  [3/4] Fusions BPE (Optimisation Index Inversé - Très Rapide)...")
        merges = []
        debut  = time.time()

        # Initialisation : comptage initial des paires et construction de l'index inversé
        paires = defaultdict(int)
        index = defaultdict(set)
        
        for mot, freq in vocab_mots.items():
            # Remplir les paires
            for j in range(len(mot) - 1):
                paires[(mot[j], mot[j+1])] += freq
            # Remplir l'index inversé
            for sym in set(mot):
                index[sym].add(mot)

        for i in range(n_fusions):
            if not paires:
                print(f"  Plus de paires à fusionner à l'étape {i}.")
                break

            # Paire la plus fréquente
            meilleure = max(paires, key=paires.get)
            freq_max  = paires[meilleure]

            if freq_max < 2:
                print(f"  Fréquence max = 1 — arrêt à {i} fusions.")
                break

            a, b = meilleure
            fusion = a + b
            merges.append(meilleure)

            # Intersection pour trouver exactement les mots qui contiennent potentiellement la paire
            mots_a_modifier = index[a] & index[b]

            for mot in list(mots_a_modifier):
                freq = vocab_mots[mot]

                nouveau_mot = []
                j = 0
                modifie = False
                while j < len(mot):
                    if j < len(mot) - 1 and mot[j] == a and mot[j+1] == b:
                        nouveau_mot.append(fusion)
                        j += 2
                        modifie = True
                    else:
                        nouveau_mot.append(mot[j])
                        j += 1

                if modifie:
                    nouveau_mot = tuple(nouveau_mot)

                    # 1. Soustraire les anciennes paires
                    for k in range(len(mot) - 1):
                        paires[(mot[k], mot[k+1])] -= freq

                    # 2. Ajouter les nouvelles paires
                    for k in range(len(nouveau_mot) - 1):
                        paires[(nouveau_mot[k], nouveau_mot[k+1])] += freq

                    # 3. Mettre à jour l'index
                    for sym in set(mot):
                        index[sym].remove(mot)
                    for sym in set(nouveau_mot):
                        index[sym].add(nouveau_mot)

                    # 4. Mettre à jour vocab_mots
                    del vocab_mots[mot]
                    vocab_mots[nouveau_mot] = freq

            # La paire courante n'existe plus (elle a été fusionnée)
            if meilleure in paires:
                del paires[meilleure]

            # Affiche la progression tous les 500 pas (car c'est très rapide maintenant)
            if (i + 1) % 500 == 0:
                ecoul = time.time() - debut
                reste = ecoul / (i + 1) * (n_fusions - i - 1)
                token_fusionne = ''.join(meilleure)
                print(f"    {i+1:4d}/{n_fusions}  {token_fusionne!r:25s}  "
                      f"freq={freq_max:>7,}   {ecoul:.1f}s écoulé / ~{reste:.1f}s restant")

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

def trouver_sources_donnees():
    """Recherche tous les manifests et tous les dossiers contenant des .txt dans data/"""
    import glob
    sources = []
    
    # 1. Manifests officiels
    for langue in ["fr", "en", "multi"]:
        manifest = os.path.join("data", langue, "manifest.json")
        if os.path.exists(manifest):
            try:
                with open(manifest, 'r', encoding='utf-8') as f:
                    m = json.load(f)
                total_mo = m.get("total_mo", 0)
                sources.append({
                    "type": "manifest",
                    "chemin": manifest,
                    "langue": langue,
                    "taille_mo": total_mo
                })
            except Exception:
                pass

    # 2. Dossiers dynamiques avec des .txt
    txt_files = glob.glob(os.path.join("data", "**", "*.txt"), recursive=True)
    dossiers = defaultdict(list)
    for txt in txt_files:
        dossiers[os.path.dirname(txt)].append(txt)
        
    for dossier, fichiers in dossiers.items():
        # Optionnel: on peut masquer les dossiers "sources" des manifests pour éviter les doublons
        # Mais pour plus de flexibilité, on les affiche.
        taille_octets = sum(os.path.getsize(f) for f in fichiers)
        taille_mo = taille_octets / 1_000_000
        if taille_mo > 0.001:
            sources.append({
                "type": "dossier",
                "chemin": dossier,
                "fichiers": fichiers,
                "taille_mo": taille_mo
            })
            
    return sources

def choisir_source_donnees(auto_select=False):
    """Propose à l'utilisateur de choisir parmi les sources trouvées."""
    sources = trouver_sources_donnees()
    if not sources:
        return None
        
    if len(sources) == 1 or auto_select:
        return sources[0]
        
    print("\n" + "="*62)
    print("  📚  SOURCES DE DONNÉES DÉTECTÉES")
    print("="*62)
    for i, src in enumerate(sources, 1):
        if src["type"] == "manifest":
            label = "🇫🇷/🇬🇧 Pack" if "multi" in src["chemin"] else "Pack"
            print(f"  [{i}] {label:<15} : {src['chemin']} ({src['taille_mo']:.1f} Mo)")
        else:
            print(f"  [{i}] Dossier local   : {src['chemin']}/ ({len(src['fichiers'])} fichiers .txt) ({src['taille_mo']:.1f} Mo)")
    print("="*62)
    
    choix = input("  Choix [1] > ").strip()
    try:
        idx = int(choix) - 1
        if 0 <= idx < len(sources):
            return sources[idx]
    except Exception:
        pass
    return sources[0]

def lire_source_donnees(source, max_chars=None):
    """Lit le texte brut depuis la source sélectionnée."""
    texte = ""
    if source["type"] == "manifest":
        with open(source["chemin"], 'r', encoding='utf-8') as f:
            m = json.load(f)
        sources_dir = os.path.join("data", m["langue"], "sources")
        for src in m.get("sources", []):
            fichier = os.path.join(sources_dir, src["fichier"])
            if os.path.exists(fichier):
                with open(fichier, 'r', encoding='utf-8', errors='ignore') as f:
                    if max_chars:
                        texte += f.read(max_chars - len(texte))
                        if len(texte) >= max_chars: break
                    else:
                        texte += f.read()
    elif source["type"] == "dossier":
        fichiers = source["fichiers"]
        # Pour le tokenizer, si on a une limite max_chars, on répartit la lecture sur tous les fichiers du dossier
        if max_chars:
            chars_par_fichier = max_chars // len(fichiers) + 1
            for f_path in fichiers:
                with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                    texte += f.read(chars_par_fichier)
                if len(texte) >= max_chars:
                    texte = texte[:max_chars]
                    break
        else:
            for f_path in fichiers:
                with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                    texte += f.read()
    return texte

def signature_dataset(source, vocab_size=None):
    """Calcule une empreinte pour savoir si le dataset a changé."""
    if vocab_size is None:
        vocab_size = VOCAB_SIZE
    h = hashlib.sha256()
    h.update(str(vocab_size).encode("utf-8"))
    
    if source["type"] == "manifest":
        h.update(source["chemin"].encode("utf-8"))
        if os.path.exists(source["chemin"]):
            h.update(str(os.path.getsize(source["chemin"])).encode("utf-8"))
    else:
        for f in sorted(source["fichiers"]):
            h.update(f.encode("utf-8"))
            if os.path.exists(f):
                h.update(str(os.path.getsize(f)).encode("utf-8"))
    return h.hexdigest()

def tokenizer_a_jour(chemin_tok=TOKENIZER_FILE, source=None):
    """Indique si tokenizer.json existe ET correspond au dataset actuel."""
    if not os.path.exists(chemin_tok):
        return False, "tokenizer.json introuvable"
    if source is None:
        return True, "aucune donnée pour vérifier — tokenizer conservé"
    try:
        with open(chemin_tok, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False, "tokenizer.json illisible — réentraînement"
        
    if meta.get("dataset_sig") != signature_dataset(source):
        return False, f"dataset ou vocab modifié"
    return True, "tokenizer à jour"

# ================================================================
# SCRIPT PRINCIPAL — entraîne et sauvegarde
# ================================================================
if __name__ == "__main__":

    # Le tokenizer sélectionne automatiquement la première source disponible
    # pour créer un vocabulaire général.
    source = choisir_source_donnees(auto_select=True)

    if not source:
        print("❌ Aucun fichier de données trouvé !")
        print("   Lance d'abord telecharger.py ou ajoute des .txt dans data/")
        raise SystemExit(1)

    print(f"  Source de tokenisation : {source['chemin']}")

    a_jour, raison = tokenizer_a_jour(TOKENIZER_FILE, source)
    if a_jour:
        print(f"  ✅ Tokenizer déjà à jour ({raison}) — pas de réentraînement.")
        raise SystemExit(0)
    print(f"  ↻ Réentraînement du tokenizer : {raison}")

    print(f"Lecture des données (max {_SAMPLE_MAX/1e6:.0f} Mo)...")
    echantillon = lire_source_donnees(source, max_chars=_SAMPLE_MAX)
    print(f"  {len(echantillon)/1e6:.1f} Mo lus")

    sig = signature_dataset(source)
    tok = TokenizerBPE()
    tok.entrainer(echantillon, vocab_size=VOCAB_SIZE)
    tok.sauvegarder(TOKENIZER_FILE, dataset_sig=sig, dataset_file=source["chemin"])

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
        ok = "OK" if decode.strip() == phrase.strip() else "ERREUR"
        print(f"  {ok} Coherence encodage/decodage")

    print("\n" + "="*55)
    print("  Tokenizer pret !")
    print("  Lance maintenant : ./wish go")
    print("="*55)
