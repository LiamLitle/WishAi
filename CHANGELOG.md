# Changelog

Toutes les corrections et améliorations notables de WishAI.

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

---

## [1.1.0] — 2026-06-21

### Architecture

- **Modele modernise (RoPE + RMSNorm + SwiGLU)** : `model.py` passe d'une
  architecture GPT-2 (2019) a une architecture style LLaMA/Mistral (2024).
  - `LayerNorm` remplace par `RMSNorm` (plus simple, plus rapide, pas de biais)
  - Positional embeddings appris supprimes, remplace par **RoPE** (Rotary Position
    Embedding) — meilleure generalisation sur les longues sequences
  - FFN `Linear → ReLU → Linear` remplace par **SwiGLU** (`SiLU(gate) * up → down`)
    — meilleure loss a budget de parametres egal
  - Toutes les couches lineaires passent sans biais (`bias=False`)
  - `nn.Sequential` des blocs remplace par `nn.ModuleList` + boucle (necessaire
    pour passer `cos`/`sin` RoPE a chaque bloc)
  - Comptage de parametres MINI inchange : **20.8M** (la suppression du
    `position_embedding` compense les 3 projections SwiGLU vs 2)

### Bugs corriges

- **`data/data.txt` vide bloquait le tokenizer** : `donnees_existent()` (dans
  `quick.py` et `go.py`) et `trouver_data_file()` (dans `tokenizer.py`) renvoient
  desormais `False`/`None` si le fichier fait moins de 1 Ko — evite un
  entraînement BPE sur texte vide (vocab = 1 token, tout encode en `</w>`)

### Ameliorations

- **Port monitor dynamique** : `monitor.py` cherche un port TCP libre au lieu
  de hardcoder 8001. Le port est ecrit dans `monitor_port.json` pour que
  `dashboard.py` puisse le lire. Plus de clash silencieux si le port est pris.
- **Avertissement virtualenv dans `require.py`** : si Python tourne hors venv
  (et hors conda), l'utilisateur voit un message clair et doit confirmer avant
  toute installation globale.
- **Dashboard SSE** : le polling `setInterval` (toutes les 2-3s) est remplace
  par **Server-Sent Events** — `dashboard.py` expose `/api/events` qui pousse
  les logs d'entrainement, les changements de session, et les donnees monitor
  en temps reel. Fallback polling automatique en cas d'echec SSE ou ouverture
  en `file://`. `dashboard.py` passe en `ThreadingHTTPServer` pour gerer
  plusieurs connexions simultanees.
- **Comparaison multi-modeles dans le dashboard** : bouton "Comparer les modeles"
  qui ouvre un panneau avec toutes les courbes de loss superposees (endpoint
  `/api/models` retourne les historiques de tous les modeles entraines).
- **Filtrage qualite Common Crawl** : `nettoyer_texte()` dans `telecharger.py`
  fait desormais 4 passes : suppression HTML residuel (`<tag>`, `&amp;`...),
  filtrage ligne par ligne (< 40 chars, 2+ URLs, emails, ponctuation > 15%),
  deduplication des lignes repetees (spam/boilerplate), nettoyage non-latin.

### Documentation

- **`CONTRIBUTING.md`** cree : structure du projet, workflow, standards de code,
  zones sensibles, guide PR, idees de contributions.

---

## [1.0.0] — 2026-06-21

Première version stable. Passe en revue du code, correction des bugs et ajout
d'un filet de sécurité (tests).

### 🐞 Bugs corrigés

- **`nanogpt_bpe.py` — ligne parasite `shu`** : une ligne `shu` traînait en fin
  de fichier (au niveau module). Elle provoquait un `NameError` à l'exécution,
  juste après la sauvegarde du modèle. Supprimée.
- **`telecharger.py` — `supprimer_donnees()` non définie** : le menu appelait
  cette fonction (option « s ») mais elle n'existait nulle part → `NameError` au
  clic. Fonction implémentée (suppression d'une source, du `data.txt` combiné, ou
  de tout, par langue, avec espace libéré affiché).
- **Tokenizer non réentraîné au changement de dataset** : il fallait supprimer
  `tokenizer.json` à la main, sinon le modèle s'entraînait sur un vocabulaire qui
  ne correspondait plus aux données (bug silencieux). Désormais une **signature**
  du dataset est stockée dans `tokenizer.json` et comparée à chaque lancement → 
  réentraînement automatique si nécessaire.
  - Sous-bug corrigé : la comparaison de `vocab_size` faisait boucler le
    réentraînement à l'infini sur les petits datasets (le vocab réel pouvant être
    plus petit que la cible). La détection repose maintenant uniquement sur la
    signature (qui encode déjà le vocab cible).
  - Sous-bug corrigé : valeur par défaut figée au chargement du module (piège
    classique de Python) — le vocab cible est désormais lu à l'exécution.
- **`control.json` — écriture non atomique** : en cas de crash en plein write, le
  fichier pouvait être corrompu et faire planter les processus qui le lisent.
  Passé en écriture atomique (fichier `.tmp` + `os.replace`) dans les **3**
  écrivains : `go.py`, `monitor.py`, `nanogpt_bpe.py`.
- **Dashboard — clignotement au démarrage** : au lancement, le dashboard
  s'affichait une fraction de seconde puis repassait sur l'écran « En attente ».
  Cause : deux pollers en conflit (`checkSession` mettait l'état à « starting »,
  `update()` le forçait à « idle » tant qu'aucun log n'existait). Corrigé avec une
  fenêtre de grâce de démarrage **et** une vérification de fraîcheur de session
  (un vieux `session.json` ne déclenche plus de faux « démarrage »).

### 🧹 Nettoyage du code

- **Imports inutilisés retirés** : `json` (`chat.py`), `shutil` et `math as
  _math2` (`nanogpt_bpe.py`), `parse_qs` (`dashboard.py`), `sys`
  (`tokenizer.py`, `telecharger.py`), `platform` / `os` / `urllib.request` /
  `tempfile` (`require.py`), et `torch.nn` / `functional as F` (`nanogpt_bpe.py`,
  après extraction du modèle).
- **Variable morte `_head_size_rec`** supprimée (`nanogpt_bpe.py`).
- **`global _st` redondant** supprimé (`chat_server.py`).
- **28 f-strings sans variable** (`f"..."` sans `{}`) nettoyés dans
  `telecharger.py`, `tokenizer.py`, `nanogpt_bpe.py`, `monitor.py`.

### ✨ Ajouté

- **Support Apple Silicon (MPS)** : détection `torch.backends.mps.is_available()`
  dans `verifier_pc()` (mémoire unifiée estimée pour la reco de preset). WishAI
  ne force plus le CPU sur Mac.
- **Précision mixte (AMP)** dans la boucle d'entraînement : `autocast` bf16/fp16,
  avec **`GradScaler` en fp16** (indispensable pour éviter les NaN — oublié dans
  le plan initial), activée uniquement sur CUDA pour ne pas casser MPS/CPU.
- **`torch.compile`** activé sur Linux+CUDA uniquement (instable sur Windows/MPS),
  avec routage des poids par un module « brut » pour des checkpoints toujours
  compatibles (torch.compile préfixe sinon les clés avec `_orig_mod.`).
- **Smoke tests** (`tests/test_smoke.py`, `unittest`, zéro dépendance) :
  round-trip du tokenizer, détection de changement de dataset, forward pass +
  loss, save/load de checkpoint, génération, et compilation de **tous** les
  fichiers `.py` (ce dernier aurait attrapé le bug `shu`).
  Lancement : `python -m unittest discover -s tests`.

### ♻️ Modifié

- **Modèle extrait dans `src/model.py`** : paramétré par `ConfigModele` (plus
  aucune variable globale, `device` déduit du tenseur d'entrée). Rend le modèle
  importable et testable **sans** déclencher l'entraînement, et allège
  `nanogpt_bpe.py`.

### 🗑️ Retiré

- **`btn_dashboard.py` (bouton flottant Tkinter)** : purement visuel, et pouvait
  crasher sur Linux headless (pas de serveur d'affichage). Le statut reste visible
  dans le terminal et le dashboard. Le lancement dans `go.py` a aussi été retiré.

### 📄 Documentation

- `README.md` et `README_FR.md` mis à jour pour la v1.0 (badge version, support
  MPS, tokenizer automatique, section tests, suppression des références au bouton
  flottant, structure de fichiers à jour).
