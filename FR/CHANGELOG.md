# Changelog

Toutes les corrections et améliorations notables de WishAI.

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/). Heures en heure américaine (ET).

---

## [1.5.0] — 2026-06-24 18h13

### Bot de téléchargement — vraies tailles et sources personnalisées

- **Vraies tailles de datasets HuggingFace** — `obtenir_taille_hf()` interroge l'API HuggingFace (`load_dataset_builder`) pour obtenir les tailles réelles au lieu de valeurs manuelles. Résultats mis en cache dans `src/sizes_cache.json` (30 jours). Fetch paresseux (déclenché à l'affichage d'une catégorie, pas au démarrage). `taille_max_mo` dans `SOURCES` sert de fallback si l'API est indisponible.
- **Affichage des tailles dans le menu** — chaque source affiche maintenant `Max: ~4 Go  Typique: 800 Mo` avec un badge de taille. Si la quantité demandée dépasse le max, un avertissement est affiché et le téléchargement est plafonné automatiquement.
- **Sources personnalisées** (`src/custom_sources.json`) — ajout de ses propres datasets HuggingFace sans toucher au code Python. Un nouveau sous-menu `[ +]` guide à travers les champs requis (chemin HF, champ texte, langue, taille max). Les sources custom apparaissent dans le menu avec le badge `[CUSTOM]` et sont fusionnées dans `SOURCES` au démarrage.
- **Filtre de langue sur les datasets HuggingFace** — champ `langue_cible` ajouté aux sources qui mélangent plusieurs langues (ex : `oasst2` qui contient FR/EN/DE/ES/ZH). Chaque texte extrait passe par `detecter_langue()` et est ignoré s'il ne correspond pas à la langue cible, garantissant des datasets propres.
- **Sélection de langue obligatoire** — `_demander_liste(defaut=None)` refuse désormais une entrée vide, forçant un choix explicite de langue en mode preset et en mode personnalisé. Plus de valeur par défaut silencieuse.
- **Intégration de `langdetect`** — bibliothèque optionnelle (~2 Mo) importée au démarrage avec `try/except`. Si disponible, elle offre ~95% de précision sur 55 langues ; sinon, retour automatique sur la détection par mots-clés existante.

### Système de manifeste — 50% d'espace disque économisé

- **`data/{langue}/manifest.json`** remplace la copie dans `data.txt`. Au lieu de concaténer toutes les sources en un gros `data.txt` (ce qui doublait l'espace disque), un manifeste JSON léger liste les fichiers sources actifs. Le tokenizer et le moteur d'entraînement les lisent directement en streaming. Sur un dataset de 950 Mo, l'espace utilisé passe de ~1,9 Go à ~950 Mo.
- **`telecharger.py`** — `combiner_sources()` remplacé par `maj_manifest()` qui écrit le manifeste sans copier aucun fichier. `reload_requested.flag` est créé après chaque téléchargement réussi.
- **`tokenizer.py`** — lit via `manifest.json` en priorité (streaming, jusqu'à 50 Mo d'échantillon), bascule sur `data.txt` si aucun manifeste n'est trouvé.
- **`nanogpt_bpe.py`** — lit les données d'entraînement via le manifeste. Vérification de validité du cache étendue pour comparer les dates de modification des sources avec le cache BPE.

### Rechargement à chaud et checkpoints pendant l'entraînement

- **Hot-reload** — après le téléchargement d'une nouvelle source, `telecharger.py` crée `data/{langue}/reload_requested.flag`. À chaque checkpoint (toutes les `checkpoint_interval` étapes), `nanogpt_bpe.py` détecte ce flag, le supprime, ré-tokenise toutes les sources (anciennes + nouvelles), et reprend l'entraînement exactement à la même étape. Aucun redémarrage nécessaire.
- **Sauvegarde des checkpoints pendant l'entraînement** — était entièrement absent de la boucle d'entraînement malgré la configuration de `checkpoint_interval`. Les checkpoints sont maintenant sauvegardés toutes les `checkpoint_interval` étapes (pas seulement en toute fin d'entraînement), permettant la reprise après crash, le hot-reload et la boucle de reprise automatique de `go.py`.

---

## [1.4.0] — 2026-06-24 9h22

### Bot automatique & téléchargement

- **Bot automatique** (`[a]` dans la bibliothèque, ou par défaut au premier lancement) — système de téléchargement intelligent qui détecte l'espace disque (plafond : 30 % de la capacité totale, max 2 Go) et propose deux modes :
  - **Mode preset** — Nano (~100 Mo) / Small (~300 Mo) / Medium (~700 Mo) / Large (~1,5 Go)
  - **Mode personnalisé** — choix du nombre de paramètres (20M → 1B+), Mo recommandés calculés automatiquement, plafond ajustable, langue (FR / EN / Multi), type d'IA (Général / Code / Sciences / Chat / Assistant). Sources sélectionnées intelligemment selon la combinaison.
- **Retry HuggingFace** — jusqu'à 4 tentatives avec backoff exponentiel (3s, 8s, 20s) en cas de connexion coupée (`WinError 10054`). Reconnexion en cours d'itération : recharge le dataset et saute les articles déjà écrits.
- **Correction menu bibliothèque** — une entrée vide ou invalide ne quitte plus le programme ; boucle jusqu'à un choix valide.

### Logs & auto-réparation

- **`src/bot_logger.py`** — système de logs pour le bot :
  - `system/logs/bot.log` — tous les événements (INFO, WARNING, ERROR, FATAL)
  - `system/logs/erreurs.log` — erreurs uniquement, pour diagnostic rapide
  - `system/logs/downloads.log` — JSON Lines structuré par téléchargement (source, Mo, succès/échec, horodatage)
  - Rotation automatique à 2 Mo, conserve 3 fichiers
- **Auto-réparation** — si une dépendance (`datasets`, `torch`, `safetensors`…) est absente ou corrompue (ImportError / OSError), elle est automatiquement désinstallée et réinstallée via pip avant une nouvelle tentative. Intégré directement dans `telecharger_hf`.
- **`tests/test_sources.py`** — envoie une requête de ~1 Mo à chaque source HuggingFace et affiche ✅ / ❌ + top 5 des sources les plus rapides.

### Nouvelles commandes

- **`./wish token`** — supprime `system/tokenizer.json` (et les tokenizers des modèles) puis réentraîne immédiatement le tokenizer BPE sur les données existantes. `./wish token reset` supprime uniquement (réentraînement différé au prochain `./wish go`).
- **`./wish repair`** — vérifie tous les paquets critiques et réinstalle automatiquement ceux qui sont absents ou corrompus.
- **`./wish logs`** — affiche les 40 derniers événements du bot. `./wish logs erreurs` pour les erreurs uniquement. `./wish logs repair` pour réparer et logger en une commande.
- **`./wish chat --terminal`** — mode génération terminal (fusionné depuis `src/generate.py` supprimé) : liste les modèles disponibles, choix, boucle interactive (`t=`, `n=`, `q`).

### Tokenizer

- **Taille d'échantillon dynamique** — l'entraînement BPE du tokenizer utilise désormais 15 % du fichier de données (min 5 Mo, max 50 Mo) au lieu d'un fixe 5 Mo. Sur un dataset de 252 Mo : ~38 Mo utilisés, vocabulaire bien plus représentatif.

---

## [1.3.3] — 2026-06-24 8h26

### Système de raccourcis

- **`wish.bat`** — système de raccourcis unifié à la racine. Point d'entrée unique pour toutes les commandes :
  ```
  ./wish go        Menu principal
  ./wish chat      Interface de chat
  ./wish quick     Entraînement rapide (zéro config, ~20M params)
  ./wish config    Gestion des modèles et données
  ./wish serve     Serveur dashboard / bibliothèque
  ./wish visual    Visualiseur d'embeddings (port 8080)
  ```
- **`scripts/config.py`** — nouvelle option **[16] Désinstaller les dépendances** : lance `pip uninstall -y` sur tous les paquets de `requirements.txt` sans toucher à `deps.lock`.

---

## [1.3.2] — 2026-06-24 07:54

### Structure du projet

- **Racine nettoyée** : seulement 8 fichiers à la racine (`go.py`, `dashboard.html`, `README.md`, `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `DATASETS.md`, `requirements.txt`).
- **`scripts/`** — scripts de lancement déplacés ici : `chat.py`, `quick.py`, `serve.py`, `config.py`.
- **`docs/`** — documentation déplacée ici : `PARAMETRES.md`, `LAUNCH.md` (guide de lancement communauté).
- **`web/`** — `library.html` déplacé ici.
- **`system/`** — `pyrightconfig.json` et `tokenizer.json` déplacés ici. Tous les chemins mis à jour dans `src/`.
- **`wish.bat`** — lanceur unique à la racine. Remplace 5 fichiers `.bat` individuels.
  ```
  ./wish go | chat | quick | config | serve | visual
  ```

---

## [1.3.1] — 2026-06-23 20:07

### Améliorations du visualiseur d'embeddings

- **Couleurs plates** : suppression de tous les effets néon/glow (`ctx.shadowBlur`, `ctx.shadowColor`). Les couleurs sont désormais solides et sobres — mots `#4db8cc`, chiffres `#c8a830`, caractères spéciaux `#c04060`.
- **Labels uniquement au survol** : les noms de tokens ne s'affichent plus automatiquement en zoomant. Ils apparaissent exclusivement au hover, avec un anneau blanc + point plein et le nom au-dessus.
- **Système de filtres** : cinq boutons dans la topbar — *Tous*, *Mots*, *Chiffres*, *Spéciaux*, *Fins de mot*. Les tokens hors filtre sont atténués à 8% d'opacité. L'état du filtre se combine avec la barre de recherche.
- **Tokens fin-de-mot** (`</w>`) dessinés à 75% du rayon de base pour les distinguer visuellement.
- **Style** : `visual/style.css` mis à jour avec les styles `.filter-btn` et `.filter-btn.active`.

---

## [1.3.0] — 2026-06-23 19:34

### Interface de chat — refonte complète

- **Suppression de la topbar** : la barre de navigation du haut a été retirée entièrement. L'interface est maintenant fullscreen, sans en-tête fixe.
- **Barre d'input pill** : la zone de saisie est désormais une capsule très arrondie (`border-radius: 32px`), centrée verticalement à l'écran au démarrage, puis ancrée en bas lors du premier envoi.
- **Sélecteur de modèle intégré** : le dropdown de modèles est maintenant intégré directement sous le textarea (plus de panneau séparé). Les modèles sont triés par taille décroissante. Le bouton **Charger** se transforme en **✓ En cache** (vert, désactivé) quand le modèle actif est déjà en mémoire.
- **"Plus d'options" avec fade-in** : Température et Longueur sont cachées derrière un bouton ⚙ discret. Le panneau s'ouvre et se ferme avec une transition fluide (`max-height` + `opacity` + `padding`) — pas de saut brutal.
- **Sidebar flottante** : l'historique des conversations s'ouvre via un bouton ☰ flottant en haut à gauche. Ce bouton disparaît quand la sidebar est ouverte. La sidebar est une carte flottante arrondie avec effet glassmorphism (`backdrop-filter: blur`), entièrement désolidarisée du flux du document.
- **Glow violet aux bords** : le fond de l'interface affiche 4 blobs elliptiques violets aux coins de l'écran via des dégradés radiaux Canvas 2D — le point cloud 3D a été retiré.
- **Bulles de chat améliorées** :
  - Arrondi renforcé, animation d'entrée (`fadeIn` + `translateY(8px) → 0`)
  - Chaque réponse IA affiche, au survol, une rangée d'actions : bouton **Copier**, temps de réponse en millisecondes, nombre de tokens générés, et bouton **↺ Régénérer**
  - Le bouton Copier affiche temporairement « ✓ Copié ! » en vert après le clic
- **Régénération** : le bouton ↺ supprime la dernière réponse IA et relance la génération à partir du dernier message utilisateur, sans recharger la page.
- **Chargement automatique au changement de modèle** : sélectionner un modèle dans le dropdown déclenche son chargement immédiatement. Si le modèle est déjà en mémoire (`_cachedModelName`), aucun rechargement n'est effectué — cache côté client.
- **Historique IndexedDB** : les conversations sont persistées dans IndexedDB (`WishAIChat` / `convs`), récupérées au démarrage, et accessibles depuis la sidebar.

### Corrections

- **Mismatch d'architecture dans `chat_server.py`** : l'ancienne fonction `build_modele()` utilisait une architecture GPT-2 (position embedding appris, `tril`, `ReLU`, `LayerNorm`) incompatible avec les modèles entraînés via `model.py` (RoPE + RMSNorm + SwiGLU). Corrigé en remplaçant `build_modele()` par un import direct de `WishAI_BPE` et `ConfigModele` depuis `src/model.py`.
- **Espaces manquants dans le texte généré** : le streaming token par token utilisait `tok.decoder()` qui appelle `.strip()` et supprimait les espaces en fin de mot (marqueur `</w>` BPE). Corrigé avec une fonction `decode_token(id)` dédiée qui convertit `</w>` → espace sans `.strip()`.
- **`KeyError: 'hyperparams'`** : compatibilité entre l'ancien format de checkpoint (`hyperparams` / `taille_vocab`) et le nouveau format (`architecture` avec `vocab_size` inclus). `_charger_modele()` détecte la clé présente et s'adapte.

---

## [1.2.0] — 2026-06-21 18:28

### Dashboard — UI complète

- **Écran idle animé** : quand aucun entraînement ne tourne, le dashboard affiche
  un écran d'attente avec particules flottantes, cerveau animé, grille en mouvement
  et un indicateur de pulse. Auto-redirect `file://` → `localhost` (port mémorisé
  en `localStorage`).
- **Bannière "Entraînement terminé"** : barre verte qui glisse du haut avec les
  statistiques clés (val loss final, durée, étapes) dès que le statut passe
  à `terminé`.
- **4 sections dépliables** (état mémorisé entre les rechargements) :
  - **📈 Tendances & Convergence** — Δ val loss sur 10 évals, tendance
    (descend / plateau / overfitting), vitesse de descente (delta/100 étapes),
    plateau estimé automatiquement
  - **⚡ Performance réelle** — tokens/s, Mo de texte traités, batch effectif
    (batch × grad_accum), nombre de checkpoints créés
  - **🧠 Analyse du modèle** — params/couche, VRAM théorique (~4 o/param),
    head size (n_embd ÷ n_head), état d'apprentissage avec conseil adapté
  - **📋 Journal des événements** — records val_loss, pauses thermiques/RAM,
    convergence — chaque événement horodaté par étape
- **Dernier texte généré** : affiché en bas de page avec bouton
  "Voir tout / Réduire" (expand/collapse).
- **Suivi de session** : `session.json` écrit par `go.py` au démarrage ; le
  dashboard le surveille via SSE et remet l'affichage à zéro automatiquement
  quand une nouvelle session démarre.
- **Tableau des hyperparamètres + diagramme d'architecture** : section fixe
  sous les graphes qui affiche tous les paramètres du modèle en cours et un
  schéma visuel de l'architecture.

### Nouveau

- **`quick.py`** — mode rapide zéro config : génère un texte de démo si aucune
  donnée n'existe (~200 phrases FR/EN), entraîne le tokenizer BPE, lance le
  dashboard et démarre l'entraînement avec le preset MINI (~20M paramètres).
  Aucune question posée — `Ctrl+C` pour arrêter.
- **Interface de chat** (`chatting/`, `chat.py`, `src/chat_server.py`) — parle
  avec ton modèle entraîné depuis une interface web. Historique des conversations
  persistant, sélecteur de modèle, chargement à chaud sans redémarrer le serveur.
- **`serve.py`** — lance uniquement le serveur de chat sans passer par `go.py`.
- **`go.py` — ouverture optionnelle de la bibliothèque** : si des données existent
  déjà, propose d'ouvrir la bibliothèque pour en rajouter avant de commencer.
- **`go.py` — boucle de reprise phase 3** : si `monitor.py` déclenche un arrêt
  critique, `go.py` attend la reprise automatique et relance l'entraînement depuis
  le checkpoint sans intervention manuelle.

### Corrections

- `library.html` — blocs de code dupliqués supprimés dans `loadLocalData()` et
  `checkServer()` (erreurs JS lignes 334 et 559)
- `src/telecharger.py` — `import sys` manquant alors que `sys.exit(1)` est utilisé
- `pyrightconfig.json` créé — Pylance pointe sur `.venv312` ; plus d'avertissements
  "import could not be resolved" dans VSCode
- `.gitignore` — `tests/` retiré (bug : `tests/test_smoke.py` n'était jamais tracké)
  ; `.ruff_cache/`, `monitor_port.json`, `deps.lock` ajoutés

### Documentation

- `config.py` — menu complet des 16 commandes documenté dans `README.md` et
  `README_FR.md` en volet déroulant (liste, hyperparamètres, export, reset, logs…)

---

## [1.1.0] — 2026-06-21 18:20

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

## [1.0.0] — 2026-06-21 18:20

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
