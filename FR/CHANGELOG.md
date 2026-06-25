# Changelog

Toutes les corrections et améliorations notables de WishAI.

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/). Heures en heure américaine (ET).

---

## [1.5.2] — 2026-06-25 12h10.pm

Mise à jour majeure améliorant la fiabilité de l'entraînement, les logs, et la structure interne.

### ✨ Ajouté

- **Système de reprise d'entraînement (Recovery)** : L'IA sauvegarde maintenant son état avec robustesse, permettant de reprendre un entraînement exactement là où il s'est arrêté (utile en cas de crash, d'interruption volontaire ou d'erreur humaine).
- **Logs avancés et système temporaire** : Création d'un sous-dossier `TEMP/` dédié avec horodatage précis (date exacte, heures, secondes) pour un historique granulaire des entraînements sans polluer le dossier principal.
- **Enrichissement de `log_active.json`** : Ajout d'informations système cruciales en temps réel :
  - Le statut d'activation d'AMP (Automatic Mixed Precision) et de `torch.compile()`.
  - Le nombre réel de paramètres du modèle.
  - Le statut d'encodage BPE et Tokenisation, incluant le pourcentage de téléchargement des données et les dossiers/fichiers spécifiques utilisés.

### ♻️ Modifié

- **Détection dynamique** : Amélioration de la détection du modèle en cours pour le dashboard (ne dépend plus aveuglément de `active.json`).
- **Expérience Utilisateur (UX) du nuage de mots** : Le nuage de 4 000 tokens a été aplati de force en 2D strict (axe Z fixé à 0) et la rotation 3D a été désactivée, rendant l'interface beaucoup plus lisible.
- **Architecture de configuration modulaire** : `scripts/config.py` a été refactorisé. Au lieu d'avoir tout dans un seul fichier géant, le code est désormais proprement séparé dans un dossier `scripts/Config/` (`donnees.py`, `modeles.py`, `systeme.py`, `utilitaires.py`).
- **Optimisation de l'exportation ONNX/GGUF** : L'export des modèles se fait désormais de manière plus propre, directement dans le répertoire de chaque modèle (`model/<nom_du_modele>/`) avec l'ajout automatique du fichier `tokenizer.json`, au lieu de tout copier de force sur le Bureau.

### 🐛 Corrigé

- **Bug de "Shadowing" (`token.py`)** : Le script `scripts/token.py` créait un conflit avec les paquets Python standards, ce qui réinitialisait inopinément le tokenizer lors de l'exportation. Il a été renommé en `scripts/reset_token.py`.
- **Crash au chargement pour l'exportation** : `export.py` a été modifié pour gérer le chargement correct des modèles à partir des checkpoints d'entraînement (dictionnaires de poids + `log_active.json` pour la configuration) au lieu de supposer que l'objet PyTorch complet était sauvegardé, évitant ainsi le crash `AttributeError: 'dict' object has no attribute 'eval'`.

---

## [1.5.1] — 2026-06-25 9h20.am

### ✨ Ajouté

- **Moteur analytique Julia — Prédictions avancées de convergence**
  - **`src/estimations.jl`** — nouveau processus Julia tournant en arrière-plan aux côtés du script Python d'entraînement. Effectue deux analyses que Python ne peut pas faire efficacement :
    - **Risque d'overfitting Chinchilla** — calcule le ratio paramètres/tokens et classe le risque.
    - **Prédiction de plateau par courbe exponentielle** — ajuste une courbe pour estimer la perte asymptotique.
  - **IPC par fichiers** — Julia lit `model/{nom}/log_active.json` et produit `model/{nom}/insights.json`.
  - **Installation automatique de Julia** — `src/require.py` appelle `check_and_install_julia()` au démarrage.
  - **Dégradation gracieuse** — si Julia est absent, passe le processus sans planter.
- **Sauvegarde du meilleur modèle** : `nanogpt_bpe.py` sauvegarde désormais `best_model.pt` chaque fois que `val_loss` atteint un nouveau record absolu.
- **Tests** : **`tests/test_all.py`** — suite de tests automatisés couvrant toutes les fonctionnalités de la v1.5.1.

### ♻️ Modifié

- **Learning Rate adaptatif (Cosine Decay)**
  - **`get_lr(iteration)`** dans `nanogpt_bpe.py` — remplace le taux d'apprentissage fixe par un planning en deux phases (Warmup puis Cosine decay).
  - Le LR courant est enregistré dans `log_active.json`.
- **Dashboard — Intégration des métriques Julia**
  - **Carte Learning Rate** — nouvelle carte dans la rangée des métriques principales.
  - **Prédiction de plateau (Julia)** — la cellule "Plateau estimé" affiche maintenant le résultat du fit exponentiel de Julia.
  - **Cellule Overfitting Chinchilla** — nouvelle cellule affichant le niveau de risque.

---

## [1.5.0] — 2026-06-24 18h13.pm

### ✨ Ajouté

- **Bot de téléchargement — vraies tailles et sources personnalisées**
  - **Vraies tailles de datasets HuggingFace** — `obtenir_taille_hf()` interroge l'API HuggingFace.
  - **Affichage des tailles dans le menu** — chaque source affiche maintenant la taille max et typique.
  - **Sources personnalisées** (`src/custom_sources.json`) — ajout de ses propres datasets sans toucher au code.
  - **Filtre de langue sur les datasets HuggingFace** — champ `langue_cible` ajouté.
  - **Sélection de langue obligatoire** — refuse désormais une entrée vide.
  - **Intégration de `langdetect`** — bibliothèque optionnelle importée au démarrage.
- **Rechargement à chaud et checkpoints pendant l'entraînement**
  - **Hot-reload** — après téléchargement d'une nouvelle source, l'entraînement la prend en compte à chaud via `reload_requested.flag`.
  - **Sauvegarde des checkpoints pendant l'entraînement** — les checkpoints sont sauvegardés toutes les `checkpoint_interval` étapes.

### ♻️ Modifié

- **Système de manifeste — 50% d'espace disque économisé**
  - **`data/{langue}/manifest.json`** remplace la copie dans `data.txt`, lu en streaming.
  - **`telecharger.py`** — écrit le manifeste sans copier aucun fichier.
  - **`tokenizer.py`** et **`nanogpt_bpe.py`** modifiés pour lire via le manifeste.

---

## [1.4.0] — 2026-06-24 9h22.am

### ✨ Ajouté

- **Bot automatique & téléchargement**
  - **Bot automatique** (`[a]` dans la bibliothèque) — système de téléchargement intelligent (Preset ou Personnalisé).
  - **Retry HuggingFace** — jusqu'à 4 tentatives en cas de connexion coupée.
- **Logs & auto-réparation**
  - **`src/bot_logger.py`** — système de logs structuré pour le bot.
  - **Auto-réparation** — désinstalle et réinstalle automatiquement via pip les dépendances corrompues.
  - **`tests/test_sources.py`** — vérifie la rapidité et la validité des sources HF.
- **Nouvelles commandes**
  - `./wish token`, `./wish repair`, `./wish logs`, `./wish chat --terminal`

### ♻️ Modifié

- **Tokenizer** : **Taille d'échantillon dynamique** — utilise 15 % du fichier (min 5 Mo, max 50 Mo) au lieu de 5 Mo fixes.

### 🐛 Corrigé

- **Menu bibliothèque** : une entrée vide ou invalide ne quitte plus le programme.

---

## [1.3.3] — 2026-06-24 8h26.am

### ✨ Ajouté

- **Système de raccourcis (`wish.bat`)** — point d'entrée unique (`go`, `chat`, `quick`, `config`, `serve`, `visual`).
- **Désinstallation des dépendances** : nouvelle option `[16]` dans `config.py`.

---

## [1.3.2] — 2026-06-24 07:54.am

### ♻️ Modifié

- **Structure du projet**
  - Racine nettoyée (8 fichiers restants).
  - Scripts de lancement dans `scripts/`.
  - Documentation dans `docs/`.
  - Interface web dans `web/`.
  - Fichiers système (`session.json`, etc.) dans `system/`.

---

## [1.3.1] — 2026-06-23 20:07.pm

### ♻️ Modifié

- **Améliorations du visualiseur d'embeddings**
  - Couleurs plates (suppression des effets néon/glow).
  - Labels affichés uniquement au survol.
  - Système de filtres intelligent (Mots, Chiffres, Spéciaux, etc.).
  - Tokens fin-de-mot (`</w>`) réduits visuellement.

---

## [1.3.0] — 2026-06-23 19:34.pm

### ✨ Ajouté

- **Historique IndexedDB** dans l'interface de chat.

### ♻️ Modifié

- **Interface de chat — refonte complète**
  - Suppression de la topbar, input pill arrondi.
  - Sélecteur de modèle intégré, panneau d'options fluide.
  - Sidebar flottante pour l'historique avec effet glassmorphism.
  - Glow violet en fond d'écran.
  - Bulles de chat améliorées (animations, bouton copier, vitesse).
  - Bouton **↺ Régénérer** pour relancer la génération.
  - Chargement automatique au changement de modèle (cache côté client).

### 🐛 Corrigé

- **Mismatch d'architecture dans `chat_server.py`** : correction de l'import pour supporter l'architecture RoPE/SwiGLU.
- **Espaces manquants** : ajout de `decode_token(id)` pour éviter les `strip()` intempestifs sur `</w>`.
- **`KeyError: 'hyperparams'`** : rétrocompatibilité des anciens checkpoints.

---

## [1.2.0] — 2026-06-21 18:28.pm

### ✨ Ajouté

- **Dashboard — UI complète**
  - Écran idle animé avec redirection auto.
  - Bannière de fin d'entraînement.
  - 4 sections dépliables (Tendances, Performance, Analyse, Journal).
  - Affichage du dernier texte généré et tableau hyperparamètres.
  - Suivi de session en temps réel.
- **`quick.py`** — mode rapide zéro config.
- **Interface de chat** web et script serveur `serve.py`.
- Boucle de reprise automatique phase 3 dans `go.py`.

### 🐛 Corrigé

- Code dupliqué dans `library.html`.
- Import manquant dans `telecharger.py`.

### 📄 Documentation

- Menu `config.py` documenté dans le README.

---

## [1.1.0] — 2026-06-21 18:20.pm

### ♻️ Modifié

- **Architecture modernisée** : passage de GPT-2 (2019) à LLaMA/Mistral (2024).
  - RMSNorm, RoPE, SwiGLU, sans biais (`bias=False`).

### ⚡ Amélioré

- Port monitor dynamique (`monitor_port.json`).
- Avertissement virtualenv.
- **Dashboard SSE** : passage en Server-Sent Events au lieu du polling setInterval.
- Comparaison multi-modèles dans le dashboard.
- Filtrage qualité robuste pour Common Crawl.

### 🐛 Corrigé

- Protection contre le plantage si `data.txt` est vide.

### 📄 Documentation

- **`CONTRIBUTING.md`** créé.

---

## [1.0.0] — 2026-06-21 18:20.pm

Première version stable. Passe en revue du code, correction des bugs et ajout d'un filet de sécurité (tests).

### ✨ Ajouté

- **Support Apple Silicon (MPS)**.
- **Précision mixte (AMP)** (autocast bf16/fp16).
- **`torch.compile`** sur Linux+CUDA.
- **Smoke tests** sans dépendances externes.

### ♻️ Modifié

- Modèle extrait dans `src/model.py`.

### 🐛 Corrigé

- Ligne parasite `shu` dans `nanogpt_bpe.py`.
- Erreur `NameError` sur `supprimer_donnees()`.
- Tokenizer non réentraîné silencieusement (ajout de la signature du dataset).
- Écriture de `control.json` rendue atomique.
- Clignotement du dashboard au démarrage.

### 🧹 Nettoyé

- Imports inutilisés et variables mortes retirés.
- Nettoyage des f-strings inutiles.

### 🗑️ Retiré

- `btn_dashboard.py` (bouton flottant).

### 📄 Documentation

- Mise à jour des README.
