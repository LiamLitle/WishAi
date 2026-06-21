<div align="right">

[🇬🇧 English](README.md) | 🇫🇷 Français

</div>

<div align="center">

# 🧠 WishAI

<!-- Stack & Compatibilité -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS%20%7C%20CPU-76b900?logo=nvidia&logoColor=white)](https://pytorch.org/get-started/locally/)
![Cross-Platform](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-0078D4)

<br>

<!-- Fonctionnalités Modernes -->
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-API_Ready-FFD21E)](https://huggingface.co/)
[![Datasets](https://img.shields.io/badge/Datasets-135_curat%C3%A9s_%2B_100k%2B-blue)](DATASETS.md)
[![Safetensors](https://img.shields.io/badge/Sauvegarde-Safetensors-green)](https://huggingface.co/docs/safetensors/index)
[![Dashboard](https://img.shields.io/badge/UI-HTML5_%7C_Vanilla_JS-E34F26?logo=html5&logoColor=white)](/)

<br>

<!-- Licence -->
[![Version](https://img.shields.io/badge/Version-1.1-success)](#)
[![License](https://img.shields.io/badge/Licence-Non--Commercial-red)](LICENSE)

<br>

**nanoGPT c'est bien. Mais t'as pas de dashboard, pas de protection VRAM, et ton PC peut mourir en 4 minutes.**

**WishAI règle ça.**

**🎉 Version 1.1 — architecture moderne + dashboard temps réel.**

*Construit par Liam — from scratch.*

</div>

---

<!-- 📸 AJOUTE TON GIF ICI — enregistre le dashboard avec ScreenToGif puis décommente : -->
<!-- ![WishAI Dashboard](assets/dashboard.gif) -->

---

> WishAI te permet d'entraîner un vrai GPT sur ta machine locale, depuis zéro, sans config compliquée.
> Télécharge des données, lance une commande, et regarde ton IA apprendre en temps réel dans un dashboard natif.

---

<div align="center">
<table>
<tr>
<td align="center" width="16%">

**📊 Dashboard natif**<br>
Fenêtre locale<br>
~60 Mo de RAM

</td>
<td align="center" width="16%">

**🛡️ Protection auto**<br>
VRAM, RAM, temp<br>
jamais de crash

</td>
<td align="center" width="16%">

**📚 Bibliothèque**<br>
135 curatés +<br>
100k+ accessibles

</td>
<td align="center" width="16%">

**🔋 Accumulation**<br>
Gros modèles sur<br>
petites VRAM

</td>
<td align="center" width="16%">

**🔄 Early stopping**<br>
S'arrête tout seul<br>
à convergence

</td>
<td align="center" width="16%">

**🧠 BPE from scratch**<br>
3× plus de contexte<br>
que char-level

</td>
</tr>
</table>
</div>

---

## Nouveautés de la v1.1

- **🧠 Architecture moderne (RoPE + RMSNorm + SwiGLU)** — `model.py` passe d'une architecture GPT-2 (2019) à un style LLaMA/Mistral (2024). Même ~20.8M paramètres, meilleure qualité.
  - `LayerNorm` → `RMSNorm` (plus simple, sans biais, légèrement plus rapide)
  - Positional embeddings appris → **RoPE** (Rotary Position Embedding) : meilleure généralisation sur les longues séquences
  - FFN `Linear → ReLU → Linear` → **SwiGLU** (`SiLU(gate) × up → down`) : meilleure loss à budget de paramètres égal
- **🔌 Port monitor dynamique** — `monitor.py` ne hardcode plus le port 8001. Il trouve un port libre et l'écrit dans `monitor_port.json`. Plus de crash silencieux si le port est déjà pris.
- **⚠️ Avertissement virtualenv dans `require.py`** — si Python tourne hors venv (et hors conda), un message clair s'affiche avec demande de confirmation avant toute installation globale.
- **📡 Dashboard SSE** — le polling `setInterval` est remplacé par **Server-Sent Events**. `dashboard.py` expose `/api/events` qui pousse les logs, les changements de session et les métriques système en temps réel. Fallback polling automatique en cas d'échec SSE.
- **📊 Comparaison multi-modèles** — nouveau bouton "Comparer les modèles" dans le dashboard : toutes les courbes de loss superposées sur un seul graphe (endpoint `/api/models`).
- **🧹 Filtrage qualité Common Crawl** — `nettoyer_texte()` dans `telecharger.py` supprime le HTML résiduel, filtre les lignes courtes/spam/truffées d'URL, déduplique le contenu répété, supprime les caractères non-latins.
- **🐛 Fix : `data/data.txt` vide cassait le tokenizer** — `donnees_existent()` et `trouver_data_file()` vérifient maintenant que le fichier fait plus de 1 Ko, ce qui évite un entraînement BPE silencieux sur du texte vide (produisant un vocab à 1 token : `</w>`).
- **📝 `CONTRIBUTING.md`** — structure du projet, workflow venv, standards de code, fichiers sensibles, guide PR.

---

## Nouveautés de la v1.0

- **🍎 Support Apple Silicon (MPS)** — entraînement sur Mac M1/M2/M3, plus seulement NVIDIA/CPU.
- **⚡ Précision mixte (AMP)** — bf16/fp16 automatique sur CUDA : entraînement plus rapide, moins de VRAM (avec un GradScaler en fp16 pour rester stable). `torch.compile` activé sur Linux+CUDA.
- **🔁 Tokenizer auto-réentraîné** — détecte quand le dataset change (via une signature stockée dans `tokenizer.json`) et se réentraîne tout seul. Fini la suppression manuelle de `tokenizer.json`.
- **🧪 Smoke tests** — une suite `tests/` (`python -m unittest discover -s tests`) qui couvre le tokenizer, le modèle (forward / save-load / génération) et un test de compilation sur chaque fichier.
- **🧩 Modèle importable** — le modèle vit maintenant dans `src/model.py`, paramétré et testable sans lancer l'entraînement.
- **💾 Écritures atomiques de `control.json`** — fichier temporaire + renommage, pour que l'état inter-processus ne se corrompe jamais en plein write.
- **🩹 Fix du démarrage dashboard** — plus de clignotement entre le dashboard et l'écran d'attente pendant le démarrage de l'entraînement.

---

## Installation

> Prérequis : **Python 3.8+**, **pip**, **git**

```bash
git clone https://github.com/ton-pseudo/wishai
cd wishai
python go.py
```

`go.py` installe les dépendances, vérifie le tokenizer, ouvre le dashboard, lance le moniteur en arrière-plan et démarre l'entraînement — tout en une commande.

**GPU (optionnel mais recommandé) :** Si tu as une carte NVIDIA, installe PyTorch avec le support CUDA depuis [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) — choisis ta version CUDA dans le sélecteur. Sur **Apple Silicon (M1/M2/M3)**, WishAI utilise automatiquement le backend **MPS**. Sans GPU, il tourne quand même sur CPU (preset NANO recommandé).

---

## Utilisation

### 1. Télécharge des données

```bash
python src/telecharger.py
```

La **Bibliothèque de Datasets** s'ouvre dans ton navigateur. Interface unifiée avec 4 sources filtrables :

- **📌 Notre Sélection** — **135 datasets** testés et organisés en 19 domaines : Encyclopédies (29 langues), Web, Littérature, Instructions, Code, Maths, Science, Médecine, Dialogues, Traduction, Droit, Finance, Éducation, et plus.
- **🤗 HuggingFace** — accès direct aux **150 000+ datasets** du Hub. Recherche en temps réel avec debounce.
- **🐙 GitHub** — recherche de dépôts datasets sur GitHub (triés par étoiles).
- **📄 Papers with Code** — datasets académiques référencés dans des publications scientifiques (proxy serveur, pas de CORS).

Filtre par source, langue ou domaine en temps réel. Total accessible : **+100k datasets**.

Les téléchargements tournent **en arrière-plan** : tu peux en lancer plusieurs à la fois et suivre leur statut dans l'interface sans bloquer quoi que ce soit.

👉 **[Voir la liste complète des Datasets disponibles](DATASETS.md)**

Tu peux aussi **ajouter tes propres textes** : mets n'importe quel fichier `.txt` dans `data/en/` ou `data/fr/`.

---

### 2. Le tokenizer *(automatique)*

`go.py` entraîne le tokenizer BPE pour toi au premier lancement, et **le réentraîne automatiquement dès que ton dataset change** — il stocke une signature des données dans `tokenizer.json` et la compare à chaque lancement. Plus besoin de supprimer `tokenizer.json` à la main.

Tu peux quand même le lancer manuellement :

```bash
python src/tokenizer.py
```

~5–10 minutes la première fois. Une barre de progression s'affiche pendant l'encodage :

```
[████████████░░░░░░░] 62.5%  (9.3/15.0M mots)
```

Résultat : `tokenizer.json`.

> Avec char-level, 256 tokens ≈ 50 mots. Avec BPE, 256 tokens ≈ **180 mots**. Même modèle, 3× plus de contexte.

---

### 3. Lance tout

```bash
python go.py
```

Le programme détecte ton matériel et propose une config :

| Preset | GPU requis | Params | |
|--------|-----------|--------|-|
| 🐢 NANO | CPU ou < 4 Go | ~2M | pour commencer |
| 🚀 SMALL | 4–6 Go | ~10M | bon équilibre |
| ⚡ MEDIUM | 6–8 Go | ~40M | rapport qualité/temps optimal |
| 🧠 LARGE | 12+ Go | ~85M | pour les patients |
| 🔧 CUSTOM | — | toi qui choisis | avec explications à chaque param |

Puis tu choisis la durée :

```
Minutes [auto] >
```

- **Entrée** → s'arrête tout seul à convergence
- **Un nombre** → calcule les étapes, affiche l'heure de fin

Pendant l'entraînement, suis la progression en direct dans le **dashboard** (il s'ouvre tout seul) — modèle en cours, étape, courbes de loss et métriques système, le tout en temps réel.

---

### 4. Parle avec ton IA

```bash
python src/generate.py
```

```
Toi > The future of artificial intelligence
IA  > The future of artificial intelligence is now being explored...

t=0.5 → prévisible    t=1.5 → créatif    n=200 → longueur    q → quitter
```

---

### 5. Lance les tests *(optionnel)*

Un petit filet de sécurité pour vérifier que rien n'est cassé après une modif du code :

```bash
python -m unittest discover -s tests
```

`OK` = tout va bien. Couvre le round-trip du tokenizer, la détection de changement de dataset, le modèle (forward / save-load / génération) et un test de compilation sur chaque fichier Python.

---

## Le Dashboard

Le dashboard s'ouvre automatiquement au lancement de `go.py`.

Il affiche en temps réel via **Server-Sent Events** (pas de polling) :

- RAM utilisée / totale, VRAM GPU, température, CPU
- Courbes de `train_loss` et `val_loss`
- Étape actuelle, vitesse d'entraînement
- Niveau de protection actif

Le bouton **📊 Comparer les modèles** superpose les courbes de loss de tous les modèles entraînés sur un seul graphe.

Le bouton **📚 Ouvrir la Bibliothèque** du dashboard ouvre `library.html` — la bibliothèque complète de datasets avec téléchargements en arrière-plan.

> `monitor.py` tourne silencieusement. L'affichage terminal en boucle est désactivé pour ne pas entrer en conflit avec les logs de l'entraînement — tout est visible dans le dashboard.

---

## Interpréter tes résultats

| Val Loss | Perplexité | Ce que ça veut dire |
|----------|-----------|---------------------|
| > 5.0 | > 148 | L'IA apprend les bases |
| 3.0 – 5.0 | 20 – 148 | Ça progresse |
| 2.0 – 3.0 | 7 – 20 | Le texte commence à être cohérent |
| < 2.0 | < 7 | Très bon — GPT-2 small (117M) tourne autour de 3.1 |

> Si `val_loss` monte alors que `train_loss` descend : **overfitting** — l'IA mémorise au lieu de comprendre. Solution : augmente `dropout` ou ajoute des données.

👉 **[Guide complet des paramètres](PARAMETRES.md)**

---

<details>
<summary><b>📊 Comparaison avec les alternatives</b></summary>
<br>

| Fonctionnalité | 🧠 WishAI | nanoGPT / minGPT | LitGPT | GPT-NeoX | Axolotl | DeepSpeed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Objectif** | Apprentissage + UI | Éducatif | Ingénierie | Échelle industrielle | Fine-tuning LoRA | Distribué multi-GPU |
| **Dashboard temps réel** | ✅ Local | ❌ Terminal | ⚠️ Cloud payant | ❌ | ❌ W&B externe | ❌ W&B externe |
| **Bibliothèque de datasets** | ✅ 135 curatés + 100k+ (HF/GitHub/PwC) | ❌ | ❌ | ❌ | ⚠️ Manuel | ❌ |
| **Protection VRAM & OOM** | ✅ Auto + Accumulation | ❌ Crash | ✅ CLI | ❌ | ⚠️ Manuel | ⚠️ Manuel |
| **Téléchargements en arrière-plan** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Early stopping** | ✅ Auto | ❌ Temps fixe | ❌ Manuel | ❌ | ⚠️ Config YAML | ❌ |
| **Niveau requis** | Débutants | Développeurs | Ingénieurs ML | Labos de recherche | Praticiens ML | Chercheurs |

**nanoGPT / minGPT** — parfaits pour comprendre les maths d'un Transformer. Pas d'interface, aucune protection VRAM ni d'accumulation de gradients par défaut, pas d'early stopping.

**LitGPT** — optimisations de pointe, pensé CLI. Pour un dashboard il faut leur cloud payant.

**GPT-NeoX** — fait pour 64 GPU en parallèle. Inutilisable sur une machine solo.

**Axolotl** — outil de fine-tuning (LoRA/QLoRA) sur des LLMs existants. Pas pour construire un GPT depuis zéro.

**DeepSpeed** — entraînement distribué à très grande échelle. Configs JSON complexes, clusters multi-GPU requis.

</details>

---

<details>
<summary><b>🛡️ Protections automatiques — la feature que personne n'a</b></summary>
<br>

Ton PC ne peut pas mourir pendant l'entraînement. À chaque lancement, tu choisis un niveau de protection parmi quatre — le niveau est sauvegardé dans `config.json` et réutilisé automatiquement.

**4 niveaux disponibles**

| Niveau | Pour qui | Alerte RAM | Pause RAM | Critique RAM / °C |
|--------|----------|-----------|-----------|-------------------|
| **Minim** | Machine puissante (> 32 Go) | 85% | 90% | 95% / 90°C |
| **Standard** ← défaut | 16–32 Go | 75% | 82% | 92% / 90°C |
| **Protection** | PC moyen ou laptop (8–16 Go) | 70% | 78% | 85% / 90°C |
| **Max** | PC ancien ou très limité (< 8 Go) | 60% | 70% | 80% / 89°C |

**3 phases par niveau — l'entraînement ne s'arrête JAMAIS définitivement**

| Phase | Déclencheur | Ce qui se passe |
|-------|------------|-----------------|
| **1 — Alerte** | RAM dépasse le seuil alerte | Message console + ralentissement automatique entre chaque itération |
| **2 — Pause** | RAM dépasse le seuil pause | L'entraînement se met en pause et attend en mémoire. `monitor.py` surveille et envoie le signal de reprise dès que la RAM redescend |
| **3 — Critique** | RAM ou température dépasse le seuil critique | Checkpoint sauvegardé, arrêt propre. `monitor.py` surveille les conditions. `go.py` relance automatiquement depuis le checkpoint dès que c'est bon |

**Autres protections toujours actives**

| Situation | Ce qui se passe |
|-----------|----------------|
| VRAM > 85% | Arrêt propre + sauvegarde |
| Ctrl+C | Arrêt propre + sauvegarde |
| PC qui s'éteint | Checkpoint toutes les N étapes — reprend au prochain lancement |

Pour changer de niveau : supprime `config.json` et relance `go.py`.

</details>

---

<details>
<summary><b>🔬 Architecture Transformer</b></summary>
<br>

```
[Tokens d'entrée]
       ↓
 Token Embedding  (+ RoPE appliqué dans l'attention)
       ↓
┌──────────────────────────────────────┐
│  × N couches (4 à 16 selon preset)   │
│                                      │
│  RMSNorm → Multi-Head Attention      │  ← RoPE fait tourner Q et K par position
│          + connexion résiduelle      │
│                                      │
│  RMSNorm → SwiGLU Feed-Forward (8/3×)│  ← SiLU(gate) × up → down
│          + connexion résiduelle      │
└──────────────────────────────────────┘
       ↓
  RMSNorm → Linear → Softmax → Token prédit
```

Architecture style LLaMA/Mistral (RoPE + RMSNorm + SwiGLU). Tout est commenté ligne par ligne dans le code.

</details>

---

<details>
<summary><b>🗂️ Structure des fichiers</b></summary>
<br>

```
wishai/
├── go.py               ← lance tout en une commande ← COMMENCE ICI
├── dashboard.html      ← interface du dashboard (métriques temps réel)
├── library.html        ← bibliothèque de datasets (téléchargements)
├── config.json         ← niveau de protection choisi (créé au 1er lancement)
├── control.json        ← communication entre go.py / nanogpt_bpe / monitor
├── DATASETS.md         ← liste complète des datasets disponibles
├── PARAMETRES.md       ← guide expert des paramètres d'entraînement
├── CONTRIBUTING.md     ← comment contribuer (venv, standards, guide PR)
├── src/                ← tous les scripts Python
│   ├── nanogpt_bpe.py  ← modèle + entraînement (cœur du projet)
│   ├── tokenizer.py    ← tokenizer BPE from scratch (avec barre de progression)
│   ├── generate.py     ← génération interactive
│   ├── telecharger.py  ← téléchargement de données (CLI + interface web)
│   ├── require.py      ← installation automatique des dépendances
│   ├── protection.py   ← seuils des 4 niveaux de protection
│   ├── dashboard.py    ← serveur HTTP local (dashboard + bibliothèque + API + SSE)
│   ├── monitor.py      ← serveur métriques HTTP (port dynamique) + watchdog
│   └── model.py        ← modèle Transformer (RoPE+RMSNorm+SwiGLU, importable)
├── tests/              ← smoke tests
│   └── test_smoke.py   ← python -m unittest discover -s tests
├── assets/             ← screenshots / GIFs pour le README
├── cache/              ← cache Python (__pycache__) — local au projet
├── data/               ← tes données d'entraînement (texte brut/nettoyé)
├── tokenizer.json      ← tokenizer entraîné (généré par src/tokenizer.py)
├── monitor_port.json   ← port dynamique écrit par monitor.py au démarrage
└── model/
    └── <nom>/          ← un dossier par modèle (créé automatiquement)
        ├── modele.pt       ← modèle final pour generate.py
        ├── modele.safetensors ← modèle final pour l'export (poids purs)
        ├── checkpoint.pt   ← reprise possible
        ├── log_active.json ← données temps réel du dashboard
        └── tokenizer.json  ← tokenizer utilisé pour ce modèle
```

#### Les dossiers vitaux
* **`data/`** : Stocke toutes les données d'entraînement. Les téléchargements depuis la bibliothèque atterrissent ici automatiquement.
* **`model/`** : Stocke tous les modèles IA créés (poids sauvegardés à chaque checkpoint ou à la fin).
* **`cache/`** : Réservé au `__pycache__` Python. Rien de l'App Data système — tout reste local au projet.

#### Les fichiers de configuration
* **`config.json`** : Enregistre le niveau de protection choisi. Supprime-le pour réafficher le menu au prochain lancement.
* **`control.json`** : Le "talkie-walkie" du projet. `go.py`, `nanogpt_bpe.py` et `monitor.py` communiquent via ce fichier — pause, reprise, arrêt critique, tout passe par là.

#### Les fichiers de documentation
* **`DATASETS.md`** : Liste complète des 50+ datasets disponibles, organisés par catégorie.
* **`PARAMETRES.md`** : Guide expert expliquant chaque paramètre d'entraînement (batch_size, learning_rate, dropout, etc.).

</details>

---

<details>
<summary><b>⚙️ Architecture interne — comment les composants communiquent</b></summary>
<br>

Quand tu lances `python go.py`, trois processus démarrent :

```
go.py (chef d'orchestre)
  ├── monitor.py        → port 8001  (métriques système en temps réel)
  ├── dashboard.py      → port auto  (sert dashboard.html + library.html + API REST)
  └── nanogpt_bpe.py   → terminal   (l'entraînement lui-même)
```

**`dashboard.py` — serveur HTTP complet**

`dashboard.py` n'est pas juste un lanceur de navigateur : c'est un serveur HTTP avec une API REST complète.

| Route | Méthode | Description |
|-------|---------|-------------|
| `/dashboard.html` | GET | Interface de monitoring |
| `/library.html` | GET | Bibliothèque de datasets |
| `/api/ping` | GET | Vérifie que le serveur est en ligne |
| `/api/events` | GET | Flux SSE — logs d'entraînement + session + métriques système |
| `/api/models` | GET | Historiques de tous les modèles pour la comparaison |
| `/api/downloads` | GET | Statut de tous les téléchargements en cours |
| `/api/download` | POST | Lance un téléchargement en arrière-plan |

Quand tu cliques **Télécharger** dans `library.html`, le navigateur envoie un POST à `/api/download`. `dashboard.py` lance `telecharger.py` dans un thread séparé et répond immédiatement. Tu peux poller `/api/downloads` pour suivre le statut (`running` / `done` / `error:N`).

**`control.json` — communication inter-processus**

```json
{"commande": "pause", "raison": "RAM 82%", "timestamp": 1718700000.0}
```

`nanogpt_bpe.py` lit ce fichier à chaque itération. `monitor.py` l'écrit quand les conditions de reprise sont réunies. `go.py` le lit après chaque run pour décider de relancer ou non.

</details>

---

<details>
<summary><b>❓ FAQ</b></summary>
<br>

**L'entraînement s'arrête tout seul, c'est cassé ?**
Non. En mode automatique, il s'arrête quand la val_loss ne bouge plus depuis 5 évaluations. C'est la convergence.

**Je veux reprendre un entraînement arrêté.**
Relance `python go.py` avec le même nom de modèle. Le checkpoint est détecté automatiquement.

**Le texte généré est du charabia.**
C'est normal au début. Avec val_loss > 4, l'IA apprend encore les structures de base. Laisse tourner.

**Je peux ajouter mes propres données ?**
Oui. N'importe quel fichier `.txt` UTF-8 dans `data/en/` ou `data/fr/`. Une phrase par ligne c'est bien, pas obligatoire.

**Je veux changer le niveau de protection.**
Supprime `config.json` à la racine du projet et relance `go.py`. Le menu s'affiche à nouveau.

**Dois-je réentraîner le tokenizer quand je change mes données ?**
Non. WishAI détecte le changement automatiquement (via une signature dans `tokenizer.json`) et le réentraîne au prochain lancement de `go.py`.

**Je veux télécharger un dataset HuggingFace que je ne trouve pas dans la liste.**
Ouvre la bibliothèque (bouton dans le dashboard ou `python src/telecharger.py`), onglet **Recherche HuggingFace**, tape un mot-clé. Tu as accès aux 150 000+ datasets du Hub en direct.

**Est-ce que je peux vendre le modèle que j'ai entraîné ?**
Oui. Le modèle t'appartient entièrement. La licence ne s'applique qu'au code.

</details>

---

## Licence

**WishAI Personal Use License v1.0**

✅ Utilisation gratuite — personnelle, éducative, recherche  
✅ Modification et partage autorisés (avec attribution)  
✅ Les modèles que tu entraînes t'appartiennent — fais-en ce que tu veux