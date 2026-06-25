<div align="right">

[🇬🇧 English](../README.md) | 🇫🇷 Français

</div>

<div align="center">

# 🧠 WishAI

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-31210/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS%20%7C%20CPU-76b900?logo=nvidia&logoColor=white)
![Cross-Platform](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-0078D4)

<br>

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-API_Ready-FFD21E)](https://huggingface.co/)
[![Datasets](https://img.shields.io/badge/Datasets-135_curated_%2B_100k%2B-blue)](DATASETS.md)
[![Safetensors](https://img.shields.io/badge/Save-Safetensors-green)](https://huggingface.co/docs/safetensors/index)
![Dashboard](https://img.shields.io/badge/UI-HTML5_%7C_Vanilla_JS-E34F26?logo=html5&logoColor=white)

<br>

![Version](https://img.shields.io/badge/Version-1.5.2-success)
[![License](https://img.shields.io/badge/License-Non--Commercial-red)](LICENSE)

<br>

**nanoGPT c'est bien. Mais t'as pas de dashboard, pas de protection VRAM, et ton PC peut mourir en 4 minutes.**

**WishAI règle ça.**

**🎉 Version 1.5.2 — Système de reprise d'entraînement, logs `TEMP/` avancés, architecture modulaire de config, et corrections d'export.**

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

## Installation

> Prérequis : **Python 3.8+**, **pip**, **git**

```bash
git clone https://github.com/LiamLitle/WishAi
cd WishAi
python go.py
```

`go.py` installe les dépendances, vérifie le tokenizer, ouvre le dashboard, lance le moniteur en arrière-plan et démarre l'entraînement — tout en une commande.

**Tu veux tester sans rien configurer ?**

```bash
./wish quick
```

Mode zéro-config : télécharge TinyStories, entraîne le tokenizer, ouvre le dashboard et démarre un modèle de ~20M paramètres — aucune question posée.

**GPU (optionnel mais recommandé) :** Si tu as une carte NVIDIA, installe PyTorch avec le support CUDA depuis [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) — choisis ta version CUDA dans le sélecteur. Sur **Apple Silicon (M1/M2/M3)**, WishAI utilise automatiquement le backend **MPS**. Sans GPU, il tourne quand même sur CPU (preset NANO recommandé).

---

## Utilisation

### 1. Télécharge des données

```bash
./wish serve library
```

Ou clique sur **📚 Ouvrir la Bibliothèque** dans le dashboard. La **Bibliothèque de Datasets** s'ouvre dans ton navigateur. Interface unifiée avec 4 sources filtrables :

- **📌 Notre Sélection** — **135 datasets** testés et organisés en 19 domaines : Encyclopédies (29 langues), Web, Littérature, Instructions, Code, Maths, Science, Médecine, Dialogues, Traduction, Droit, Finance, Éducation, et plus.
- **🤗 HuggingFace** — accès direct aux **150 000+ datasets** du Hub. Recherche en temps réel avec debounce.
- **🐙 GitHub** — recherche de dépôts datasets sur GitHub (triés par étoiles).
- **📄 Papers with Code** — datasets académiques référencés dans des publications scientifiques (proxy serveur, pas de CORS).

Filtre par source, langue ou domaine en temps réel. Total accessible : **+100k datasets**.

Les téléchargements tournent **en arrière-plan** : tu peux en lancer plusieurs à la fois et suivre leur statut dans l'interface sans bloquer quoi que ce soit.

👉 **[Voir la liste complète des Datasets disponibles](DATASETS.md)**

Tu peux aussi **ajouter tes propres textes** : mets n'importe quel fichier `.txt` dans `data/en/` ou `data/fr/`.

---

### 2. Lance tout

```bash
./wish go
```

> Le tokenizer BPE s'entraîne automatiquement au premier lancement et se réentraîne dès que le dataset change — rien à faire manuellement.

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

### 3. Parle avec ton IA

**Mode terminal :**

```bash
./wish chat --terminal
```

```
Toi > The future of artificial intelligence
IA  > The future of artificial intelligence is now being explored...

t=0.5 → prévisible    t=1.5 → créatif    n=200 → longueur    q → quitter
```

**Interface de chat (UI web) :**

```bash
./wish chat
```

Ouvre une interface de chat entièrement repensée dans le navigateur :

- Layout fullscreen sans barre de navigation — sidebar flottante (☰) pour l'historique des conversations
- Barre de saisie pill centrée au chargement, qui se fixe en bas au premier message
- Sélecteur de modèle intégré à la zone d'input — trié par taille, chargement automatique à la sélection, affiche **✓ En cache** si le modèle est déjà en mémoire (pas de rechargement inutile)
- Température et longueur max cachées derrière **⚙ Plus d'options** avec une animation fade fluide
- Chaque réponse IA affiche au survol : un bouton **Copier**, le temps de réponse en ms, le nombre de tokens, et un bouton **↺ Régénérer**
- Historique des conversations persisté dans IndexedDB, restauré à l'ouverture suivante
- Fond avec glow violet aux bords (dégradés radiaux Canvas 2D)

Ou utilise `./wish serve` pour ouvrir le dashboard sans démarrer l'entraînement.

---

### 4. Lance les tests *(optionnel)*

Un petit filet de sécurité pour vérifier que rien n'est cassé après une modif du code :

```bash
python -m unittest discover -s tests
```

`OK` = tout va bien. Couvre le round-trip du tokenizer, la détection de changement de dataset, le modèle (forward / save-load / génération) et un test de compilation sur chaque fichier Python.

---

### 5. Gérer tes modèles *(optionnel)*

```bash
./wish config
```

<details>
<summary><b>⚙️ Menu complet de config.py</b></summary>
<br>

```
  ╔══════════════════════════════════════════════╗
  ║           WishAI  —  Configuration          ║
  ╚══════════════════════════════════════════════╝

  ── MODELES ─────────────────────────────────────
  [ 1]  Lister les modeles
  [ 2]  Supprimer un modele
  [ 3]  Supprimer TOUS les modeles
  [ 4]  Voir les hyperparametres
  [ 5]  Renommer un modele
  [ 6]  Dupliquer un modele
  [ 7]  Exporter un modele

  ── DONNEES ─────────────────────────────────────
  [ 8]  Voir les donnees disponibles
  [ 9]  Supprimer les donnees de demo
  [10]  Supprimer le cache BPE
  [11]  Regenerer le tokenizer

  ── SYSTEME ─────────────────────────────────────
  [12]  Infos PC / GPU
  [13]  Tester PyTorch + GPU
  [14]  Logs du dernier entrainement
  [15]  Reinitialiser les dependances (suppr. deps.lock)
  [16]  Desinstaller les dependances (pip uninstall)
  [17]  Reset complet (tout effacer)

  [ 0]  Quitter
```

Chaque option est interactive — elle demande confirmation avant toute suppression.

**Les options les plus utiles :**
- **[1]** — voir tous tes modèles entraînés avec leur val loss, taille et date
- **[4]** — inspecter l'architecture et les hyperparamètres d'un modèle
- **[12]** — vérifier ton GPU, VRAM, RAM, versions Python et PyTorch
- **[13]** — lancer un benchmark rapide de multiplication matricielle sur GPU
- **[14]** — voir les 5 dernières évaluations (train loss / val loss) dans le terminal

</details>

---

## Le Dashboard

Le dashboard s'ouvre automatiquement au lancement de `go.py`.

Il affiche en temps réel via **Server-Sent Events** (pas de polling) :

- RAM utilisée / totale, VRAM GPU, température, CPU
- Courbes de `train_loss` et `val_loss`
- Étape actuelle, vitesse d'entraînement, niveau de protection actif

**Quand aucun entraînement ne tourne :** écran idle animé avec particules flottantes et cerveau en lévitation — auto-redirect de `file://` vers `localhost` si le serveur est trouvé.

**Quand l'entraînement se termine :** bannière verte "Entraînement terminé" avec les stats finales.

**4 sections dépliables** avec état mémorisé entre les rechargements :

| Section | Ce qu'elle affiche |
|---------|-------------------|
| 📈 Tendances & Convergence | Δ val loss, direction, vitesse de descente, plateau estimé |
| ⚡ Performance réelle | Tokens/s, Mo traités, batch effectif, nombre de checkpoints |
| 🧠 Analyse du modèle | Params/couche, VRAM théorique, head size, état + conseil |
| 📋 Journal des événements | Records val_loss, pauses thermiques/RAM, alertes convergence |

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

👉 **[Guide complet des paramètres](../docs/PARAMETRES.md)**

---

<details>
<summary><b>📊 Comparaison avec les alternatives</b></summary>
<br>

| Fonctionnalité | 🧠 WishAI | nanoGPT | nanochat | LitGPT | GPT-NeoX | Axolotl | DeepSpeed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Objectif** | Apprentissage + UI | Éducatif | Édu / Full-stack | Ingénierie | Échelle industrielle | Fine-tuning LoRA | Distribué multi-GPU |
| **Dashboard temps réel** | ✅ Local | ❌ Terminal | ⚠️ Basic UI | ⚠️ Cloud payant | ❌ | ❌ W&B externe | ❌ W&B externe |
| **Bibliothèque de datasets** | ✅ 135 curatés + 100k+ (HF/GitHub/PwC) | ❌ | ❌ | ❌ | ❌ | ⚠️ Manuel | ❌ |
| **Protection VRAM & OOM** | ✅ Auto + Accumulation | ❌ Crash | ❌ Crash | ✅ CLI | ❌ | ⚠️ Manuel | ⚠️ Manuel |
| **Téléchargements en arrière-plan** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Early stopping** | ✅ Auto | ❌ Temps fixe | ❌ Temps fixe | ❌ Manuel | ❌ | ⚠️ Config YAML | ❌ |
| **Niveau requis** | Débutants | Développeurs | Développeurs | Ingénieurs ML | Labos de recherche | Praticiens ML | Chercheurs |
| **Multi-GPU / Cluster** | ❌ Solo (1 GPU) | ✅ Basique (DDP) | ❌ Solo | ✅ FSDP/DDP | ✅ Megatron | ✅ FSDP | ✅ Natif |
| **LoRA / Fine-tuning** | ❌ Entraînement 0 | ⚠️ Basique | ✅ Oui | ✅ Oui | ❌ | ✅ QLoRA | ✅ Oui |
| **API Prod / Serving** | ❌ Usage local | ❌ | ⚠️ API Minime | ✅ vLLM/LitServe | ❌ | ❌ | ✅ Oui |
| **Quantification (4/8-bit)** | ❌ FP16/BF16 | ❌ FP16/BF16 | ❌ FP16/BF16 | ✅ Oui | ❌ | ✅ Oui | ✅ Oui |
| **Exportation (GGUF/ONNX)** | ❌ PyTorch unique | ⚠️ Scripts persos | ❌ | ✅ Oui | ❌ | ✅ Oui | ❌ |
| **Poids pré-entraînés** | ❌ Entraînement de 0 | ✅ GPT-2 | ❌ | ✅ Nombreux | ✅ EleutherAI | ✅ Tout HuggingFace | ⚠️ Selon framework |
| **Architectures Custom** | ❌ Fixe (Style LLaMA) | ✅ Code modifiable | ✅ Code modifiable | ✅ Modulaire | ✅ Modulaire | ❌ Config YAML | ✅ Flexible |

**[nanoGPT](https://github.com/karpathy/nanoGPT)** — parfait pour comprendre les maths d'un Transformer. Pas d'interface, aucune protection VRAM ni d'accumulation de gradients par défaut, pas d'early stopping. **WishAI** s'inspire grandement de son moteur.

**[nanochat](https://github.com/karpathy/nanochat)** — "le meilleur ChatGPT pour 100$" (par Andrej Karpathy, 2025/2026). Couvre tout le pipeline (pré-entraînement, finetuning, UI). Génial pour avoir la stack complète la plus minimale possible, mais manque de protections matérielles et de datasets automatiques par rapport à WishAI.

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

Pour changer de niveau : supprime `system/config.json` et relance `go.py`.

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
├── wish.bat            ← raccourcis : ./wish go | chat | quick | config | serve | visual
├── dashboard.html      ← dashboard (métriques temps réel)
├── requirements.txt
├── DATASETS.md         ← liste complète des datasets disponibles
├── CONTRIBUTING.md
├── scripts/            ← scripts secondaires
│   ├── chat.py         ← ./wish chat
│   ├── quick.py        ← ./wish quick  (mode zéro-config)
│   ├── serve.py        ← ./wish serve  (dashboard/bibliothèque sans entraînement)
│   └── config.py       ← ./wish config (gestion modèles et données)
├── docs/               ← documentation
│   ├── PARAMETRES.md   ← guide expert des paramètres d'entraînement
│   └── LAUNCH.md       ← guide de lancement communauté
├── web/
│   └── library.html    ← bibliothèque de datasets (./wish serve library)
├── FR/                 ← versions françaises de toute la documentation
│   ├── README.md, CHANGELOG.md, CONTRIBUTING.md
│   ├── DATASETS.md, PARAMETRES.md, LICENSE.md
├── system/             ← fichiers runtime (générés automatiquement, ne pas modifier)
│   ├── control.json, session.json, tokenizer.json …
├── chatting/           ← interface de chat web
│   ├── index.html, style.css, app.js
├── src/                ← tous les scripts Python core
│   ├── nanogpt_bpe.py  ← modèle + entraînement (cœur du projet)
│   ├── tokenizer.py    ← tokenizer BPE from scratch
│   ├── chat_server.py  ← serveur web + génération terminal
│   ├── telecharger.py  ← téléchargement de datasets
│   ├── require.py      ← installation automatique des dépendances
│   ├── protection.py   ← seuils VRAM/RAM/temp
│   ├── dashboard.py    ← serveur HTTP (dashboard + API + SSE)
│   ├── monitor.py      ← serveur métriques système + watchdog
│   ├── chat_server.py  ← serveur HTTP chat (stream SSE)
│   └── model.py        ← Transformer (RoPE + RMSNorm + SwiGLU)
├── tests/
│   └── test_smoke.py   ← python -m unittest discover -s tests
├── assets/             ← screenshots / GIFs pour le README
├── data/               ← tes données d'entraînement (fichiers .txt)
├── visual/             ← visualiseur d'embeddings (./wish visual)
└── model/
    └── <nom>/
        ├── modele.pt, modele.safetensors, checkpoint.pt
        ├── log_active.json, tokenizer.json
```

</details>

---

<details>
<summary><b>⚙️ Architecture interne — comment les composants communiquent</b></summary>
<br>

Quand tu lances `python go.py`, trois processus démarrent :

```
go.py (chef d'orchestre)
  ├── monitor.py        → port auto  (métriques système en temps réel)
  ├── dashboard.py      → port auto  (sert dashboard.html + library.html + API REST)
  └── nanogpt_bpe.py   → terminal   (l'entraînement lui-même)
```

**`dashboard.py` — serveur HTTP complet**

| Route | Méthode | Description |
|-------|---------|-------------|
| `/dashboard.html` | GET | Interface de monitoring |
| `/library.html` | GET | Bibliothèque de datasets |
| `/api/ping` | GET | Vérifie que le serveur est en ligne |
| `/api/events` | GET | Flux SSE — logs d'entraînement + session + métriques système |
| `/api/models` | GET | Historiques de tous les modèles pour la comparaison |
| `/api/downloads` | GET | Statut de tous les téléchargements en cours |
| `/api/download` | POST | Lance un téléchargement en arrière-plan |

**`system/control.json` — communication inter-processus**

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
Supprime `system/config.json` et relance `go.py`. Le menu s'affiche à nouveau.

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
❌ Tu ne peux pas vendre ce logiciel sans autorisation écrite  
❌ Tu ne peux pas te prétendre auteur de ce projet  

Voir [LICENSE.md](LICENSE.md) pour les termes complets.

---

<div align="center">

Construit par Liam — from scratch.

</div>
