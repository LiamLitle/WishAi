# Contribuer à WishAI

Merci de vouloir améliorer WishAI ! Ce guide explique comment contribuer de manière efficace.

---

## Structure du projet

```
AI/
├── go.py               # Point d'entrée principal (menu interactif)
├── quick.py            # Mode rapide 20M params, zéro question
├── config.py           # Gestion des modèles / données / système
├── CONTRIBUTING.md     # Ce fichier
│
├── src/
│   ├── model.py        # Architecture du modèle (RoPE + RMSNorm + SwiGLU)
│   ├── nanogpt_bpe.py  # Boucle d'entraînement principale
│   ├── tokenizer.py    # Tokenizer BPE from scratch
│   ├── telecharger.py  # Téléchargement et nettoyage des données
│   ├── dashboard.py    # Serveur HTTP + API + SSE
│   ├── monitor.py      # Surveillance mémoire / GPU (port dynamique)
│   ├── generate.py     # Génération de texte
│   ├── require.py      # Gestion des dépendances avec lock file
│   └── protection.py   # Seuils de sécurité mémoire / température
│
├── data/               # Données d'entraînement (.txt)
├── model/              # Modèles sauvegardés (modele.pt + .safetensors)
├── dashboard.html      # Interface de surveillance
└── library.html        # Bibliothèque de datasets
```

---

## Avant de coder

1. **Fork** le dépôt et clone ta fork localement.
2. Crée un **virtualenv** :
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # Linux / Mac
   ```
3. Installe les dépendances :
   ```bash
   python src/require.py
   ```
4. Crée une **branche** dédiée :
   ```bash
   git checkout -b fix/nom-du-bug
   # ou
   git checkout -b feat/nom-de-la-fonctionnalite
   ```

---

## Standards de code

- **Langue** : commentaires et noms de variables en **français** (convention du projet). Les messages d'erreur critiques peuvent être en anglais.
- **Encodage** : UTF-8 partout. Évite les emojis dans les f-strings et les dicts Python (risque de troncature par certains éditeurs).
- **Imports** : pas de dépendances externes non listées dans `require.py`. Si tu en ajoutes une, mets-la à jour.
- **Style** : pas de linter imposé, mais reste cohérent avec le code existant (4 espaces, pas de trailing whitespace).
- **Pas de checkpoint intermédiaire** : le projet sauvegarde uniquement un modèle final (`modele.pt` + `modele.safetensors`). Ne réintroduis pas de sauvegarde périodique sans discussion.

---

## Zones sensibles

| Fichier | Attention |
|---|---|
| `model.py` | Tout changement d'architecture **casse la compatibilité** avec les modèles existants. Documente clairement. |
| `tokenizer.py` | Le BPE doit rester reproductible (même seed → même vocab). |
| `nanogpt_bpe.py` | Les sections Unicode (em dash, emojis) peuvent tronquer le fichier avec certains outils. Utilise du bash/heredoc pour les éditer. |
| `monitor.py` | Le port est désormais dynamique (`monitor_port.json`). Ne hardcode plus de port. |
| `require.py` | La détection de venv et le lock file sont intentionnels. |

---

## Faire une Pull Request

1. Assure-toi que la syntaxe est valide :
   ```bash
   python -c "import ast; ast.parse(open('src/model.py').read()); print('OK')"
   ```
2. Lance un entraînement rapide pour vérifier que rien ne casse :
   ```bash
   python quick.py
   ```
3. Décris dans la PR :
   - **Quoi** : ce que tu changes
   - **Pourquoi** : le problème que ça résout
   - **Comment tester** : les étapes pour vérifier que ça marche

---

## Idées de contributions bienvenues

- Nouveaux presets GPU (RTX 4090, A100, Mac M3...)
- Support multi-GPU (`torch.nn.DataParallel`)
- Export HuggingFace (config.json + tokenizer HF)
- Interface web pour `generate.py`
- Tests unitaires (il n'en existe pas encore)
- Traduction de commentaires / docs

---

## Questions ?

Ouvre une **issue** avant de commencer un gros chantier. Ca évite les doublons et permet d'aligner sur la direction du projet.
