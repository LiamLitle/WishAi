<div align="right">

[🇬🇧 English](DATASETS.md) | 🇫🇷 Français

</div>

# 📚 Bibliothèque de Datasets — WishAI

**135 datasets curatés**, testés et organisés par domaine. Accessibles directement depuis `library.html`.  
En plus de cette sélection : **150 000+ datasets** via la recherche HuggingFace en direct, et des milliers de dépôts GitHub via l'onglet GitHub.

---

## 🌐 Web & Corpus Généraux (16)

| Dataset | Langue | Description |
|---------|--------|-------------|
| **FineWeb** | 🇬🇧 | Le standard actuel pour entraîner les LLMs professionnels. |
| **FineWeb-Edu** | 🇬🇧 | FineWeb filtré sur le contenu éducatif uniquement. |
| **RedPajama** | 🇬🇧 | Reproduction open-source de LLaMA data : Wikipedia, C4, GitHub, Books, ArXiv. 1.2T tokens. |
| **RefinedWeb** | 🇬🇧 | Web ultra-filtré pour entraîner Falcon. 968B tokens de très haute qualité. |
| **Dolmino** | 🇬🇧 | Dataset de préentraînement AI2 : web, code, academic, books. Licence ouverte. |
| **OpenWebText** | 🇬🇧 | Réplication des données ayant servi à entraîner GPT-2. |
| **C4 (Google)** | 🇬🇧 | Corpus web propre utilisé pour entraîner T5 et Flan. |
| **Common Crawl Anglais** | 🇬🇧 | Web brut en anglais, extrêmement massif. |
| **mC4 Français** | 🇫🇷 | Texte web massif nettoyé par Google. Idéal pour la variété lexicale. |
| **CulturaX FR** | 🇫🇷 | Dataset français récent de très haute qualité. |
| **OSCAR FR** | 🇫🇷 | Grand corpus web en français. *(Compte HuggingFace requis)* |
| **Common Crawl FR** | 🇫🇷 | Web brut extrait directement depuis Common Crawl. |
| **CaBeRNet FR** | 🇫🇷 | Corpus natif : littérature contemporaine, blogs, forums. |
| **Common Crawl Multilingue** | 🌍 | Web brut dans toutes les langues. Extrêmement massif. |
| **mC4 Multilingue** | 🌍 | Version multilingue de C4 : 101 langues, 26TB de texte propre. |
| **CC-100** | 🌍 | Web crawl en 100 langues filtré depuis CommonCrawl. Base de XLM-R. |

---

## 📖 Encyclopédies & Wikis (29)

| Dataset | Langue | Articles |
|---------|--------|---------|
| **Wikipedia Français** | 🇫🇷 | 2.6M articles |
| **Wikipedia Anglais** | 🇬🇧 | 6.7M articles |
| **Wikitext-103** | 🇬🇧 | Extrait formaté, idéal pour prototyper |
| **Wikipedia Allemand** | 🇩🇪 | 2.8M articles |
| **Wikipedia Espagnol** | 🇪🇸 | 1.8M articles |
| **Wikipedia Russe** | 🇷🇺 | 1.9M articles |
| **Wikipedia Japonais** | 🇯🇵 | 1.4M articles |
| **Wikipedia Néerlandais** | 🇳🇱 | 2.1M articles |
| **Wikipedia Suédois** | 🇸🇪 | 2.6M articles |
| **Wikipedia Polonais** | 🇵🇱 | 1.5M articles |
| **Wikipedia Vietnamien** | 🇻🇳 | 1.3M articles |
| **Wikipedia Persan** | 🇮🇷 | 930k articles |
| **Wikipedia Arabe** | 🇸🇦 | 1.2M articles |
| **Wikipedia Ukrainien** | 🇺🇦 | 1.3M articles |
| **Wikipedia Turc** | 🇹🇷 | 490k articles |
| **Wikipedia Coréen** | 🇰🇷 | 690k articles |
| **Wikipedia Hongrois** | 🇭🇺 | 540k articles |
| **Wikipedia Finnois** | 🇫🇮 | 535k articles |
| **Wikipedia Tchèque** | 🇨🇿 | 500k articles |
| **Wikipedia Norvégien** | 🇳🇴 | 600k articles |
| **Wikipedia Indonésien** | 🇮🇩 | 680k articles |
| **Wikipedia Portugais** | 🇧🇷 | 1M+ articles |
| **Wikipedia Chinois** | 🇨🇳 | 1.3M articles |
| **Wikipedia Hindi** | 🇮🇳 | 160k articles |
| **Wikipedia Hébreu** | 🇮🇱 | 320k articles |
| **Wikipedia Roumain** | 🇷🇴 | 400k articles |
| **Wikipedia Danois** | 🇩🇰 | 290k articles |
| **Wikipedia Thaï** | 🇹🇭 | 160k articles |
| **Wikipedia Italien** | 🇮🇹 | 1.8M articles |

---

## 📚 Littérature & Fiction (10)

| Dataset | Langue | Description |
|---------|--------|-------------|
| **Project Gutenberg FR** | 🇫🇷 | Hugo, Zola, Verne, Maupassant — domaine public. |
| **Project Gutenberg EN** | 🇬🇧 | Shakespeare, Dickens, Austen — 50 000+ livres. |
| **Project Gutenberg Multi** | 🌍 | 70k livres en 50+ langues, classiques mondiaux. |
| **Wikisource Français** | 🇫🇷 | Poèmes, essais, discours libres de droits. |
| **Books3 (The Pile)** | 🇬🇧 | 200 000+ livres numérisés. |
| **BookCorpus** | 🇬🇧 | 11k livres indie. Base d'entraînement BERT et GPT-2. |
| **TinyStories** | 🇬🇧 | Histoires simples. Idéal pour les petits modèles. |
| **Histoires Françaises** | 🇫🇷 | Nouvelles et histoires courtes en français. |
| **FanFiction Stories** | 🇬🇧 | Fantasy, romance, sci-fi — style narratif libre. |
| **Poetry Foundation** | 🇬🇧 | 10k poèmes classiques et contemporains. |

---

## 🤖 Instructions & Alignement (15)

Ces datasets transforment un modèle brut en **assistant** capable de suivre des instructions.

| Dataset | Langue | Volume | Particularité |
|---------|--------|--------|---------------|
| **OpenHermes 2.5** | 🇬🇧 | 1M | Généré par GPT-4. Le meilleur pour l'instruction tuning. |
| **SlimOrca** | 🇬🇧 | 517k | Subset de haute qualité sélectionné depuis OpenOrca. |
| **UltraFeedback** | 🇬🇧 | 254k | Évaluations GPT-4 pour l'apprentissage par préférence (DPO). |
| **Evol-Instruct** | 🇬🇧 | 250k | Instructions évoluées automatiquement par WizardLM. |
| **UltraChat 200k** | 🇬🇧 | 200k | Conversations synthétiques de haute qualité. |
| **Helpful & Harmless RLHF** | 🇬🇧 | 170k | Anthropic : comparaisons humaines pour l'alignement. |
| **LMSYS-Chat-1M** | 🇬🇧 | 1M | Vraies conversations avec des LLMs. |
| **Capybara** | 🇬🇧 | — | 16 domaines académiques, multi-tours. |
| **Airoboros** | 🇬🇧 | — | Autogénéré : raisonnement, code, créativité, maths. |
| **Dolly 15k** | 🇬🇧 | 15k | Écrit par de vrais humains. Ton naturel. |
| **Alpaca Clean** | 🇬🇧 | 52k | Stanford. Les données originelles de l'instruction tuning. |
| **OpenAssistant OASST1** | 🌍 | — | Conversations annotées. Qualité exceptionnelle. |
| **OpenHermes Français** | 🇫🇷 | — | OpenHermes traduit en français. |
| **SQuAD Français** | 🇫🇷 | — | Compréhension de texte en français. |
| **SQuAD 2.0** | 🇬🇧 | 150k | 50k questions sans réponse incluses. |

---

## 💻 Programmation & Code (14)

| Dataset | Spécialité | Description |
|---------|-----------|-------------|
| **The Stack v2** | Tous langages | 67TB de code GitHub dans 600+ langages. |
| **The Stack — Python** | Python | Code Python propre de GitHub. Pour créer un Copilot. |
| **The Stack — JS/HTML/CSS** | Frontend | JavaScript, HTML, CSS de GitHub. |
| **StarCoder Data** | Multi-lang | Données de code multi-langage de haute qualité. |
| **CodeSearchNet** | Python | Code couplé à ses docstrings. |
| **StackOverflow Q&A** | Debugging | Questions et réponses de développeurs. |
| **CodeContests** | Compétitif | Codeforces, AtCoder, HackerEarth — DeepMind. |
| **APPS** | Compétitif | 10k problèmes avec 3 niveaux de difficulté. |
| **HumanEval** | Benchmark | 164 problèmes Python d'OpenAI. Référence absolue. |
| **MBPP** | Benchmark | 974 problèmes Python débutant→intermédiaire. |
| **LeetCode Problems** | Algorithmes | Algorithmique et structures de données. |
| **Python Code Instructions** | Instructions | 18k instructions Python commentées. |
| **Jupyter Notebooks** | Data Science | Notebooks Python publics depuis GitHub. |
| **Rust Code** | Systèmes | Codebase Rust extrait de GitHub et crates.io. |

---

## 🔢 Mathématiques & Raisonnement (8)

| Dataset | Volume | Description |
|---------|--------|-------------|
| **NuminaMath** | 860k | Olympiades, AMC, AIME, Lean — algèbre, géométrie, combinatoire. |
| **MetaMath** | 395k | Problèmes augmentés depuis GSM8K et MATH. |
| **AQuA-RAT** | 100k | Algèbre avec rationale (explication étape par étape). |
| **MATH Competition** | 12.5k | AMC, AIME, MATHCOUNTS — 5 niveaux de difficulté. |
| **GSM8K** | 8.5k | Maths primaire/collège avec solutions en langage naturel. |
| **OpenWebMath** | — | Maths haut niveau avec LaTeX. |
| **Proof-Pile** | — | Démonstrations mathématiques formelles. |
| **HotpotQA** | 113k | Questions multi-saut sur Wikipedia. |

---

## 🔬 Sciences & Recherche (8)

| Dataset | Domaine | Description |
|---------|---------|-------------|
| **Semantic Scholar ORC** | Tous | 81M papiers académiques full-text. |
| **peS2o** | Tous | 60M papiers de Semantic Scholar. |
| **ArXiv ML Papers** | ML/IA | Résumés de recherche en Machine Learning. |
| **OpenReview Papers** | ML/IA | NeurIPS, ICLR, ICML avec reviews. |
| **Biology Literature** | Bio | PubMed Central — génomique, protéomique. |
| **ChemRxiv Preprints** | Chimie | Synthèse, réactions, matériaux. |
| **OpenWebMath** | Maths | Mathématiques web de haut niveau. |
| **Proof-Pile** | Maths | Démonstrations et preuves formelles. |

---

## ⚕️ Médecine & Santé (3)

| Dataset | Description |
|---------|-------------|
| **PubMed** | Littérature médicale et biologique de référence. |
| **MedQA (USMLE)** | 12k QCM d'examens médicaux avec explications. |
| **Clinical Notes (MIMIC-III)** | Notes cliniques anonymisées de médecins. |

---

## 💬 Dialogues & Conversations (8)

| Dataset | Description |
|---------|-------------|
| **ShareGPT Conversations** | 90k conversations réelles avec ChatGPT. |
| **Movie Scripts** | Scénarios de films et séries TV. |
| **DailyDialog** | Conversations quotidiennes de haute qualité. |
| **OpenSubtitles** | Millions de répliques de films et séries. |
| **PersonaChat** | Personnages IA avec traits de personnalité. |
| **Ubuntu Dialogue** | Support technique Ubuntu. |
| **Reddit TL;DR** | Résumés automatiques de discussions Reddit. |
| **Quora Question Pairs** | Paires de questions sémantiquement proches. |

---

## ❓ Question-Réponse (3)

| Dataset | Volume | Description |
|---------|--------|-------------|
| **TriviaQA** | 650k | Quiz : histoire, science, culture populaire. |
| **Natural Questions** | 307k | Vraies requêtes Google sur Wikipedia. |
| **CommonsenseQA** | 12k | Questions de bon sens sur le monde réel. |

---

## 🔄 Traduction Multilingue (5)

| Dataset | Langues | Description |
|---------|---------|-------------|
| **OPUS-100** | 100 | Corpus de traduction parallèle massif. |
| **Tatoeba** | 400 | 10M+ paires de phrases créées par la communauté. |
| **ParaCrawl** | 23 → EN | Corpus parallèle extrait du web. |
| **FLORES-200** | 200 | Benchmark de traduction par Meta AI. |
| **Europarl Français** | FR/EU | Débats du Parlement Européen. |

---

## 📰 Actualités & Presse (4)

| Dataset | Description |
|---------|-------------|
| **CC-News** | Actualités mondiales depuis Common Crawl. |
| **CC-News Corpus** | 200k articles/jour, 60+ langues. |
| **Actualités Françaises** | Le Monde, Figaro, Libération, AFP. |
| **Wikinews** | Actualités open-source de Wikimedia. |

---

## 🎓 Éducation (4)

| Dataset | Description |
|---------|-------------|
| **FineWeb-Edu** | Web filtré sur le contenu éducatif. |
| **Cosmopedia** | 2.7M textes synthétiques éducatifs (Mixtral). |
| **Open Textbooks** | Manuels scolaires open-source (maths, sciences, économie). |
| **KhanAcademy Content** | Explications structurées niveau collège-lycée. |

---

## ⚖️ Droit & Finance (4)

| Dataset | Domaine | Description |
|---------|---------|-------------|
| **The Pile of Law** | Droit | 250GB de jurisprudence US, EU, Royaume-Uni. |
| **Textes Juridiques FR** | Droit | Légifrance, Journal Officiel, codes français. |
| **Financial News** | Finance | Bloomberg, Reuters, CNBC. |
| **Finance Instruct** | Finance | Q&A financier : rapports annuels, analyse boursière. |

---

## 🧠 Philosophie, Histoire & Culture (4)

| Dataset | Description |
|---------|-------------|
| **PhilPapers** | Articles académiques de philosophie. |
| **Textes Philosophiques** | Platon, Kant, Nietzsche, Descartes, Spinoza. |
| **WikiEvents** | Chronologie d'événements historiques mondiaux. |
| **HackerNews (The Pile)** | Discussions tech de haut niveau. |

---

## 🧭 Comment choisir ?

| Objectif | Datasets recommandés |
|----------|---------------------|
| **Modèle français général** | Wikipedia FR + mC4 + CulturaX + Gutenberg FR |
| **Assistant intelligent** | OpenHermes 2.5 + UltraChat + SlimOrca |
| **Copilot de code** | The Stack v2 + CodeContests + APPS + HumanEval |
| **Modèle de maths** | NuminaMath + MetaMath + GSM8K + MATH |
| **Médical** | PubMed + MedQA + Clinical Notes |
| **Multilingue** | mC4 Multi + CC-100 + OPUS-100 + Wikipedia (toutes langues) |
| **Légal** | Pile of Law + Textes Juridiques FR |
| **Tout en un** | RedPajama ou Dolmino (corpus de préentraînement complets) |

---

*Accès via le bouton **📚 Bibliothèque** du dashboard, ou `python src/telecharger.py`.*  
*En plus des 135 datasets curatés : **150 000+** via HuggingFace live · **GitHub** · **Papers with Code**.*
