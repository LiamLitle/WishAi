# 🔧 Paramètres de WishAI — Guide Expert

Ce fichier explique chaque paramètre de la configuration Expert.
Reviens ici quand tu as une question sur un paramètre spécifique.

---

## batch_size & grad_accum_steps — L'Accumulation de Gradients

À chaque étape, l'IA apprend sur un "lot" d'exemples simultanément.
`batch_size` = taille du lot chargé en VRAM à la fois.
Pour éviter l'erreur `CUDA Out of Memory`, on utilise l'**Accumulation de Gradients**.

**Comment ça marche :**
Avec `batch_size = 4` et `grad_accum_steps = 4` :
L'IA lit 4 fois un lot de 4 exemples, accumule les erreurs, et fait une seule mise à jour.
**Batch effectif = 16** — sans faire exploser la VRAM.

**Configurations recommandées :**

| Config | batch_size | grad_accum | Effectif | Profil |
|--------|-----------|------------|---------|--------|
| Légère | 4 | 2 | 8 | CPU / < 4 Go VRAM |
| Standard | 4 | 4 | 16 | 6–8 Go VRAM |
| Gros modèle | 4 | 8 | 32 | 12+ Go VRAM |

> **Règle d'or :** garde `batch_size` bas (4), augmente `grad_accum_steps` si tu veux plus de stabilité.

---

## block_size — La Fenêtre de Contexte

Nombre de tokens que l'IA peut lire simultanément pour prédire le suivant. C'est sa mémoire à court terme.

| block_size | ≈ mots de contexte | Usage |
|---|---|---|
| 64 | ~45 mots | Très rapide, phrases très courtes |
| 128 | ~90 mots | Prototypage |
| 256 | ~180 mots ← | **Bon équilibre** |
| 512 | ~360 mots | Paragraphes complets |
| 1024 | ~730 mots | Textes longs, lent |

> Avec BPE, 1 token ≈ 0.7 mot en anglais, ≈ 0.6 mot en français.

---

## n_embd — La Richesse Interne

Chaque token est représenté par un vecteur de `n_embd` nombres.
Plus c'est grand, plus l'IA nuance sa compréhension — mais plus ça coûte en VRAM.

| Preset | n_embd | Analogie |
|---|---|---|
| NANO | 128 | Esquisse rapide |
| SMALL | 256 | Dessin au crayon |
| MEDIUM | 512 | Peinture détaillée |
| LARGE | 768 | GPT-2 small |

> **Contrainte :** `n_embd` doit être divisible par `n_head`.  
> Exemple : n_embd=512 → n_head valides : 1, 2, 4, **8**, 16, 32

---

## n_head — Les Têtes d'Attention

Le mécanisme d'attention est divisé en `n_head` têtes parallèles.
Chaque tête apprend à repérer un type de relation différent dans le texte.

**Ce que les têtes apprennent (en pratique) :**
- Sujets et verbes
- Pronoms et références (il, elle, ce...)
- Négations et nuances
- Relations syntaxiques
- L'IA décide elle-même — on ne contrôle pas ça

**Règle absolue :** `n_embd` divisible par `n_head` sans reste.

---

## n_layer — La Profondeur du Réseau

Nombre de blocs Transformer empilés. Chaque couche raffine la compréhension.

| n_layer | Vitesse | Qualité | Recommandé pour |
|---|---|---|---|
| 4 | ⚡ Très rapide | Structures simples | NANO, tests |
| 6 | ⚡ Rapide | Bon équilibre | SMALL |
| 8 | 🔄 Moyen | Bien | Usage général |
| 12 | 🐢 Lent | Relations complexes | MEDIUM, LARGE |
| 16 | 🐢 Très lent | Très profond | Gros modèles |

---

## dropout — Protection contre l'Overfitting

À chaque étape, `dropout × 100`% des connexions sont désactivées aléatoirement.
Cela force l'IA à généraliser plutôt que mémoriser.

| dropout | Effet | Quand l'utiliser |
|---|---|---|
| 0.0 | Désactivé | Grands datasets (>1M tokens) |
| 0.1 | Léger | Bon point de départ |
| 0.2 | **Recommandé** | Cas général |
| 0.3 | Fort | Petit dataset, risque d'overfitting |
| 0.5 | Très fort | Dataset très petit |

---

## learning_rate — La Vitesse d'Apprentissage

Contrôle la taille du "pas" lors de chaque mise à jour des paramètres.

| Valeur | Effet | Usage |
|---|---|---|
| `1e-3` = 0.001 | Trop grand — instable | Éviter |
| `3e-4` = 0.0003 | **Standard GPT** | Préentraînement depuis zéro |
| `1e-4` = 0.0001 | Conservateur | Grands modèles |
| `1e-5` = 0.00001 | Très petit | Fine-tuning uniquement |

**Symptôme d'un LR trop grand :** `train_loss` remonte ou devient NaN.  
**Symptôme d'un LR trop petit :** convergence très lente, plateau précoce.

> WishAI utilise un **LR scheduler cosinus** : décroissance douce jusqu'à `min_lr = lr × 0.1`.

---

## eval_interval & eval_iters — Fréquence des Mesures

`eval_interval` : tous les N pas, calculer la `val_loss`.
`eval_iters` : nombre de batches utilisés pour estimer la val_loss.

| eval_interval | eval_iters | Effet |
|---|---|---|
| 100 | 50 | Dashboard très réactif, ~5% plus lent |
| 500 | 100 | **Bon équilibre** |
| 1000 | 200 | Entraînement rapide, courbe moins fine |

---

## Comprendre val_loss et train_loss

**train_loss** — erreur sur les données d'entraînement. Doit descendre.  
**val_loss** — erreur sur données jamais vues. C'est la vraie mesure de performance.

| Situation | Interprétation | Action |
|---|---|---|
| Les deux descendent | ✅ Apprentissage normal | Continuer |
| train_loss ↓, val_loss ↑ | ⚠️ Overfitting | Augmenter dropout, ajouter données |
| Les deux stagnent | 🔄 Convergence | WishAI s'arrête automatiquement |
| val_loss → NaN | 🚨 LR trop grand | Diviser learning_rate par 3 |

---

## Perplexité — Lire les Résultats

`perplexité = e^(val_loss)`

Représente "entre combien de mots l'IA hésite" à chaque prédiction.

| val_loss | Perplexité | Interprétation |
|---|---|---|
| > 5.0 | > 148 | Apprentissage des bases |
| 3.5–5.0 | 33–148 | Structures émergentes |
| 2.5–3.5 | 12–33 | Texte commence à être cohérent |
| 2.0–2.5 | 7–12 | Bon niveau |
| < 2.0 | < 7 | Excellent |

> GPT-2 small (117M params) atteint ~3.1 sur WikiText-103.  
> Preset MEDIUM (~40M params) : viser **2.5–3.5** est réaliste.

---

## Overfitting — L'Ennemi Silencieux

L'overfitting = l'IA mémorise les données au lieu de comprendre le langage.

**Symptômes :**
- `train_loss` très bas, `val_loss` qui remonte
- Texte généré répète les phrases d'entraînement mot pour mot

**Solutions (par ordre d'efficacité) :**
1. Ajouter plus de données d'entraînement
2. Augmenter `dropout` (ex : 0.2 → 0.3)
3. Réduire la taille du modèle (`n_layer` ou `n_embd`)
4. Arrêter l'entraînement tôt — WishAI le détecte automatiquement

---

## Checkpoints — Ne Jamais Perdre son Travail

Un checkpoint = sauvegarde complète (poids + optimizer + étape actuelle).

**Si ton PC s'éteint ou si tu arrêtes l'entraînement :**
Relance `python go.py` avec le même nom de modèle.
WishAI détecte le checkpoint et reprend exactement là où il s'était arrêté.

| Fréquence | Effet |
|---|---|
| 100 étapes | Très sûr, léger ralentissement |
| 500 étapes | **Recommandé** |
| 5000 étapes | Rapide, mais risque de perdre du travail |

---

## Présets Recommandés

| Preset | n_embd | n_head | n_layer | block_size | Params | VRAM min |
|--------|--------|--------|---------|-----------|--------|---------|
| 🐢 NANO | 128 | 4 | 4 | 256 | ~2M | CPU |
| 🚀 SMALL | 256 | 8 | 6 | 256 | ~10M | 4 Go |
| ⚡ MEDIUM | 512 | 8 | 12 | 512 | ~40M | 6 Go |
| 🧠 LARGE | 768 | 12 | 12 | 1024 | ~85M | 12 Go |
