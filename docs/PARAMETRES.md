<div align="right">

🇬🇧 English | [🇫🇷 Français](FR/PARAMETRES.md)

</div>

# 🔧 WishAI Parameters — Expert Guide

This file explains every parameter in the Expert configuration.
Come back here whenever you have a question about a specific parameter.

---

## batch_size & grad_accum_steps — Gradient Accumulation

At each step, the AI learns on a "batch" of examples simultaneously.
`batch_size` = size of the batch loaded into VRAM at once.
To avoid `CUDA Out of Memory` errors, we use **Gradient Accumulation**.

**How it works:**
With `batch_size = 4` and `grad_accum_steps = 4`:
The AI reads 4 batches of 4 examples, accumulates the errors, and does a single update.
**Effective batch = 16** — without blowing up VRAM.

**Recommended configurations:**

| Config | batch_size | grad_accum | Effective | Profile |
|--------|-----------|------------|-----------|---------|
| Light | 4 | 2 | 8 | CPU / < 4 GB VRAM |
| Standard | 4 | 4 | 16 | 6–8 GB VRAM |
| Large model | 4 | 8 | 32 | 12+ GB VRAM |

> **Golden rule:** keep `batch_size` low (4), increase `grad_accum_steps` for more stability.

---

## block_size — The Context Window

Number of tokens the AI can read simultaneously to predict the next one. It is its short-term memory.

| block_size | ≈ words of context | Usage |
|---|---|---|
| 64 | ~45 words | Very fast, very short sentences |
| 128 | ~90 words | Prototyping |
| 256 | ~180 words ← | **Good balance** |
| 512 | ~360 words | Full paragraphs |
| 1024 | ~730 words | Long texts, slow |

> With BPE, 1 token ≈ 0.7 word in English, ≈ 0.6 word in French.

---

## n_embd — Internal Richness

Each token is represented by a vector of `n_embd` numbers.
The larger it is, the more the AI nuances its understanding — but the more VRAM it costs.

| Preset | n_embd | Analogy |
|---|---|---|
| NANO | 128 | Quick sketch |
| SMALL | 256 | Pencil drawing |
| MEDIUM | 512 | Detailed painting |
| LARGE | 768 | GPT-2 small |

> **Constraint:** `n_embd` must be divisible by `n_head`.
> Example: n_embd=512 → valid n_head values: 1, 2, 4, **8**, 16, 32

---

## n_head — Attention Heads

The attention mechanism is split into `n_head` parallel heads.
Each head learns to spot a different type of relationship in the text.

**What heads learn (in practice):**
- Subjects and verbs
- Pronouns and references (he, she, it...)
- Negations and nuances
- Syntactic relationships
- The AI decides itself — we don't control this

**Absolute rule:** `n_embd` must be divisible by `n_head` with no remainder.

---

## n_layer — Network Depth

Number of stacked Transformer blocks. Each layer refines understanding.

| n_layer | Speed | Quality | Recommended for |
|---|---|---|---|
| 4 | ⚡ Very fast | Simple structures | NANO, tests |
| 6 | ⚡ Fast | Good balance | SMALL |
| 8 | 🔄 Medium | Good | General use |
| 12 | 🐢 Slow | Complex relationships | MEDIUM, LARGE |
| 16 | 🐢 Very slow | Very deep | Large models |

---

## dropout — Protection Against Overfitting

At each step, `dropout × 100`% of connections are randomly disabled.
This forces the AI to generalize rather than memorize.

| dropout | Effect | When to use |
|---|---|---|
| 0.0 | Disabled | Large datasets (>1M tokens) |
| 0.1 | Light | Good starting point |
| 0.2 | **Recommended** | General case |
| 0.3 | Strong | Small dataset, overfitting risk |
| 0.5 | Very strong | Very small dataset |

---

## learning_rate — Learning Speed

Controls the size of the "step" at each parameter update.

| Value | Effect | Usage |
|---|---|---|
| `1e-3` = 0.001 | Too large — unstable | Avoid |
| `3e-4` = 0.0003 | **GPT standard** | Pretraining from scratch |
| `1e-4` = 0.0001 | Conservative | Large models |
| `1e-5` = 0.00001 | Very small | Fine-tuning only |

**Symptom of LR too large:** `train_loss` rises or becomes NaN.
**Symptom of LR too small:** very slow convergence, early plateau.

> WishAI uses a **cosine LR scheduler**: smooth decay down to `min_lr = lr × 0.1`.

---

## eval_interval & eval_iters — Measurement Frequency

`eval_interval`: every N steps, calculate the `val_loss`.
`eval_iters`: number of batches used to estimate val_loss.

| eval_interval | eval_iters | Effect |
|---|---|---|
| 100 | 50 | Very reactive dashboard, ~5% slower |
| 500 | 100 | **Good balance** |
| 1000 | 200 | Fast training, less detailed curve |

---

## Understanding val_loss and train_loss

**train_loss** — error on training data. Should go down.
**val_loss** — error on never-seen data. This is the real performance measure.

| Situation | Interpretation | Action |
|---|---|---|
| Both going down | ✅ Normal learning | Keep going |
| train_loss ↓, val_loss ↑ | ⚠️ Overfitting | Increase dropout, add data |
| Both stagnating | 🔄 Convergence | WishAI stops automatically |
| val_loss → NaN | 🚨 LR too large | Divide learning_rate by 3 |

---

## Perplexity — Reading the Results

`perplexity = e^(val_loss)`

Represents "between how many words the AI hesitates" at each prediction.

| val_loss | Perplexity | Interpretation |
|---|---|---|
| > 5.0 | > 148 | Learning the basics |
| 3.5–5.0 | 33–148 | Emerging structures |
| 2.5–3.5 | 12–33 | Text starts to be coherent |
| 2.0–2.5 | 7–12 | Good level |
| < 2.0 | < 7 | Excellent |

> GPT-2 small (117M params) reaches ~3.1 on WikiText-103.
> MEDIUM preset (~40M params): targeting **2.5–3.5** is realistic.

---

## Overfitting — The Silent Enemy

Overfitting = the AI memorizes data instead of understanding language.

**Symptoms:**
- `train_loss` very low, `val_loss` rising
- Generated text repeats training sentences word for word

**Solutions (by effectiveness):**
1. Add more training data
2. Increase `dropout` (e.g. 0.2 → 0.3)
3. Reduce model size (`n_layer` or `n_embd`)
4. Stop training early — WishAI detects this automatically

---

## Checkpoints — Never Lose Your Work

A checkpoint = complete save (weights + optimizer + current step).

**If your PC shuts down or you stop training:**
Relaunch `python go.py` with the same model name.
WishAI detects the checkpoint and resumes exactly where it left off.

| Frequency | Effect |
|---|---|
| 100 steps | Very safe, slight slowdown |
| 500 steps | **Recommended** |
| 5000 steps | Fast, but risk of losing work |

---

## Recommended Presets

| Preset | n_embd | n_head | n_layer | block_size | Params | Min VRAM |
|--------|--------|--------|---------|-----------|--------|---------|
| 🐢 NANO | 128 | 4 | 4 | 256 | ~2M | CPU |
| 🚀 SMALL | 256 | 8 | 6 | 256 | ~10M | 4 GB |
| ⚡ MEDIUM | 512 | 8 | 12 | 512 | ~40M | 6 GB |
| 🧠 LARGE | 768 | 12 | 12 | 1024 | ~85M | 12 GB |
