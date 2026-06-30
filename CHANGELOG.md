# Changelog

All notable changes and improvements to WishAI.

Format inspired by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

[🇫🇷 Version française](FR/CHANGELOG.md)

---

## [1.5.2] — 2026-06-25

Major update improving training reliability, logging, and internal architecture.

### ✨ Added

- **Training Recovery System**: The AI now robustly saves its state, allowing users to resume a training session exactly where it stopped (useful after a crash, manual interruption, or human error).
- **Advanced Logging & Temporary Storage**: Created a dedicated `TEMP/` subdirectory with precise timestamps (exact date, hours, seconds) for a granular training history without polluting the main directory.
- **Enriched `log_active.json`**: Added critical real-time system information:
  - AMP (Automatic Mixed Precision) and `torch.compile()` activation status.
  - Real model parameter count.
  - BPE and Tokenization status, including data download percentages and specific files/folders used.

### ♻️ Changed

- **Dynamic Model Detection**: Improved the dashboard's active model detection (no longer blindly relies on `active.json`).
- **Word Cloud User Experience (UX)**: The 4,000 token point cloud has been strictly flattened into 2D (Z-axis fixed to 0) and 3D rotation has been disabled, making the interface significantly more legible.
- **Modular configuration architecture**: `scripts/config.py` has been refactored. Instead of one giant file, the codebase is now cleanly separated into a `scripts/Config/` directory (`donnees.py`, `modeles.py`, `systeme.py`, `utilitaires.py`).
- **ONNX/GGUF export optimization**: Model exports are now neatly generated directly inside each model's directory (`model/<model_name>/`) along with an automatic copy of the `tokenizer.json` file, rather than prompting the user to copy the entire folder to the Desktop.

### 🐛 Fixed

- **"Shadowing" bug (`token.py`)**: The `scripts/token.py` script created an internal namespace conflict with standard Python packages, which randomly reset the tokenizer during model exports. It has been renamed to `scripts/reset_token.py`.
- **Export loading crash**: `export.py` was modified to correctly handle dynamic model loading from training checkpoints (weight dictionaries + `log_active.json` for config) instead of assuming the full PyTorch object was saved in the `.pt` file, fixing the `AttributeError: 'dict' object has no attribute 'eval'` crash.

---

## [1.5.1] — 2026-06-25

### ✨ Added

- **Julia Analytic Engine — Advanced Convergence Predictions**
  - **`src/estimations.jl`** — new Julia background process running alongside the Python training loop. Performs two analyses Python cannot do efficiently:
    - **Chinchilla overfitting risk** — computes the parameter-to-token ratio and classifies risk.
    - **Exponential plateau prediction** — fits a curve to estimate the asymptotic loss the model will converge to.
  - **File-based IPC** — Julia reads `model/{name}/log_active.json` and outputs `model/{name}/insights.json`.
  - **Automatic Julia installation** — `src/require.py` calls `check_and_install_julia()` at startup.
  - **Graceful degradation** — if Julia is absent, `go.py` skips the Julia process without crashing.
- **Best Model Checkpointing**: `nanogpt_bpe.py` now saves `best_model.pt` whenever `val_loss` reaches a new all-time low.
- **Tests**: **`tests/test_all.py`** — automated test suite covering all v1.5.1 features.

### ♻️ Changed

- **Adaptive Learning Rate (Cosine Decay)**
  - **`get_lr(iteration)`** in `nanogpt_bpe.py` — replaces the previous fixed learning rate with a two-phase schedule (Warmup then Cosine decay).
  - Current LR is logged to `log_active.json` as `lr_actuel`.
- **Dashboard — Julia Metrics Integration**
  - **Learning Rate card** — new card in the main metrics row showing the current LR.
  - **Plateau prediction (Julia)** — the "Plateau estimé" cell now displays Julia's exponential curve fit result.
  - **Chinchilla Overfitting cell** — new cell showing the risk level.

---

## [1.5.0] — 2026-06-24

### ✨ Added

- **Download bot — real sizes & custom sources**
  - **Real HuggingFace dataset sizes** — `obtenir_taille_hf()` queries the HuggingFace API for actual dataset sizes.
  - **Size display in source menu** — each source now shows its max and typical size.
  - **Custom sources** (`src/custom_sources.json`) — add your own HuggingFace datasets without touching the Python code.
  - **Language filter on HuggingFace datasets** — new `langue_cible` field on sources that mix multiple languages.
  - **Mandatory language selection** — forces an explicit language choice in both preset and custom bot modes.
  - **`langdetect` integration** — optional library imported at startup via `try/except`.
- **Hot-reload & checkpoints during training**
  - **Hot-reload** — after a new source is downloaded, training automatically incorporates it without restarting via `reload_requested.flag`.
  - **Checkpoint saving during training** — checkpoints are now saved every `checkpoint_interval` steps.

### ♻️ Changed

- **Manifest system — 50% disk space saved**
  - **`data/{lang}/manifest.json`** replaces `data.txt` copying, files read via streaming.
  - **`telecharger.py`** — writes the manifest without copying any file.
  - **`tokenizer.py`** and **`nanogpt_bpe.py`** modified to read via manifest.

---

## [1.4.0] — 2026-06-24

### ✨ Added

- **Auto-bot & data download**
  - **Auto-bot** (`[a]` in the library) — intelligent download system detecting free disk space (Preset or Custom mode).
  - **HuggingFace retry** — up to 4 attempts with exponential backoff on connection reset.
- **Logging & auto-repair**
  - **`src/bot_logger.py`** — logging system for the bot.
  - **Auto-repair** — automatically uninstalls and reinstalls missing or corrupted dependencies via pip.
  - **`tests/test_sources.py`** — sends a request to every HuggingFace source and reports speed/status.
- **New commands**
  - `./wish token`, `./wish repair`, `./wish logs`, `./wish chat --terminal`

### ♻️ Changed

- **Tokenizer**: **Dynamic sample size** — BPE training now uses 15% of the data file (min 5 MB, max 50 MB) instead of a fixed 5 MB.

### 🐛 Fixed

- **Library menu**: empty input or invalid choice no longer exits; loops until a valid selection is made.

---

## [1.3.3] — 2026-06-24

### ✨ Added

- **Launcher system (`wish.bat`)** — single entry point for all commands (`go`, `chat`, `quick`, `config`, `serve`, `visual`).
- **Uninstall dependencies**: new option `[16]` in `config.py`.

---

## [1.3.2] — 2026-06-24

### ♻️ Changed

- **Project structure**
  - Root cleanup (only 8 files remain).
  - Launcher scripts moved to `scripts/`.
  - Documentation moved to `docs/`.
  - Interface HTML moved to `web/`.
  - System files (`session.json`, etc.) moved to `system/`.

---

## [1.3.1] — 2026-06-23 20:07.pm

### ♻️ Changed

- **Embedding visualizer improvements**
  - Flat colors (removed all neon/glow effects).
  - Hover-only labels.
  - Smart filter system (Words, Numbers, Specials, End-of-word).
  - End-of-word tokens (`</w>`) rendered at 75% of base radius.

---

## [1.3.0] — 2026-06-23

### ✨ Added

- **IndexedDB history** in the chat interface.

### ♻️ Changed

- **Chat interface — complete redesign**
  - Topbar removed, pill input bar.
  - Integrated model selector, "More options" with fade-in.
  - Floating sidebar for history with glassmorphism effect.
  - Violet edge glow background.
  - Improved chat bubbles (animations, copy button, response time).
  - **↺ Regenerate** button.
  - Auto-load on model change (client-side cache).

### 🐛 Fixed

- **Architecture mismatch in `chat_server.py`**: fixed by replacing `build_modele()` with a direct import of `WishAI_BPE` and `ConfigModele`.
- **Missing spaces in generated text**: added `decode_token(id)` to prevent premature `.strip()` on `</w>`.
- **`KeyError: 'hyperparams'`**: backwards compatibility for older checkpoints.

---

## [1.2.0] — 2026-06-21

### ✨ Added

- **Dashboard — complete UI**
  - Animated idle screen with auto-redirect.
  - "Training complete" banner.
  - 4 collapsible sections (Trends, Real performance, Model analysis, Event log).
  - Last generated text display and hyperparameter table.
  - Session tracking via SSE.
- **`quick.py`** — zero-config fast mode.
- **Chat interface** web UI and `serve.py` server.
- **Phase 3 resume loop** in `go.py`.

### 🐛 Fixed

- Duplicate code blocks removed in `library.html`.
- Missing `import sys` in `src/telecharger.py`.

### 📄 Documentation

- Full 16-command menu documented in `README.md`.

---

## [1.1.0] — 2026-06-21

### ♻️ Changed

- **Modernized model architecture**: moved from GPT-2 (2019) to LLaMA/Mistral-style (2024).
  - RMSNorm, RoPE, SwiGLU, all linear layers without bias (`bias=False`).

### ⚡ Improved

- **Dynamic monitor port**: looks for a free TCP port (`monitor_port.json`).
- **Virtualenv warning** in `require.py`.
- **Dashboard SSE**: replaced polling with Server-Sent Events.
- **Multi-model comparison** panel in dashboard.
- **Common Crawl quality filtering**: robust 4-pass cleaning.

### 🐛 Fixed

- **Empty `data/data.txt` blocker**: tokenizer avoids BPE training on empty text.

### 📄 Documentation

- **`CONTRIBUTING.md`** created.

---

## [1.0.0] — 2026-06-21

First stable version. Code review, bug fixes and safety net (tests).

### ✨ Added

- **Apple Silicon (MPS) support**.
- **Mixed precision (AMP)** (autocast bf16/fp16).
- **`torch.compile`** enabled on Linux+CUDA.
- **Smoke tests** (`tests/test_smoke.py`).

### ♻️ Changed

- Model extracted to `src/model.py`.

### 🐛 Fixed

- Stray `shu` line in `nanogpt_bpe.py`.
- `supprimer_donnees()` undefined error.
- Tokenizer not retrained on dataset change (added dataset signature).
- `control.json` non-atomic write issue.
- Dashboard flash at startup.

### 🧹 Cleaned

- Unused imports and dead variables removed.
- Redundant f-strings cleaned up.

### 🗑️ Removed

- `btn_dashboard.py` (floating Tkinter button).

### 📄 Documentation

- `README.md` updated for v1.0.
