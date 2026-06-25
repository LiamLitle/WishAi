# Changelog

All notable changes and improvements to WishAI.

Format inspired by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Times are US Eastern (ET).

[🇫🇷 Version française](FR/CHANGELOG.md)

---

## [1.5.2] — 2026-06-25 12h10.pm

Major update improving training reliability, logging, and internal architecture.

### New Features & Training

- **Training Recovery System**: The AI now robustly saves its state, allowing users to resume a training session exactly where it stopped (useful after a crash, manual interruption, or human error).
- **Advanced Logging & Temporary Storage**: Created a dedicated `TEMP/` subdirectory with precise timestamps (exact date, hours, seconds) for a granular training history without polluting the main directory.
- **Enriched `log_active.json`**: Added critical real-time system information:
  - AMP (Automatic Mixed Precision) and `torch.compile()` activation status.
  - Real model parameter count.
  - BPE and Tokenization status, including data download percentages and specific files/folders used.
- **Dynamic Model Detection**: Improved the dashboard's active model detection (no longer blindly relies on `active.json`).

### Interface & Visualization

- **Word Cloud User Experience (UX)**: The 4,000 token point cloud has been strictly flattened into 2D (Z-axis fixed to 0) and 3D rotation has been disabled, making the interface significantly more legible.

### Configuration Refactor & Bugfixes

- **Modular configuration architecture**: `scripts/config.py` has been refactored. Instead of one giant file, the codebase is now cleanly separated into a `scripts/Config/` directory (`donnees.py`, `modeles.py`, `systeme.py`, `utilitaires.py`).
- **"Shadowing" bug fix (`token.py`)**: The `scripts/token.py` script created an internal namespace conflict with standard Python packages, which randomly reset the tokenizer during model exports. It has been renamed to `scripts/reset_token.py`.
- **ONNX/GGUF export optimization**: Model exports are now neatly generated directly inside each model's directory (`model/<model_name>/`) along with an automatic copy of the `tokenizer.json` file, rather than prompting the user to copy the entire folder to the Desktop.
- **Export loading fix**: `export.py` was modified to correctly handle dynamic model loading from training checkpoints (weight dictionaries + `log_active.json` for config) instead of assuming the full PyTorch object was saved in the `.pt` file, fixing the `AttributeError: 'dict' object has no attribute 'eval'` crash.

---

## [1.5.1] — 2026-06-25 9h18.am

### Julia Analytic Engine — Advanced Convergence Predictions

- **`src/estimations.jl`** — new Julia background process running alongside the Python training loop. Performs two analyses Python cannot do efficiently:
  - **Chinchilla overfitting risk** — computes the parameter-to-token ratio and classifies risk as `faible` (low), `modéré` (moderate), `élevé` (high), or `critique` (critical) according to the Chinchilla scaling law (target ratio ≥ 20 tokens per parameter).
  - **Exponential plateau prediction** — fits a `L(s) = a + b·exp(-c·s)` curve via `LsqFit.jl` to estimate the asymptotic loss the model will converge to and how many steps remain before reaching it. Confidence level reported (`haute`, `bonne`, `faible`).
- **File-based IPC** — Julia reads `model/{name}/log_active.json` (written by Python) and outputs `model/{name}/insights.json`. The two processes never lock each other.
- **Automatic Julia installation** — `src/require.py` calls `check_and_install_julia()` at startup. On Windows, silently installs Julia via `winget install --id Julialang.Juliaup -e --silent` if not found. On other platforms, prints a manual install prompt. Packages `LsqFit` and `JSON` are installed/verified with `Pkg.add([...])`.
- **Graceful degradation** — if Julia is absent or packages are not yet installed, `go.py` skips the Julia process without crashing. The dashboard displays `"waiting for Julia..."` instead of an error.

### Adaptive Learning Rate (Cosine Decay)

- **`get_lr(iteration)`** in `nanogpt_bpe.py` — replaces the previous fixed learning rate with a two-phase schedule:
  - **Warmup** (first 100 steps): LR ramps linearly from `lr/100` to `learning_rate`.
  - **Cosine decay** (remaining steps): LR follows a cosine curve from `learning_rate` down to `learning_rate × 0.1`. This is the same schedule used by GPT-3, LLaMA, and Mistral.
- Current LR is logged to `log_active.json` as `lr_actuel` at every evaluation step.

### Best Model Checkpointing

- `nanogpt_bpe.py` now saves `best_model.pt` (alongside `checkpoint.pt`) whenever `val_loss` reaches a new all-time low. If training overshoots or overfits, the best weights are always preserved.

### Dashboard — Julia Metrics Integration

- **Learning Rate card** — new card in the main metrics row showing the current LR value with a label indicating whether it is in warmup phase or cosine decay.
- **Plateau prediction (Julia)** — the "Plateau estimé" cell in the Trends section now displays Julia's exponential curve fit result (`~2.87` target loss, `~800 steps remaining`, confidence level) instead of the previous linear JavaScript estimate.
- **Chinchilla Overfitting cell** — new cell in the Trends section showing the risk level (`FAIBLE` / `MODÉRÉ` / `ÉLEVÉ` / `CRITIQUE`) with color coding (green / yellow / red) and the raw ratio.
- **`fetchLog()`** updated to also fetch `insights.json` and merge it into the dashboard data object after each polling cycle.

### Tests

- **`tests/test_all.py`** — automated test suite covering all v1.5.1 features:
  - Julia binary detection
  - `LsqFit` + `JSON` package verification (auto-installs if missing)
  - Synthetic training curve → Julia generates `insights.json` → results validated
  - Cosine decay scheduler: warmup ascent, cosine descent, and final value assertions
  - **Auto-cleanup**: saves and restores any existing `active.json`; deletes all test artifacts after the run.

---

## [1.5.0] — 2026-06-24 18h13.pm


### Download bot — real sizes & custom sources

- **Real HuggingFace dataset sizes** — `obtenir_taille_hf()` queries the HuggingFace API (`load_dataset_builder`) for actual dataset sizes instead of relying on manual estimates. Results cached in `src/sizes_cache.json` for 30 days. Fetch is lazy (triggered when browsing a category, not at startup). Manual `taille_max_mo` values in `SOURCES` serve as fallback if the API is unavailable.
- **Size display in source menu** — each source now shows `Max: ~4 GB  Typical: 800 MB` alongside a size badge. If the requested amount exceeds the source's max, a warning is shown and the download is capped automatically.
- **Custom sources** (`src/custom_sources.json`) — add your own HuggingFace datasets without touching the Python code. A new `[ +]` menu option walks through the required fields (HF path, text field, language, max size). Custom sources appear in the menu with a `[CUSTOM]` badge and are merged into `SOURCES` at startup.
- **Language filter on HuggingFace datasets** — new `langue_cible` field on sources that mix multiple languages (e.g. `oasst2` which contains FR/EN/DE/ES/ZH). Each extracted text is run through `detecter_langue()` and discarded if it doesn't match the target language, keeping datasets clean.
- **Mandatory language selection** — `_demander_liste(defaut=None)` now refuses empty input, forcing an explicit language choice in both preset and custom bot modes. No more silent defaults.
- **`langdetect` integration** — optional library (~2 MB) imported at startup via `try/except`. When available it provides ~95% accuracy across 55 languages; if absent, falls back to the existing keyword-matching approach automatically.

### Manifest system — 50% disk space saved

- **`data/{lang}/manifest.json`** replaces `data.txt` copying. Instead of concatenating all source files into one large `data.txt` (doubling disk usage), a lightweight JSON manifest lists the active source files. The tokenizer and training engine read them directly in streaming. On a 950 MB dataset, disk usage drops from ~1.9 GB to ~950 MB.
- **`telecharger.py`** — `combiner_sources()` replaced by `maj_manifest()`, which writes the manifest without copying any file. `reload_requested.flag` is created after each successful download.
- **`tokenizer.py`** — reads via `manifest.json` first (streaming, up to 50 MB sample), falls back to `data.txt` if no manifest found.
- **`nanogpt_bpe.py`** — reads training data via manifest. Cache validity check extended to compare source file modification times against the BPE cache mtime.

### Hot-reload & checkpoints during training

- **Hot-reload** — after a new source is downloaded, `telecharger.py` creates `data/{lang}/reload_requested.flag`. At every checkpoint (`checkpoint_interval` steps), `nanogpt_bpe.py` detects this flag, removes it, re-tokenizes all sources (old + new combined), and resumes training from the exact same step. No restart required.
- **Checkpoint saving during training** — was entirely absent from the training loop despite `checkpoint_interval` being configured. Checkpoints are now saved every `checkpoint_interval` steps (not only at the very end of training), enabling crash recovery, hot-reload, and `go.py`'s auto-resume loop.

---

## [1.4.0] — 2026-06-24 9h22.am

### Auto-bot & data download

- **Auto-bot** (`[a]` in the library, or default on first launch) — intelligent download system that detects free disk space (cap: 30% of total disk, max 2 GB) and offers two modes:
  - **Preset mode** — Nano (~100 MB) / Small (~300 MB) / Medium (~700 MB) / Large (~1.5 GB)
  - **Custom mode** — choose number of parameters (20M → 1B+), recommended data size calculated automatically, adjustable MB cap, language (FR / EN / Multi), and AI type (General / Code / Science / Chat / Assistant). Sources selected intelligently per combination.
- **HuggingFace retry** — up to 4 attempts with exponential backoff (3s, 8s, 20s) on connection reset (`WinError 10054`). Mid-stream reconnect: reloads the dataset and skips already-written articles.
- **Library menu fix** — empty input or invalid choice no longer exits; loops until a valid selection is made.

### Logging & auto-repair

- **`src/bot_logger.py`** — logging system for the bot:
  - `system/logs/bot.log` — all events (INFO, WARNING, ERROR, FATAL)
  - `system/logs/erreurs.log` — errors only, for quick diagnosis
  - `system/logs/downloads.log` — structured JSON Lines per download (source, MB, success/fail, timestamp)
  - Auto-rotation at 2 MB, keeps 3 files
- **Auto-repair** — if a dependency (`datasets`, `torch`, `safetensors`…) is absent or corrupted (ImportError / OSError), it is automatically uninstalled and reinstalled via pip before retrying. Integrated directly into `telecharger_hf`.
- **`tests/test_sources.py`** — sends a ~1 MB request to every HuggingFace source and reports ✅ / ❌ + top 5 fastest sources.

### New commands

- **`./wish token`** — deletes `system/tokenizer.json` (and model-specific tokenizer files) then immediately retrains the BPE tokenizer on existing data. `./wish token reset` deletes only (retrain deferred to next `./wish go`).
- **`./wish repair`** — scans all critical packages and auto-reinstalls any that are missing or corrupted.
- **`./wish logs`** — shows the last 40 bot log entries. `./wish logs erreurs` for errors only. `./wish logs repair` to repair and log in one step.
- **`./wish chat --terminal`** — terminal generation mode (merged from deleted `src/generate.py`): lists available models, pick one, interactive loop (`t=`, `n=`, `q`).

### Tokenizer

- **Dynamic sample size** — tokenizer BPE training now uses 15% of the data file (min 5 MB, max 50 MB) instead of a fixed 5 MB. On a 252 MB dataset: ~38 MB used, giving a significantly more representative vocabulary.

---

## [1.3.3] — 2026-06-24 8h26.am

### Launcher system

- **`wish.bat`** — unified shortcut system at root. Single entry point for all commands:
  ```
  ./wish go        Main launcher
  ./wish chat      Chat interface
  ./wish quick     Zero-config training (~20M params)
  ./wish config    Model & data management
  ./wish serve     Dashboard / library server
  ./wish visual    Embedding visualizer (port 8080)
  ```
- **`scripts/config.py`** — new option **[16] Uninstall dependencies**: runs `pip uninstall -y` on all packages listed in `requirements.txt` without touching `deps.lock`.

---

## [1.3.2] — 2026-06-24 07:54.am

### Project structure

- **Root cleanup**: only 8 files remain at root (`go.py`, `dashboard.html`, `README.md`, `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `DATASETS.md`, `requirements.txt`).
- **`scripts/`** — launcher scripts moved here: `chat.py`, `quick.py`, `serve.py`, `config.py`.
- **`docs/`** — documentation moved here: `PARAMETRES.md`, `LAUNCH.md` (community launch guide).
- **`web/`** — `library.html` moved here.
- **`system/`** — `pyrightconfig.json` and `tokenizer.json` moved here. All path references updated in `src/`.
- **`wish.bat`** — single unified launcher at root. Replaces 5 individual `.bat` files.
  ```
  ./wish go | chat | quick | config | serve | visual
  ```

---

## [1.3.1] — 2026-06-23 20:07.pm

### Embedding visualizer improvements

- **Flat colors**: removed all neon/glow effects (`ctx.shadowBlur`, `ctx.shadowColor`). Colors are now solid and muted — words `#4db8cc`, numbers `#c8a830`, special characters `#c04060`.
- **Hover-only labels**: token names are no longer shown automatically when zoomed in. Labels now appear exclusively on hover, with a white ring + filled dot and the token name drawn above.
- **Smart filter system**: five filter buttons in the topbar — *All*, *Words*, *Numbers*, *Specials*, *End-of-word*. Tokens that don't match the active filter are dimmed to 8% opacity. Filter state is reactive and combines with the search bar.
- **End-of-word tokens** (`</w>`) rendered at 75% of base radius to visually distinguish them from full words.
- **Style**: `visual/style.css` updated with styles for `.filter-btn` and `.filter-btn.active`.

---

## [1.3.0] — 2026-06-23 19:34.pm

### Chat interface — complete redesign

- **Topbar removed**: the top navigation bar has been removed entirely. The interface is now fullscreen with no fixed header.
- **Pill input bar**: the input area is now a heavily rounded capsule (`border-radius: 32px`), centered vertically on screen at startup, then docked to the bottom on first send.
- **Integrated model selector**: the model dropdown is now embedded directly below the textarea (no separate panel). Models are sorted by descending size. The **Load** button becomes **✓ Cached** (green, disabled) when the active model is already in memory.
- **"More options" with fade-in**: Temperature and Length are hidden behind a discreet ⚙ button. The panel opens and closes with a smooth transition (`max-height` + `opacity` + `padding`) — no abrupt jump.
- **Floating sidebar**: conversation history opens via a floating ☰ button in the top-left. This button disappears when the sidebar is open. The sidebar is a rounded floating card with glassmorphism effect (`backdrop-filter: blur`), fully detached from the document flow.
- **Violet edge glow**: the interface background displays 4 elliptical violet blobs at screen corners via Canvas 2D radial gradients — the 3D point cloud has been removed.
- **Improved chat bubbles**:
  - Stronger rounding, entrance animation (`fadeIn` + `translateY(8px) → 0`)
  - Each AI response shows on hover an action row: **Copy** button, response time in milliseconds, number of tokens generated, and **↺ Regenerate** button
  - The Copy button temporarily shows "✓ Copied!" in green after clicking
- **Regenerate**: the ↺ button removes the last AI response and relaunches generation from the last user message, without reloading the page.
- **Auto-load on model change**: selecting a model in the dropdown triggers its loading immediately. If the model is already in memory (`_cachedModelName`), no reload is performed — client-side cache.
- **IndexedDB history**: conversations are persisted in IndexedDB (`WishAIChat` / `convs`), retrieved on startup, and accessible from the sidebar.

### Bug fixes

- **Architecture mismatch in `chat_server.py`**: the old `build_modele()` function used a GPT-2 architecture (learned position embedding, `tril`, `ReLU`, `LayerNorm`) incompatible with models trained via `model.py` (RoPE + RMSNorm + SwiGLU). Fixed by replacing `build_modele()` with a direct import of `WishAI_BPE` and `ConfigModele` from `src/model.py`.
- **Missing spaces in generated text**: token-by-token streaming used `tok.decoder()` which calls `.strip()` and removed end-of-word spaces (BPE `</w>` marker). Fixed with a dedicated `decode_token(id)` function that converts `</w>` → space without `.strip()`.
- **`KeyError: 'hyperparams'`**: compatibility between the old checkpoint format (`hyperparams` / `taille_vocab`) and the new format (`architecture` with `vocab_size` included). `_charger_modele()` detects the present key and adapts accordingly.

### Project structure

- **`system/` folder**: runtime JSON files (`control.json`, `session.json`, `monitor_port.json`, `chat_url.json`, `dashboard_url.json`) and `deps.lock` moved to `system/` to keep the root clean. All scripts updated to use the new path.
- **`FR/` folder**: French versions of all documentation files (`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `DATASETS.md`, `PARAMETRES.md`, `LICENSE.md`).

---

## [1.2.0] — 2026-06-21 18:28.pm

### Dashboard — complete UI

- **Animated idle screen**: when no training is running, the dashboard displays a waiting screen with floating particles, animated brain, moving grid and a pulse indicator. Auto-redirect `file://` → `localhost` (port stored in `localStorage`).
- **"Training complete" banner**: green bar that slides from the top with key statistics (final val loss, duration, steps) as soon as status changes to `done`.
- **4 collapsible sections** (state saved across reloads):
  - **📈 Trends & Convergence** — Δ val loss over 10 evals, trend (descending / plateau / overfitting), descent speed (delta/100 steps), estimated plateau
  - **⚡ Real performance** — tokens/s, MB of text processed, effective batch (batch × grad_accum), number of checkpoints created
  - **🧠 Model analysis** — params/layer, theoretical VRAM (~4 bytes/param), head size (n_embd ÷ n_head), learning state with tailored advice
  - **📋 Event log** — val_loss records, thermal/RAM pauses, convergence — each event timestamped by step
- **Last generated text**: displayed at the bottom of the page with "Show all / Collapse" button.
- **Session tracking**: `session.json` written by `go.py` at startup; dashboard monitors it via SSE and resets display automatically when a new session starts.
- **Hyperparameter table + architecture diagram**: fixed section below graphs showing all current model parameters and a visual architecture diagram.

### New

- **`quick.py`** — zero-config fast mode: generates demo text if no data exists (~200 FR/EN sentences), trains the BPE tokenizer, launches the dashboard and starts training with the MINI preset (~20M parameters). No questions asked — `Ctrl+C` to stop.
- **Chat interface** (`chatting/`, `chat.py`, `src/chat_server.py`) — talk to your trained model from a web interface. Persistent conversation history, model selector, hot-reload without restarting the server.
- **`serve.py`** — launches only the chat server without going through `go.py`.
- **`go.py` — optional library opening**: if data already exists, offers to open the library to add more before starting.
- **`go.py` — phase 3 resume loop**: if `monitor.py` triggers a critical stop, `go.py` waits for automatic resume and restarts training from checkpoint without manual intervention.

### Bug fixes

- `library.html` — duplicate code blocks removed in `loadLocalData()` and `checkServer()` (JS errors lines 334 and 559)
- `src/telecharger.py` — missing `import sys` while `sys.exit(1)` was used
- `pyrightconfig.json` created — Pylance points to `.venv312`; no more "import could not be resolved" warnings in VSCode
- `.gitignore` — `tests/` removed (bug: `tests/test_smoke.py` was never tracked); `.ruff_cache/`, `monitor_port.json`, `deps.lock` added

### Documentation

- `config.py` — full 16-command menu documented in `README.md` as a collapsible section (list, hyperparams, export, reset, logs…)

---

## [1.1.0] — 2026-06-21 18:20.pm

### Architecture

- **Modernized model (RoPE + RMSNorm + SwiGLU)**: `model.py` moves from a GPT-2 (2019) to a LLaMA/Mistral-style (2024) architecture.
  - `LayerNorm` replaced by `RMSNorm` (simpler, faster, no bias)
  - Learned positional embeddings removed, replaced by **RoPE** (Rotary Position Embedding) — better generalization on long sequences
  - FFN `Linear → ReLU → Linear` replaced by **SwiGLU** (`SiLU(gate) * up → down`) — better loss at equal parameter budget
  - All linear layers switched to no bias (`bias=False`)
  - `nn.Sequential` of blocks replaced by `nn.ModuleList` + loop (necessary to pass `cos`/`sin` RoPE to each block)
  - MINI parameter count unchanged: **20.8M** (removing `position_embedding` offsets the 3 SwiGLU projections vs 2)

### Bug fixes

- **Empty `data/data.txt` blocked the tokenizer**: `donnees_existent()` (in `quick.py` and `go.py`) and `trouver_data_file()` (in `tokenizer.py`) now return `False`/`None` if the file is smaller than 1 KB — avoids BPE training on empty text (vocab = 1 token, everything encoded as `</w>`)

### Improvements

- **Dynamic monitor port**: `monitor.py` looks for a free TCP port instead of hardcoding 8001. Port written to `system/monitor_port.json` so `dashboard.py` can read it. No more silent clash if the port is taken.
- **Virtualenv warning in `require.py`**: if Python runs outside a venv (and outside conda), the user sees a clear message and must confirm before any global installation.
- **Dashboard SSE**: `setInterval` polling replaced by **Server-Sent Events** — `dashboard.py` exposes `/api/events` pushing training logs, session changes, and monitor data in real time. Automatic polling fallback if SSE fails or when opened via `file://`. `dashboard.py` switches to `ThreadingHTTPServer` to handle multiple simultaneous connections.
- **Multi-model comparison in dashboard**: "Compare models" button opening a panel with all loss curves overlaid (endpoint `/api/models` returns all trained model histories).
- **Common Crawl quality filtering**: `nettoyer_texte()` in `telecharger.py` now does 4 passes: residual HTML removal, line-by-line filtering (< 40 chars, 2+ URLs, emails, punctuation > 15%), deduplication of repeated lines, non-latin cleanup.

### Documentation

- **`CONTRIBUTING.md`** created: project structure, workflow, code standards, sensitive zones, PR guide, contribution ideas.

---

## [1.0.0] — 2026-06-21 18:20.pm

First stable version. Code review, bug fixes and safety net (tests).

### Bug fixes

- **`nanogpt_bpe.py` — stray `shu` line**: a `shu` line was sitting at module level at the end of the file. It caused a `NameError` at runtime, just after model save. Removed.
- **`telecharger.py` — `supprimer_donnees()` undefined**: the menu called this function (option "s") but it didn't exist anywhere → `NameError` on click. Function implemented (delete one source, the combined `data.txt`, or everything, by language, with freed space shown).
- **Tokenizer not retrained on dataset change**: you had to manually delete `tokenizer.json`, otherwise the model trained on a vocabulary that no longer matched the data (silent bug). Now a **signature** of the dataset is stored in `tokenizer.json` and compared at each launch → automatic retraining if needed.
  - Sub-bug fixed: `vocab_size` comparison was causing infinite retraining loop on small datasets. Detection now relies solely on the signature.
  - Sub-bug fixed: default value frozen at module load (classic Python trap) — target vocab is now read at runtime.
- **`control.json` — non-atomic write**: in case of crash mid-write, the file could be corrupted and crash reading processes. Switched to atomic write (`.tmp` file + `os.replace`) in all **3** writers: `go.py`, `monitor.py`, `nanogpt_bpe.py`.
- **Dashboard — flash at startup**: fixed with a startup grace window and session freshness check (an old `session.json` no longer triggers a false "starting").

### Code cleanup

- Unused imports removed across `chat.py`, `nanogpt_bpe.py`, `dashboard.py`, `tokenizer.py`, `telecharger.py`, `require.py`.
- Dead variable `_head_size_rec` removed (`nanogpt_bpe.py`).
- Redundant `global _st` removed (`chat_server.py`).
- 28 f-strings without variables cleaned up across multiple files.

### Added

- **Apple Silicon (MPS) support**: `torch.backends.mps.is_available()` detection in `verifier_pc()`. WishAI no longer forces CPU on Mac.
- **Mixed precision (AMP)** in training loop: `autocast` bf16/fp16, with `GradScaler` in fp16, enabled only on CUDA.
- **`torch.compile`** enabled on Linux+CUDA only (unstable on Windows/MPS), with weights routed through a "raw" module for always-compatible checkpoints.
- **Smoke tests** (`tests/test_smoke.py`): tokenizer round-trip, dataset change detection, forward pass + loss, checkpoint save/load, generation, and compilation of all `.py` files.

### Changed

- **Model extracted to `src/model.py`**: parameterized by `ConfigModele`. Makes the model importable and testable without triggering training.

### Removed

- **`btn_dashboard.py` (floating Tkinter button)**: could crash on headless Linux. Status remains visible in the terminal and dashboard.

### Documentation

- `README.md` updated for v1.0 (version badge, MPS support, automatic tokenizer, tests section, updated file structure).
