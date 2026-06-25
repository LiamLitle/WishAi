<div align="right">

🇬🇧 English | [🇫🇷 Français](FR/README.md)

</div>

<div align="center">

# 🧠 WishAI

<!-- Stack & Compatibility -->
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

**nanoGPT is great. But you get no dashboard, no VRAM protection, and your PC can die in 4 minutes.**

**WishAI fixes that.**

**🎉 Version 1.5.2 — Training recovery system, advanced `TEMP/` logging, modular config architecture, and export bug fixes.**

*Built by Liam — learned from scratch.*

</div>

---

<!-- 📸 ADD YOUR GIF HERE — record the dashboard with ScreenToGif then uncomment: -->
<!-- ![WishAI Dashboard](assets/dashboard.gif) -->

---

> WishAI lets you train a real GPT on your local machine, from scratch, without complicated setup.
> Download data, run one command, and watch your AI learn in real time in a native dashboard.

---

<div align="center">
<table>
<tr>
<td align="center" width="16%">

**📊 Native Dashboard**<br>
Local window<br>
~60 MB RAM

</td>
<td align="center" width="16%">

**🛡️ Auto Protection**<br>
VRAM, RAM, temp<br>
never crashes

</td>
<td align="center" width="16%">

**📚 Dataset Library**<br>
135 curated +<br>
100k+ accessible

</td>
<td align="center" width="16%">

**🔋 Accumulation**<br>
Large models on<br>
small VRAM

</td>
<td align="center" width="16%">

**🔄 Early stopping**<br>
Stops automatically<br>
at convergence

</td>
<td align="center" width="16%">

**🧠 BPE from scratch**<br>
3× more context<br>
than char-level

</td>
</tr>
</table>
</div>

---

## Installation

> Requirements: **Python 3.8+**, **pip**, **git**

```bash
git clone https://github.com/LiamLitle/WishAi
cd WishAi
./wish go
```

`go.py` installs dependencies, checks the tokenizer, opens the dashboard, starts the monitor in the background and begins training — all in one command.

**Just want to test without configuring anything?**

```bash
./wish quick
```

Zero-config mode: downloads TinyStories, trains the tokenizer, opens the dashboard and starts a ~20M parameter model — no questions asked.

**GPU (optional but recommended):** If you have an NVIDIA card, install PyTorch with CUDA support from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) — pick your CUDA version in the selector. On **Apple Silicon (M1/M2/M3)** WishAI uses the **MPS** backend automatically. Without a GPU, it still runs on CPU (NANO preset recommended).

---

## Usage

### 1. Download data

```bash
./wish serve library
```

Or click **📚 Open Library** in the dashboard. The **Dataset Library** opens in your browser. Unified interface with 4 filterable sources:

- **📌 Our Selection** — **135 tested datasets** organized in 19 domains: Encyclopedias (29 languages), Web, Literature, Instructions, Code, Math, Science, Medicine, Dialogues, Translation, Law, Finance, Education, and more.
- **🤗 HuggingFace** — direct access to **150,000+ datasets** from the Hub. Real-time search with debounce.
- **🐙 GitHub** — search dataset repositories on GitHub (sorted by stars).
- **📄 Papers with Code** — academic datasets referenced in scientific publications (server proxy, no CORS).

Filter by source, language or domain in real time. Total accessible: **+100k datasets**.

Downloads run **in the background**: you can launch several at once and track their status in the interface without blocking anything.

👉 **[See the full list of available Datasets](DATASETS.md)**

You can also **add your own texts**: put any `.txt` file in `data/en/` or `data/fr/`.

---

### 2. Launch everything

```bash
./wish go
```

> The BPE tokenizer trains automatically on first run and retrains itself whenever your dataset changes — nothing to do manually.

The program detects your hardware and suggests a config:

| Preset | GPU required | Params | |
|--------|-------------|--------|-|
| 🐢 NANO | CPU or < 4 GB | ~2M | to get started |
| 🚀 SMALL | 4–6 GB | ~10M | good balance |
| ⚡ MEDIUM | 6–8 GB | ~40M | best quality/time ratio |
| 🧠 LARGE | 12+ GB | ~85M | for the patient |
| 🔧 CUSTOM | — | you choose | with explanations for each param |

Then you choose the duration:

```
Minutes [auto] >
```

- **Enter** → stops automatically at convergence
- **A number** → calculates steps, shows estimated end time

During training, follow progress live in the **dashboard** (it opens automatically) — current model, step, loss curves and system metrics, all in real time.

---

### 3. Talk to your AI

**Terminal mode:**

```bash
./wish chat --terminal
```

```
You > The future of artificial intelligence
AI  > The future of artificial intelligence is now being explored...

t=0.5 → predictable    t=1.5 → creative    n=200 → length    q → quit
```

**Chat interface (web UI):**

```bash
./wish chat
```

Opens a fully redesigned browser chat interface:

- Fullscreen layout with no top navigation bar — floating sidebar (☰) for conversation history
- Pill-shaped input bar centered on load, docks to the bottom on first message
- Model selector integrated directly into the input zone — sorted by size, auto-loads on selection, shows **✓ Cached** when the model is already in memory (no unnecessary reload)
- Temperature and max length hidden behind **⚙ More options** with a smooth fade animation
- Each AI response shows, on hover: a **Copy** button, response time in ms, token count, and a **↺ Regenerate** button
- Conversation history persisted in IndexedDB, restored on next open
- Violet edge glow background (Canvas 2D radial gradients)

Or use `./wish serve` to open the dashboard without launching training.

---

### 4. Run the tests *(optional)*

A small safety net to confirm nothing is broken after editing the code:

```bash
python -m unittest discover -s tests
```

`OK` means all good. Covers the tokenizer round-trip, dataset-change detection, model forward / save-load / generation, and a compile check on every Python file.

---

### 5. Manage your models *(optional)*

```bash
./wish config
```

<details>
<summary><b>⚙️ Full config.py menu</b></summary>
<br>

```
  ╔══════════════════════════════════════════════╗
  ║           WishAI  —  Configuration          ║
  ╚══════════════════════════════════════════════╝

  ── MODELS ──────────────────────────────────────
  [ 1]  List models
  [ 2]  Delete a model
  [ 3]  Delete ALL models
  [ 4]  View hyperparameters
  [ 5]  Rename a model
  [ 6]  Duplicate a model
  [ 7]  Export a model

  ── DATA ────────────────────────────────────────
  [ 8]  View available data
  [ 9]  Delete demo data
  [10]  Delete BPE cache
  [11]  Regenerate tokenizer

  ── SYSTEM ──────────────────────────────────────
  [12]  PC / GPU info
  [13]  Test PyTorch + GPU
  [14]  Last training logs
  [15]  Reset dependencies (delete deps.lock)
  [16]  Uninstall dependencies (pip uninstall)
  [17]  Full reset (erase everything)

  [ 0]  Quit
```

Each option is interactive — it asks for confirmation before deleting anything.

**Most useful options:**
- **[1]** — see all your trained models with their val loss, size, date
- **[4]** — inspect the architecture and hyperparameters of any model
- **[12]** — check your GPU, VRAM, RAM, Python and PyTorch versions
- **[13]** — run a quick matrix multiplication benchmark on GPU
- **[14]** — view the last 5 evaluation steps (train loss / val loss) in the terminal

</details>

---

## The Dashboard

The dashboard opens automatically when you launch `go.py`.

It displays in real time via **Server-Sent Events** (no polling):

- Used / total RAM, GPU VRAM, temperature, CPU
- `train_loss` and `val_loss` curves
- Current step, training speed, active protection level

**When no training is running:** animated idle screen with floating particles and a glowing brain — auto-redirects from `file://` to `localhost` if the server is found.

**When training finishes:** a green "Training complete" banner slides in with final stats.

**4 collapsible sections** with state saved across reloads:

| Section | What it shows |
|---------|--------------|
| 📈 Trends & Convergence | Δ val loss, trend direction, descent speed, estimated plateau step |
| ⚡ Real performance | Tokens/s, MB processed, effective batch, checkpoint count |
| 🧠 Model analysis | Params/layer, theoretical VRAM, head size, learning state + advice |
| 📋 Event log | val_loss records, thermal/RAM pauses, convergence alerts |

The **📊 Compare models** button overlays the loss curves of all trained models on a single chart.

The **📚 Open Library** button in the dashboard opens `library.html` — the full dataset library with background downloads.

> `monitor.py` runs silently. The looping terminal display is disabled to avoid conflicting with training logs — everything is visible in the dashboard.

---

## Interpreting your results

| Val Loss | Perplexity | What it means |
|----------|-----------|---------------|
| > 5.0 | > 148 | AI is learning the basics |
| 3.0 – 5.0 | 20 – 148 | Making progress |
| 2.0 – 3.0 | 7 – 20 | Text starts to be coherent |
| < 2.0 | < 7 | Very good — GPT-2 small (117M) sits around 3.1 |

> If `val_loss` rises while `train_loss` drops: **overfitting** — the AI is memorizing instead of understanding. Fix: increase `dropout` or add more data.

👉 **[Full parameter guide](docs/PARAMETRES.md)**

---

<details>
<summary><b>📊 Comparison with alternatives</b></summary>
<br>

| Feature | 🧠 WishAI | nanoGPT | nanochat | LitGPT | GPT-NeoX | Axolotl | DeepSpeed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Goal** | Learning + UI | Educational | Edu / Full-stack | Engineering | Industrial scale | LoRA fine-tuning | Distributed Multi-GPU |
| **Real-time dashboard** | ✅ Local | ❌ Terminal | ⚠️ Basic UI | ⚠️ Paid cloud | ❌ | ❌ External W&B | ❌ External W&B |
| **Dataset library** | ✅ 135 curated + 100k+ (HF/GitHub/PwC) | ❌ | ❌ | ❌ | ❌ | ⚠️ Manual | ❌ |
| **VRAM & OOM protection** | ✅ Auto + Accumulation | ❌ Crash | ❌ Crash | ✅ CLI | ❌ | ⚠️ Manual | ⚠️ Manual |
| **Background downloads** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Early stopping** | ✅ Auto | ❌ Fixed time | ❌ Fixed time | ❌ Manual | ❌ | ⚠️ YAML config | ❌ |
| **Required level** | Beginners | Developers | Developers | ML Engineers | Research labs | ML practitioners | Researchers |
| **Multi-GPU / Cluster** | ❌ Single GPU only | ✅ Basic (DDP) | ❌ Single node | ✅ FSDP/DDP | ✅ Megatron | ✅ FSDP | ✅ Native |
| **LoRA / Fine-tuning** | ❌ Pretrain only | ⚠️ Basic | ✅ Yes | ✅ Yes | ❌ | ✅ QLoRA | ✅ Yes |
| **Production API / Serving** | ❌ Local use only | ❌ | ⚠️ Minimal API | ✅ vLLM/LitServe | ❌ | ❌ | ✅ Yes |
| **Quantization (4/8-bit)** | ❌ FP16/BF16 only | ❌ FP16/BF16 | ❌ FP16/BF16 | ✅ Yes | ❌ | ✅ Yes | ✅ Yes |
| **Model Export (GGUF/ONNX)** | ❌ PyTorch only | ⚠️ Custom scripts | ❌ | ✅ Yes | ❌ | ✅ Yes | ❌ |
| **Pre-trained Weights** | ❌ Trains from scratch | ✅ GPT-2 | ❌ | ✅ Many | ✅ EleutherAI | ✅ All HuggingFace | ⚠️ Framework dependent |
| **Custom Architectures** | ❌ Fixed (LLaMA-style) | ✅ Hackable code | ✅ Hackable code | ✅ Modular | ✅ Modular | ❌ YAML Config | ✅ Flexible |

**[nanoGPT](https://github.com/karpathy/nanoGPT)** — great for understanding Transformer math. No interface, no VRAM protection or gradient accumulation by default, no early stopping. **WishAI** is heavily inspired by its core engine.

**[nanochat](https://github.com/karpathy/nanochat)** — "the best ChatGPT that $100 can buy" (by Andrej Karpathy, 2025/2026). Covers the full pipeline (pretraining, finetuning, UI). Great for a complete minimal stack, but lacks automatic hardware protection and automatic datasets compared to WishAI.

**LitGPT** — cutting-edge optimizations, CLI-focused. For a dashboard you need their paid cloud.

**GPT-NeoX** — built for 64 GPUs in parallel. Unusable on a solo machine.

**Axolotl** — fine-tuning tool (LoRA/QLoRA) on existing LLMs. Not for building a GPT from scratch.

**DeepSpeed** — very large scale distributed training. Complex JSON configs, multi-GPU clusters required.

</details>

---

<details>
<summary><b>🛡️ Automatic protections — the feature nobody else has</b></summary>
<br>

Your PC cannot die during training. At each launch, you choose a protection level from four — the level is saved in `config.json` and reused automatically.

**4 available levels**

| Level | For whom | RAM alert | RAM pause | Critical RAM / °C |
|-------|---------|-----------|-----------|-------------------|
| **Minim** | Powerful machine (> 32 GB) | 85% | 90% | 95% / 90°C |
| **Standard** ← default | 16–32 GB | 75% | 82% | 92% / 90°C |
| **Protection** | Average PC or laptop (8–16 GB) | 70% | 78% | 85% / 90°C |
| **Max** | Old or very limited PC (< 8 GB) | 60% | 70% | 80% / 89°C |

**3 phases per level — training NEVER stops permanently**

| Phase | Trigger | What happens |
|-------|---------|--------------|
| **1 — Alert** | RAM exceeds alert threshold | Console message + automatic slowdown between iterations |
| **2 — Pause** | RAM exceeds pause threshold | Training pauses and waits in memory. `monitor.py` watches and sends resume signal when RAM drops |
| **3 — Critical** | RAM or temperature exceeds critical threshold | Checkpoint saved, clean stop. `monitor.py` monitors conditions. `go.py` automatically restarts from checkpoint when conditions are met |

**Other always-active protections**

| Situation | What happens |
|-----------|-------------|
| VRAM > 85% | Clean stop + save |
| Ctrl+C | Clean stop + save |
| PC shuts down | Checkpoint every N steps — resumes on next launch |

To change level: delete `system/config.json` and relaunch `go.py`.

</details>

---

<details>
<summary><b>🔬 Transformer Architecture</b></summary>
<br>

```
[Input Tokens]
       ↓
 Token Embedding  (+ RoPE applied inside attention)
       ↓
┌──────────────────────────────────────┐
│  × N layers (4 to 16 by preset)      │
│                                      │
│  RMSNorm → Multi-Head Attention      │  ← RoPE rotates Q and K by position
│          + residual connection       │
│                                      │
│  RMSNorm → SwiGLU Feed-Forward (8/3×)│  ← SiLU(gate) × up → down
│          + residual connection       │
└──────────────────────────────────────┘
       ↓
  RMSNorm → Linear → Softmax → Predicted token
```

LLaMA/Mistral-style architecture (RoPE + RMSNorm + SwiGLU). Everything is commented line by line in the code.

</details>

---

<details>
<summary><b>🗂️ File structure</b></summary>
<br>

```
wishai/
├── go.py               ← main launcher ← START HERE
├── wish.bat            ← shortcuts: ./wish go | chat | quick | config | serve | visual
├── dashboard.html      ← dashboard (real-time metrics)
├── requirements.txt
├── DATASETS.md         ← full list of available datasets
├── CONTRIBUTING.md
├── scripts/            ← secondary launchers
│   ├── chat.py         ← ./wish chat
│   ├── quick.py        ← ./wish quick  (zero-config mode)
│   ├── serve.py        ← ./wish serve  (dashboard/library without training)
│   └── config.py       ← ./wish config (model & data management)
├── docs/               ← documentation
│   ├── PARAMETRES.md   ← expert guide to training parameters
│   └── LAUNCH.md       ← community launch guide
├── web/
│   └── library.html    ← dataset library (./wish serve library)
├── FR/                 ← French versions of all documentation
│   ├── README.md, CHANGELOG.md, CONTRIBUTING.md
│   ├── DATASETS.md, PARAMETRES.md, LICENSE.md
├── system/             ← runtime files (auto-generated, do not edit)
│   ├── control.json, session.json, tokenizer.json …
├── chatting/           ← web chat interface
│   ├── index.html, style.css, app.js
├── src/                ← all Python core scripts
│   ├── nanogpt_bpe.py  ← model + training (core of the project)
│   ├── tokenizer.py    ← BPE tokenizer from scratch
│   ├── chat_server.py  ← web + terminal generation server
│   ├── telecharger.py  ← dataset downloader
│   ├── require.py      ← automatic dependency installation
│   ├── protection.py   ← VRAM/RAM/temp thresholds
│   ├── dashboard.py    ← HTTP server (dashboard + API + SSE)
│   ├── monitor.py      ← system metrics server + watchdog
│   ├── chat_server.py  ← chat HTTP server (SSE token stream)
│   └── model.py        ← Transformer (RoPE + RMSNorm + SwiGLU)
├── tests/
│   └── test_smoke.py   ← python -m unittest discover -s tests
├── assets/             ← screenshots / GIFs for the README
├── data/               ← your training data (.txt files)
├── visual/             ← embedding visualizer (./wish visual)
└── model/
    └── <name>/
        ├── modele.pt, modele.safetensors, checkpoint.pt
        ├── log_active.json, tokenizer.json
```

</details>

---

<details>
<summary><b>⚙️ Internal architecture — how components communicate</b></summary>
<br>

When you run `python go.py`, three processes start:

```
go.py (orchestrator)
  ├── monitor.py        → port 8001  (real-time system metrics)
  ├── dashboard.py      → auto port  (serves dashboard.html + library.html + REST API)
  └── nanogpt_bpe.py   → terminal   (the training itself)
```

**`dashboard.py` — full HTTP server**

| Route | Method | Description |
|-------|--------|-------------|
| `/dashboard.html` | GET | Monitoring interface |
| `/library.html` | GET | Dataset library |
| `/api/ping` | GET | Checks server is online |
| `/api/events` | GET | SSE stream — training logs + session + system metrics |
| `/api/models` | GET | All model histories for multi-model comparison |
| `/api/downloads` | GET | Status of all ongoing downloads |
| `/api/download` | POST | Starts a background download |

**`control.json` — inter-process communication**

```json
{"commande": "pause", "raison": "RAM 82%", "timestamp": 1718700000.0}
```

`nanogpt_bpe.py` reads this file at every iteration. `monitor.py` writes it when resume conditions are met. `go.py` reads it after each run to decide whether to restart or not.

</details>

---

<details>
<summary><b>❓ FAQ</b></summary>
<br>

**Training stops on its own, is it broken?**
No. In auto mode, it stops when val_loss hasn't moved for 5 evaluations. That's convergence.

**I want to resume a stopped training.**
Relaunch `python go.py` with the same model name. The checkpoint is detected automatically.

**The generated text is gibberish.**
That's normal at first. With val_loss > 4, the AI is still learning basic structures. Let it run.

**Can I add my own data?**
Yes. Any UTF-8 `.txt` file in `data/en/` or `data/fr/`. One sentence per line is fine, not required.

**I want to change the protection level.**
Delete `system/config.json` and relaunch `go.py`. The menu appears again.

**Do I need to retrain the tokenizer when I change my data?**
No. WishAI detects the change automatically (via a signature stored in `tokenizer.json`) and retrains it on the next `go.py` run.

**I want to download a HuggingFace dataset not in the list.**
Open the library (button in the dashboard or `python src/telecharger.py`), **HuggingFace Search** tab, type a keyword. You have access to all 150,000+ Hub datasets live.

**Can I sell the model I trained?**
Yes. The model is entirely yours. The license only applies to the code.

</details>

---

## License

**WishAI Personal Use License v1.0**

✅ Free to use — personal, educational, research
✅ Modification and sharing allowed (with attribution)
✅ Models you train are yours — do whatever you want with them
❌ Cannot sell this software without written permission
❌ Cannot claim authorship of this project

See [LICENSE](LICENSE) for the full terms.

---

<div align="center">

Built by Liam — learned from scratch.

</div>
