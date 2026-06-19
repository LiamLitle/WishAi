<div align="center">

# 🧠 WishAI

<!-- Stack & Compatibility -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20CPU-76b900?logo=nvidia&logoColor=white)](https://pytorch.org/get-started/locally/)
![Cross-Platform](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-0078D4)

<br>

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-API_Ready-FFD21E)](https://huggingface.co/)
[![Datasets](https://img.shields.io/badge/Datasets-135_curated_%2B_100k%2B-blue)](DATASETS.md)
[![Safetensors](https://img.shields.io/badge/Save-Safetensors-green)](https://huggingface.co/docs/safetensors/index)
[![Dashboard](https://img.shields.io/badge/UI-HTML5_%7C_Vanilla_JS-E34F26?logo=html5&logoColor=white)](/)

<br>

[![License](https://img.shields.io/badge/License-Non--Commercial-red)](LICENSE)

<br>

**nanoGPT is great. But you get no dashboard, no VRAM protection, and your PC can die in 4 minutes.**

**WishAI fixes that.**

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
python go.py
```

`go.py` installs dependencies, checks the tokenizer, opens the dashboard, starts the monitor in the background and begins training — all in one command.

**GPU (optional but recommended):** If you have an NVIDIA card, install PyTorch with CUDA support from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) — pick your CUDA version in the selector. Without a GPU, WishAI still runs on CPU (NANO preset recommended).

---

## Usage

### 1. Download data

```bash
python src/telecharger.py
```

The **Dataset Library** opens in your browser. Unified interface with 4 filterable sources:

- **📌 Our Selection** — **135 tested datasets** organized in 19 domains: Encyclopedias (29 languages), Web, Literature, Instructions, Code, Math, Science, Medicine, Dialogues, Translation, Law, Finance, Education, and more.
- **🤗 HuggingFace** — direct access to **150,000+ datasets** from the Hub. Real-time search with debounce.
- **🐙 GitHub** — search dataset repositories on GitHub (sorted by stars).
- **📄 Papers with Code** — academic datasets referenced in scientific publications (server proxy, no CORS).

Filter by source, language or domain in real time. Total accessible: **+100k datasets**.

Downloads run **in the background**: you can launch several at once and track their status in the interface without blocking anything.

👉 **[See the full list of available Datasets](DATASETS.md)**

You can also **add your own texts**: put any `.txt` file in `data/en/` or `data/fr/`.

---

### 2. Train the tokenizer *(once only)*

```bash
python src/tokenizer.py
```

~5–10 minutes. A progress bar is shown during encoding:

```
[████████████░░░░░░░] 62.5%  (9.3/15.0M words)
```

Result: `tokenizer.json`.

> With char-level, 256 tokens ≈ 50 words. With BPE, 256 tokens ≈ **180 words**. Same model, 3× more context.

---

### 3. Launch everything

```bash
python go.py
```

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

During training, a **floating button** appears on your desktop. It shows the current model and step, and opens the dashboard with one click. It disappears automatically when training is done.

---

### 4. Talk to your AI

```bash
python src/generate.py
```

```
You > The future of artificial intelligence
AI  > The future of artificial intelligence is now being explored...

t=0.5 → predictable    t=1.5 → creative    n=200 → length    q → quit
```

---

## The Dashboard

The dashboard opens automatically when you launch `go.py`. You can also access it via the floating button during training.

It displays in real time (via `monitor.py` on port 8001):

- Used / total RAM, GPU VRAM, temperature, CPU
- `train_loss` and `val_loss` curves
- Current step, training speed
- Active protection level

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

👉 **[Full parameter guide](PARAMETRES.md)**

---

<details>
<summary><b>📊 Comparison with alternatives</b></summary>
<br>

| Feature | 🧠 WishAI | nanoGPT / minGPT | LitGPT | GPT-NeoX | Axolotl | DeepSpeed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Goal** | Learning + UI | Educational | Engineering | Industrial scale | LoRA fine-tuning | Multi-GPU distributed |
| **Real-time dashboard** | ✅ Local | ❌ Terminal | ⚠️ Paid cloud | ❌ | ❌ External W&B | ❌ External W&B |
| **Dataset library** | ✅ 135 curated + 100k+ (HF/GitHub/PwC) | ❌ | ❌ | ❌ | ⚠️ Manual | ❌ |
| **VRAM & OOM protection** | ✅ Auto + Accumulation | ❌ Crash | ✅ CLI | ❌ | ⚠️ Manual | ⚠️ Manual |
| **Background downloads** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Early stopping** | ✅ Auto | ❌ Fixed time | ❌ Manual | ❌ | ⚠️ YAML config | ❌ |
| **Required level** | Beginners | Developers | ML Engineers | Research labs | ML practitioners | Researchers |

**nanoGPT / minGPT** — great for understanding Transformer math. No interface, no VRAM protection or gradient accumulation by default, no early stopping.

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

To change level: delete `config.json` at the project root and relaunch `go.py`.

</details>

---

<details>
<summary><b>🔬 Transformer Architecture</b></summary>
<br>

```
[Input Tokens]
       ↓
 Token Embedding + Position Embedding
       ↓
┌──────────────────────────────────┐
│  × N layers (4 to 16 by preset)  │
│                                  │
│  LayerNorm → Multi-Head Attention │  ← each token looks at others
│           + residual connection   │
│                                  │
│  LayerNorm → Feed-Forward (×4)   │  ← local reasoning
│           + residual connection   │
└──────────────────────────────────┘
       ↓
  LayerNorm → Linear → Softmax → Predicted token
```

Same architecture as GPT-2, just smaller. Everything is commented line by line in the code.

</details>

---

<details>
<summary><b>🗂️ File structure</b></summary>
<br>

```
wishai/
├── go.py               ← launches everything in one command ← START HERE
├── dashboard.html      ← dashboard interface (real-time metrics)
├── library.html        ← dataset library (downloads)
├── config.json         ← chosen protection level (created on first launch)
├── control.json        ← communication between go.py / nanogpt_bpe / monitor
├── DATASETS.md         ← full list of available datasets
├── PARAMETRES.md       ← expert guide to training parameters
├── src/                ← all Python scripts
│   ├── nanogpt_bpe.py  ← model + training (core of the project)
│   ├── tokenizer.py    ← BPE tokenizer from scratch (with progress bar)
│   ├── generate.py     ← interactive generation
│   ├── telecharger.py  ← data download (CLI + web interface)
│   ├── require.py      ← automatic dependency installation
│   ├── protection.py   ← thresholds for the 4 protection levels
│   ├── dashboard.py    ← local HTTP server (dashboard + library + REST API)
│   ├── monitor.py      ← HTTP metrics server (port 8001) + watchdog
│   └── btn_dashboard.py← floating button — appears during training
├── assets/             ← screenshots / GIFs for the README
├── cache/              ← Python cache (__pycache__) — local to project
├── data/               ← your training data (raw/cleaned text)
├── tokenizer.json      ← trained tokenizer (generated by src/tokenizer.py)
└── model/
    └── <name>/         ← one folder per model (created automatically)
        ├── modele.pt           ← final model for generate.py
        ├── modele.safetensors  ← final model for export (pure weights)
        ├── checkpoint.pt       ← resumable checkpoint
        ├── log_active.json     ← real-time dashboard data
        └── tokenizer.json      ← tokenizer used for this model
```

</details>

---

<details>
<summary><b>⚙️ Internal architecture — how components communicate</b></summary>
<br>

When you run `python go.py`, four processes start:

```
go.py (orchestrator)
  ├── monitor.py        → port 8001  (real-time system metrics)
  ├── dashboard.py      → auto port  (serves dashboard.html + library.html + REST API)
  ├── btn_dashboard.py  → tkinter    (floating button, watches log_active.json)
  └── nanogpt_bpe.py   → terminal   (the training itself)
```

**`dashboard.py` — full HTTP server**

| Route | Method | Description |
|-------|--------|-------------|
| `/dashboard.html` | GET | Monitoring interface |
| `/library.html` | GET | Dataset library |
| `/api/ping` | GET | Checks server is online |
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
Delete `config.json` at the project root and relaunch `go.py`. The menu appears again.

**The floating button doesn't appear.**
It only shows when training is detected (reads `model/*/log_active.json`). If it never appears, check that `tkinter` is available: `python -c "import tkinter"`.

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
