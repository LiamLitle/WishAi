# Contributing to WishAI

Thank you for wanting to improve WishAI! This guide explains how to contribute effectively.

[🇫🇷 Version française](FR/CONTRIBUTING.md)

---

## Project structure

```
AI/
├── go.py               # Main entry point (interactive menu)
├── quick.py            # Fast mode: 20M params, zero questions
├── config.py           # Model / data / system management
├── CONTRIBUTING.md     # This file
│
├── src/
│   ├── model.py        # Model architecture (RoPE + RMSNorm + SwiGLU)
│   ├── nanogpt_bpe.py  # Main training loop
│   ├── tokenizer.py    # BPE tokenizer from scratch
│   ├── telecharger.py  # Data download and cleaning
│   ├── dashboard.py    # HTTP server + API + SSE
│   ├── monitor.py      # Memory / GPU monitoring (dynamic port)
│   ├── chat_server.py  # Web + terminal generation server
│   ├── require.py      # Dependency management with lock file
│   └── protection.py   # Memory / temperature safety thresholds
│
├── chatting/           # Web chat interface
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── system/             # Runtime config files (auto-generated)
├── data/               # Training data (.txt)
├── model/              # Saved models (modele.pt + .safetensors)
├── dashboard.html      # Monitoring interface
└── library.html        # Dataset library
```

---

## Before coding

1. **Fork** the repository and clone your fork locally.
2. Create a **virtualenv**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # Linux / Mac
   ```
3. Install dependencies:
   ```bash
   python src/require.py
   ```
4. Create a dedicated **branch**:
   ```bash
   git checkout -b fix/bug-name
   # or
   git checkout -b feat/feature-name
   ```

---

## Code standards

- **Language**: comments and variable names in **French** (project convention). Critical error messages can be in English.
- **Encoding**: UTF-8 everywhere. Avoid emojis in f-strings and Python dicts (risk of truncation by some editors).
- **Imports**: no external dependencies not listed in `require.py`. If you add one, update it.
- **Style**: no imposed linter, but stay consistent with existing code (4 spaces, no trailing whitespace).
- **No intermediate checkpoint**: the project saves only a final model (`modele.pt` + `modele.safetensors`). Don't reintroduce periodic saving without discussion.

---

## Sensitive zones

| File | Warning |
|---|---|
| `model.py` | Any architecture change **breaks compatibility** with existing models. Document clearly. |
| `tokenizer.py` | BPE must remain reproducible (same seed → same vocab). |
| `nanogpt_bpe.py` | Unicode sections (em dash, emojis) can truncate the file with some tools. Use bash/heredoc to edit them. |
| `monitor.py` | Port is now dynamic (`system/monitor_port.json`). Do not hardcode a port. |
| `require.py` | Venv detection and lock file are intentional. |
| `system/` | Files in this folder are auto-generated at runtime. Do not edit them manually. |

---

## Making a Pull Request

1. Make sure syntax is valid:
   ```bash
   python -c "import ast; ast.parse(open('src/model.py').read()); print('OK')"
   ```
2. Run a quick training to verify nothing breaks:
   ```bash
   python quick.py
   ```
3. Describe in the PR:
   - **What**: what you're changing
   - **Why**: the problem it solves
   - **How to test**: steps to verify it works

---

## Welcome contribution ideas

- New GPU presets (RTX 4090, A100, Mac M3...)
- Multi-GPU support (`torch.nn.DataParallel`)
- HuggingFace export (config.json + HF tokenizer)
- Streaming output in terminal mode (`./wish chat --terminal`)
- Unit tests (still few of them)
- Comment / doc translation

---

## Questions?

Open an **issue** before starting a large piece of work. It avoids duplicates and allows alignment on the project direction.
