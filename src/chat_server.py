# chat_server.py -- serveur de chat WishAI
# Lance via : python chat.py  (depuis la racine)

import os, sys, json, socket, threading, time, glob, webbrowser
import torch
import torch.nn as nn
from torch.nn import functional as F
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from model import WishAI_BPE, ConfigModele

# ── Etat global ────────────────────────────────────────────────────
_st = {
    "model":        None,
    "encoder":      None,
    "decoder":      None,
    "decode_token": None,   # décode 1 token sans strip (pour le streaming)
    "hp":           None,
    "name":         None,
    "type":         None,
    "loading":      False,
    "error":        None,
}

def _charger_modele(nom):
    _st["loading"] = True
    _st["error"]   = None
    try:
        pt = os.path.join(ROOT, "model", nom, "modele.pt")
        if not os.path.exists(pt):
            raise FileNotFoundError(f"modele.pt introuvable pour '{nom}'")

        ckpt = torch.load(pt, map_location=device, weights_only=False)

        if "char_vers_int" in ckpt:
            # Modele char-level
            c2i = ckpt["char_vers_int"]
            i2c = ckpt["int_vers_char"]
            enc          = lambda t: [c2i.get(c, 0) for c in t]
            dec          = lambda ids: ''.join([i2c.get(i, '?') for i in ids])
            decode_token = lambda id_tok: i2c.get(id_tok, '?')
            _type = "CHAR"
        else:
            # Modele BPE
            from tokenizer import TokenizerBPE
            tok_path = os.path.join(ROOT, "model", nom, "tokenizer.json")
            if not os.path.exists(tok_path):
                tok_path = os.path.join(ROOT, "system", "tokenizer.json")
            tok = TokenizerBPE()
            tok.charger(tok_path)
            enc = tok.encoder
            dec = tok.decoder
            # Décodeur de streaming : </w> → espace, SANS strip()
            _i2t = tok.id_vers_token
            def decode_token(id_tok, _m=_i2t):
                t = _m.get(id_tok, '')
                if t.endswith('</w>'):
                    return t[:-4] + ' '
                return t
            _type = "BPE"

        # Compatibilite ancien format (hyperparams/taille_vocab)
        # et nouveau format (architecture avec vocab_size dedans)
        if "hyperparams" in ckpt:
            hp = ckpt["hyperparams"]
            tv = ckpt["taille_vocab"]
        else:
            arch = ckpt["architecture"]
            hp = {
                "n_embd"    : arch["n_embd"],
                "n_head"    : arch["n_head"],
                "n_layer"   : arch["n_layer"],
                "block_size": arch["block_size"],
                "dropout"   : arch.get("dropout", 0.0),
            }
            tv = arch["vocab_size"]

        cfg  = ConfigModele(
            vocab_size = tv,
            n_embd     = hp["n_embd"],
            n_head     = hp["n_head"],
            n_layer    = hp["n_layer"],
            block_size = hp["block_size"],
            dropout    = hp.get("dropout", 0.0),
        )
        mdl  = WishAI_BPE(cfg).to(device)
        mdl.load_state_dict(ckpt["modele_state"])
        mdl.eval()

        _st.update({
            "model":        mdl,
            "encoder":      enc,
            "decoder":      dec,
            "decode_token": decode_token,
            "hp":           hp,
            "name":         nom,
            "type":         _type,
            "loading":      False,
            "error":        None,
        })
        print(f"  Modele charge : {nom}  ({_type})  {sum(p.numel() for p in mdl.parameters()):,} params")

    except Exception as e:
        _st["loading"] = False
        _st["error"]   = str(e)
        print(f"  Erreur chargement : {e}")

def _lister_modeles():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "model", "*", "modele.pt"))):
        nom  = os.path.basename(os.path.dirname(path))
        size = round(os.path.getsize(path) / 1e6, 1)
        out.append({"name": nom, "size_mb": size})
    return out

# ── Handler HTTP ───────────────────────────────────────────────────
class ChatHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path.rstrip("/")
        if p == "/api/models":
            self._json(_lister_modeles())
        elif p == "/api/status":
            mdl = _st["model"]
            self._json({
                "loaded":  mdl is not None,
                "loading": _st["loading"],
                "name":    _st["name"],
                "type":    _st["type"],
                "error":   _st["error"],
                "params":  sum(x.numel() for x in mdl.parameters()) if mdl else 0,
                "device":  device,
            })
        elif p in ("", "/"):
            self.send_response(302)
            self.send_header("Location", "/chatting/index.html")
            self.end_headers()
        else:
            super().do_GET()

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        p       = urlparse(self.path).path.rstrip("/")

        if p == "/api/load":
            nom = payload.get("name", "").strip()
            if not nom:
                self._json({"error": "name manquant"}, 400); return
            if _st["loading"]:
                self._json({"status": "already_loading"}); return
            threading.Thread(target=_charger_modele, args=(nom,), daemon=True).start()
            self._json({"status": "loading", "name": nom})

        elif p == "/api/chat":
            if not _st["model"]:
                self._json({"error": "Aucun modele charge"}, 400); return
            prompt  = payload.get("prompt", "")
            temp    = max(0.1, min(2.0, float(payload.get("temperature", 0.8))))
            max_tok = max(10,  min(500,  int(payload.get("max_tokens", 200))))
            self._stream(prompt, temp, max_tok)
        else:
            self._json({"error": "route inconnue"}, 404)

    def _stream(self, prompt, temperature, max_tokens):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def emit(data):
            try:
                self.wfile.write(("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except Exception: pass

        try:
            mdl          = _st["model"]
            enc          = _st["encoder"]
            decode_token = _st["decode_token"]
            bs           = _st["hp"]["block_size"]

            ids = enc(prompt) if prompt else [0]
            idx = torch.tensor([ids], dtype=torch.long, device=device)

            with torch.no_grad():
                for _ in range(max_tokens):
                    ctx       = idx[:, -bs:]
                    logits, _ = mdl(ctx)
                    logits    = logits[:, -1, :] / temperature
                    probs     = F.softmax(logits, dim=-1)
                    next_t    = torch.multinomial(probs, num_samples=1)
                    idx       = torch.cat((idx, next_t), dim=1)
                    tok_text  = decode_token(next_t[0].item())
                    emit({"token": tok_text, "done": False})

            emit({"token": "", "done": True})

        except Exception as e:
            emit({"error": str(e), "done": True})

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

def _port_libre():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def main():
    os.chdir(ROOT)
    port = _port_libre()
    srv  = HTTPServer(("localhost", port), ChatHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    url = "http://localhost:" + str(port) + "/chatting/index.html"
    print("\n  WishAI Chat -> " + url)
    print("  Ctrl+C pour quitter\n")

    with open(os.path.join(ROOT, "system", "chat_url.json"), "w", encoding="utf-8") as f:
        json.dump({"port": port, "url": url}, f)

    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    srv.shutdown()

if __name__ == "__main__":
    main()
