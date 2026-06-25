"""
=================================================================
  MONITEUR SYSTÈME — SERVEUR HTTP + WATCHDOG PROTECTION
  Tourne sur http://localhost:8001/
  Le dashboard fetch les données directement, sans fichier.
  -> Plus de rechargement de page !
=================================================================
"""

import json, time, os, sys, threading, subprocess, socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

try:
    import psutil
except ImportError:
    os.system("pip install psutil")
    import psutil

def _nvidia_smi_info():
    """Récupère nom GPU + VRAM totale via nvidia-smi (pas besoin de torch)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        parts = r.stdout.strip().split(",")
        nom   = parts[0].strip()
        total = float(parts[1].strip()) / 1024  # MiB → Go
        return True, nom, total
    except Exception:
        return False, "Pas de GPU", 0.0

CUDA, GPU_NOM, GPU_TOTAL = _nvidia_smi_info()

# ── Chemins ───────────────────────────────────────────────────────
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.dirname(_SRC_DIR)
_SYS_DIR     = os.path.join(_ROOT_DIR, "system")
CONTROL_FILE = os.path.join(_SYS_DIR, "control.json")
CONFIG_FILE  = os.path.join(_ROOT_DIR, "config.json")

# ── Niveau de protection ──────────────────────────────────────────
sys.path.insert(0, _SRC_DIR)
from protection import charger_seuils as _charger_seuils
_PROT_SEUILS, _PROT_NIVEAU = _charger_seuils(CONFIG_FILE)


def lire_vram_nvidia_smi():
    """Lit la VRAM réelle via nvidia-smi (tous processus confondus)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        return float(r.stdout.strip()) / 1024  # MiB → Go
    except Exception:
        return 0.0

def lire_temp_gpu():
    """Lit la température GPU via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        return int(r.stdout.strip())
    except Exception:
        return 0

def _lire_control():
    try:
        with open(CONTROL_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"commande": "run"}

def _ecrire_control(commande, **extra):
    data = {"commande": commande, "timestamp": time.time()}
    data.update(extra)
    # Ecriture atomique (.tmp + os.replace) pour eviter un fichier corrompu.
    try:
        tmp = CONTROL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, CONTROL_FILE)
    except Exception:
        pass


# ── Données partagées entre threads ──────────────────────────────
etat      = {}
etat_lock = threading.Lock()

LIMITE_VRAM = round(GPU_TOTAL * 0.85, 1) if GPU_TOTAL else 5.0
LIMITE_RAM  = round(psutil.virtual_memory().total / 1e9 * 0.96, 1)


def lire_memoire():
    """Lit la mémoire + température et met à jour l'état global."""
    while True:
        vram = lire_vram_nvidia_smi()
        temp = lire_temp_gpu()
        ram  = psutil.virtual_memory()
        cpu  = psutil.cpu_percent(interval=None)
        s    = _PROT_SEUILS

        # Statut thermique selon niveau de protection
        crit_temp = s["critique_temp"]
        if temp >= crit_temp:
            statut_temp = "🔴 CRITIQUE"
        elif temp >= crit_temp - 5:
            statut_temp = "🟠 CHAUD"
        elif temp >= 65:
            statut_temp = "🟡 OK"
        else:
            statut_temp = "🟢 OK"

        data = {
            "timestamp": time.strftime("%H:%M:%S"),
            "protection": _PROT_NIVEAU,
            "gpu": {
                "nom":        GPU_NOM,
                "total_go":   round(GPU_TOTAL, 1),
                "utilise_go": round(vram, 3),
                "pct":        round(vram / GPU_TOTAL * 100, 2) if GPU_TOTAL else 0,
                "limite_go":  LIMITE_VRAM,
                "limite_pct": round(LIMITE_VRAM / GPU_TOTAL * 100, 1) if GPU_TOTAL else 66.7,
                "alerte":     vram > LIMITE_VRAM,
                "temp_c":     temp,
                "temp_statut": statut_temp,
                "temp_alerte": temp >= (crit_temp - 5),
            },
            "ram": {
                "total_go":   round(ram.total / 1e9, 1),
                "utilise_go": round(ram.used  / 1e9, 2),
                "pct":        round(ram.percent, 1),
                "limite_go":  LIMITE_RAM,
                "limite_pct": round(LIMITE_RAM / (ram.total / 1e9) * 100, 1),
                "alerte":     ram.percent >= s["alerte_ram"],
                "alerte_pct": s["alerte_ram"],
            },
            "cpu": { "pct": cpu },
        }

        with etat_lock:
            etat.update(data)

        # L'affichage terminal en boucle est désactivé car il entre en conflit
        # avec les logs d'entraînement (nanogpt.py) et crée des clones à l'écran.
        # Toutes ces données sont visibles en temps réel et en plus joli sur dashboard.html !
        time.sleep(1)

lire_memoire._init = False


def watchdog():
    """
    Surveille control.json.
    - Si "pause"         : attend RAM OK → écrit "resume"    (nanogpt_bpe reprend)
    - Si "arret_critique": attend RAM+temp OK → écrit "reprendre" (go.py relance)
    """
    s = _PROT_SEUILS
    en_attente = False

    while True:
        try:
            ctrl     = _lire_control()
            commande = ctrl.get("commande", "run")

            if commande in ("pause", "arret_critique") and not en_attente:
                en_attente = True
                raison     = ctrl.get("raison", "")
                signal_cible = "resume" if commande == "pause" else "reprendre"

                print(f"\n  ⏸️  Watchdog : {commande} ({raison})")
                print(f"     Attente : RAM < {s['resume_ram']}%  GPU < {s['resume_temp']}°C")

                while True:
                    ram_pct = psutil.virtual_memory().percent
                    temp    = lire_temp_gpu()
                    ram_ok  = ram_pct <= s["resume_ram"]
                    temp_ok = temp == 0 or temp <= s["resume_temp"]

                    if ram_ok and temp_ok:
                        break

                    # Affiche l'état d'attente
                    ram_icon  = "✅" if ram_ok  else "⏳"
                    temp_icon = "✅" if temp_ok else "⏳"
                    print(f"\r     {ram_icon} RAM {ram_pct:.1f}%  "
                          f"{temp_icon} GPU {temp}°C", end="", flush=True)
                    time.sleep(5)

                print(f"\n  ✅ Conditions OK — signal '{signal_cible}'")
                _ecrire_control(signal_cible)
                en_attente = False

            elif commande in ("run", "resume", "reprendre"):
                en_attente = False

        except Exception:
            pass

        time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        if self.path == "/events":
            self._sse()
        else:
            with etat_lock:
                payload = json.dumps(etat, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)

    def _sse(self):
        """Server-Sent Events — pousse l'etat systeme en temps reel."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                with etat_lock:
                    payload = json.dumps(etat, ensure_ascii=False)
                self.wfile.write(("data: " + payload + "\n\n").encode())
                self.wfile.flush()
                time.sleep(1)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


# -- Lancement des threads -------------------------------------------------
threading.Thread(target=lire_memoire, daemon=True).start()
threading.Thread(target=watchdog,     daemon=True).start()


def _port_libre():
    """Trouve un port TCP disponible."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

port = _port_libre()

# Ecrit le port dans monitor_port.json pour que dashboard.py puisse le lire
_port_file = os.path.join(_SYS_DIR, "monitor_port.json")
try:
    with open(_port_file, "w", encoding="utf-8") as _f:
        json.dump({"port": port}, _f)
except Exception:
    pass

print("=" * 54)
print("  MONITEUR DEMARRE")
print("  GPU  : " + GPU_NOM)
print("  VRAM : " + str(round(GPU_TOTAL, 1)) + " Go  |  Limite : " + str(LIMITE_VRAM) + " Go")
print("  RAM  : " + str(round(psutil.virtual_memory().total / 1e9, 1)) + " Go")
print("  Protection : " + _PROT_NIVEAU.upper())
print("  Serveur    : http://localhost:" + str(port) + "/")
print("  SSE        : http://localhost:" + str(port) + "/events")
print("=" * 54 + "\n")

try:
    ThreadingHTTPServer(("localhost", port), Handler).serve_forever()
except KeyboardInterrupt:
    print("\n\nMoniteur arrete.")
