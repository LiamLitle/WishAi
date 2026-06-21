# chat.py -- lance l'interface de chat WishAI
# python chat.py

import os, sys, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
URL_FILE = os.path.join(ROOT, "chat_url.json")

# Supprimer l'ancien fichier de port
if os.path.exists(URL_FILE):
    try: os.remove(URL_FILE)
    except: pass

print("\n  WishAI Chat")
print("  Demarrage du serveur...\n")

proc = subprocess.Popen(
    [sys.executable, os.path.join(SRC, "chat_server.py")],
    cwd=ROOT
)

# Attendre que chat_server.py ecrive chat_url.json
for _ in range(40):
    if os.path.exists(URL_FILE):
        break
    time.sleep(0.25)

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    print("\n  Serveur arrete.")
