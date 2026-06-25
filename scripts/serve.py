# serve.py -- serveur local WishAI (remplace Live Server)
# python serve.py            -> ouvre dashboard.html
# python serve.py library    -> ouvre library.html

import os, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")

page = "web/library.html" if (len(sys.argv) > 1 and sys.argv[1] == "library") else "dashboard.html"

# Lance dashboard.py qui sert TOUS les fichiers HTML + API
proc = subprocess.Popen(
    [sys.executable, os.path.join(SRC, "dashboard.py")],
    cwd=ROOT,
    env={**os.environ, "_WISHAI_OPEN_PAGE": page}
)

print(f"\n  Serveur demarre -- ouverture de {page}")
print("  Ctrl+C pour arreter\n")

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    print("\n  Serveur arrete.")
