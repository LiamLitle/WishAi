# btn_dashboard.py -- petit bouton flottant qui apparait pendant l'entrainement
# Lance automatiquement avec go.py, ou manuellement :
#   python src/btn_dashboard.py

import os, sys, json, subprocess, glob, tkinter as tk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.dirname(os.path.abspath(__file__))

W, H = 210, 64

BLEU   = "#3b82f6"
BLEU_H = "#2563eb"
BG     = "#111827"
VERT   = "#10b981"
SUBTIL = "#6b7280"
BLANC  = "#f9fafb"


def _trouver_log():
    for log in glob.glob(os.path.join(ROOT, "model", "*", "log_active.json")):
        try:
            with open(log, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("status") in ("entrainement", "running", "en cours", "pause"):
                return d, log
        except Exception:
            pass
    return None, None


def _ouvrir_dashboard():
    subprocess.Popen([sys.executable, os.path.join(SRC, "dashboard.py")], cwd=ROOT)


class BoutonDashboard:
    def __init__(self, root):
        self.root = root
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg=BG)
        root.resizable(False, False)

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{W}x{H}+{sw - W - 18}+{sh - H - 56}")

        outer = tk.Frame(root, bg=BG, padx=1, pady=1)
        outer.pack(fill="both", expand=True)
        inner = tk.Frame(outer, bg=BG, padx=10, pady=8)
        inner.pack(fill="both", expand=True)

        top = tk.Frame(inner, bg=BG)
        top.pack(fill="x")

        self.lbl = tk.Label(top, text="", bg=BG, fg=VERT,
                            font=("Segoe UI", 8), anchor="w")
        self.lbl.pack(side="left", fill="x", expand=True)

        close_btn = tk.Label(top, text="x", bg=BG, fg=SUBTIL,
                             font=("Segoe UI", 8), cursor="hand2")
        close_btn.pack(side="right")
        close_btn.bind("<Button-1>", lambda e: root.destroy())

        self.btn = tk.Button(
            inner, text="Dashboard",
            command=_ouvrir_dashboard,
            bg=BLEU, fg=BLANC,
            activebackground=BLEU_H, activeforeground=BLANC,
            relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"),
            padx=0, pady=3, bd=0,
        )
        self.btn.pack(fill="x", pady=(4, 0))

        for w in (outer, inner, top, self.lbl):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>",     self._drag_move)

        self._dx = self._dy = 0
        self._poll()

    def _drag_start(self, e):
        self._dx = e.x_root - self.root.winfo_x()
        self._dy = e.y_root - self.root.winfo_y()

    def _drag_move(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _poll(self):
        data, _ = _trouver_log()
        if data:
            nom   = data.get("nom_modele", "modele")
            etape = data.get("iteration_courante", data.get("epoch", "?"))
            total = data.get("max_iters", data.get("epochs", "?"))
            self.lbl.config(text=f"  {nom}  etape {etape}/{total}", fg=VERT)
            self.btn.config(state="normal", bg=BLEU)
            self.root.deiconify()
        else:
            self.root.withdraw()
        self.root.after(2000, self._poll)


if __name__ == "__main__":
    root = tk.Tk()
    BoutonDashboard(root)
    root.mainloop()
