"""
=================================================================
  GÉNÉRATEUR DE TEXTE — WishAI by Liam
  Supporte les modèles char-level ET BPE
=================================================================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================================================================
# SCAN DES MODÈLES DISPONIBLES
# ================================================================

def lister_modeles():
    modeles = []
    # Modèles BPE : model/<nom>/modele.pt
    for f in sorted(glob.glob(os.path.join("model", "*", "modele.pt"))):
        nom = os.path.basename(os.path.dirname(f))
        modeles.append({"fichier": f, "nom": nom, "type": "BPE"})
    # Modèle char-level (legacy)
    if os.path.exists("modele_sauvegarde.pt"):
        modeles.append({"fichier": "modele_sauvegarde.pt", "nom": "char-level", "type": "CHAR"})
    return modeles

modeles = lister_modeles()

if not modeles:
    print("="*55)
    print("  ❌ Aucun modèle trouvé !")
    print()
    print("  Lance d'abord l'entraînement :")
    print("    python nanogpt_bpe.py   ← modèle BPE (recommandé)")
    print("    python nanogpt.py       ← modèle char-level")
    print("="*55)
    sys.exit(0)

print("\n" + "="*55)
print("  MODÈLES DISPONIBLES")
print("="*55)
for i, m in enumerate(modeles):
    taille = os.path.getsize(m["fichier"]) / 1_000_000
    print(f"  [{i+1}]  {m['nom']:<20} ({m['type']})  {taille:.0f} Mo")
print("="*55)

choix = input("\n  Choix > ").strip()
try:
    idx = int(choix) - 1
    if not (0 <= idx < len(modeles)):
        raise ValueError
except:
    idx = 0

modele_choisi = modeles[idx]
print(f"\n  → {modele_choisi['nom']} ({modele_choisi['type']})\n")

# ================================================================
# CHARGEMENT
# ================================================================

print("Chargement du modèle...")
ckpt = torch.load(modele_choisi["fichier"], map_location=device, weights_only=False)

# ── Architecture commune ──────────────────────────────────────────

def build_modele(taille_vocab, n_embd, n_head, n_layer, block_size, dropout):

    class CoucheAttention(nn.Module):
        def __init__(self, taille_tete):
            super().__init__()
            self.key   = nn.Linear(n_embd, taille_tete, bias=False)
            self.query = nn.Linear(n_embd, taille_tete, bias=False)
            self.value = nn.Linear(n_embd, taille_tete, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x); q = self.query(x)
            scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
            scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            scores = F.softmax(scores, dim=-1)
            scores = self.dropout(scores)
            return scores @ self.value(x)

    class MultiTetesAttention(nn.Module):
        def __init__(self, n_tetes, taille_tete):
            super().__init__()
            self.tetes     = nn.ModuleList([CoucheAttention(taille_tete) for _ in range(n_tetes)])
            self.projection = nn.Linear(taille_tete * n_tetes, n_embd)
            self.dropout   = nn.Dropout(dropout)
        def forward(self, x):
            return self.dropout(self.projection(torch.cat([t(x) for t in self.tetes], dim=-1)))

    class CoucheFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.reseau = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
            )
        def forward(self, x): return self.reseau(x)

    class BlocTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            taille_tete = n_embd // n_head
            self.attention = MultiTetesAttention(n_head, taille_tete)
            self.ffn   = CoucheFFN()
            self.norm1 = nn.LayerNorm(n_embd)
            self.norm2 = nn.LayerNorm(n_embd)
        def forward(self, x):
            x = x + self.attention(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

    class WishAI(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding    = nn.Embedding(taille_vocab, n_embd)
            self.position_embedding = nn.Embedding(block_size, n_embd)
            self.blocs       = nn.Sequential(*[BlocTransformer() for _ in range(n_layer)])
            self.norm_finale = nn.LayerNorm(n_embd)
            self.tete_lm     = nn.Linear(n_embd, taille_vocab)
        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=device))
            x = self.norm_finale(self.blocs(x))
            logits = self.tete_lm(x)
            perte = None
            if targets is not None:
                B, T, V = logits.shape
                perte = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
            return logits, perte
        def generer(self, idx, max_nouveaux_tokens, temperature=1.0):
            for _ in range(max_nouveaux_tokens):
                idx_ctx = idx[:, -block_size:]
                logits, _ = self(idx_ctx)
                logits = logits[:, -1, :] / temperature
                probs  = F.softmax(logits, dim=-1)
                idx    = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
            return idx

    return WishAI()

# ── Chargement selon le type ──────────────────────────────────────

if modele_choisi["type"] == "BPE":
    # Tokenizer BPE
    from tokenizer import TokenizerBPE
    nom = modele_choisi["nom"]
    # Cherche le tokenizer dans le dossier du modèle, puis à la racine
    tok_file = os.path.join("model", nom, "tokenizer.json")
    if not os.path.exists(tok_file):
        tok_file = "tokenizer.json"
    tok = TokenizerBPE()
    tok.charger(tok_file)

    def encoder(texte): return tok.encoder(texte)
    def decoder(ids):   return tok.decoder(ids)

    hp           = ckpt["hyperparams"]
    taille_vocab = ckpt["taille_vocab"]
    n_embd       = hp["n_embd"]
    n_head       = hp["n_head"]
    n_layer      = hp["n_layer"]
    block_size   = hp["block_size"]
    dropout      = hp["dropout"]

else:
    # Char-level
    char_vers_int = ckpt["char_vers_int"]
    int_vers_char = ckpt["int_vers_char"]
    taille_vocab  = ckpt["taille_vocab"]

    def encoder(texte): return [char_vers_int.get(c, 0) for c in texte]
    def decoder(ids):   return ''.join([int_vers_char.get(i, '?') for i in ids])

    hp         = ckpt["hyperparams"]
    n_embd     = hp["n_embd"]
    n_head     = hp["n_head"]
    n_layer    = hp["n_layer"]
    block_size = hp["block_size"]
    dropout    = hp["dropout"]

modele = build_modele(taille_vocab, n_embd, n_head, n_layer, block_size, dropout).to(device)
modele.load_state_dict(ckpt["modele_state"])
modele.eval()

nb_params = sum(p.numel() for p in modele.parameters())
print(f"✅ Modèle chargé : {nb_params:,} paramètres ({modele_choisi['type']})")
print(f"   Device : {device}\n")

# ================================================================
# MODE INTERACTIF
# ================================================================

print("="*60)
print("  GÉNÉRATEUR DE TEXTE INTERACTIF")
print("="*60)
print("  Tape un début de phrase → l'IA continue")
print("  t=0.5  → texte prévisible   t=1.5 → créatif")
print("  n=200  → longueur de la réponse")
print("  q      → quitter")
print("="*60 + "\n")

temperature = 0.8
nb_tokens   = 200

while True:
    try:
        entree = input("Toi > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAu revoir !")
        break

    if entree.lower() == 'q':
        print("Au revoir !"); break

    if entree.startswith('t='):
        try:
            temperature = float(entree[2:])
            print(f"  → Température : {temperature}")
        except: print("Format : t=0.8")
        continue

    if entree.startswith('n='):
        try:
            nb_tokens = int(entree[2:])
            print(f"  → Tokens : {nb_tokens}")
        except: print("Format : n=200")
        continue

    with torch.no_grad():
        if entree:
            ids = encoder(entree)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
        else:
            idx = torch.zeros((1, 1), dtype=torch.long, device=device)

        sortie = modele.generer(idx, max_nouveaux_tokens=nb_tokens, temperature=temperature)
        texte  = decoder(sortie[0].tolist())

    print(f"\nIA  > {texte}\n")
