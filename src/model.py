"""
=================================================================
  MODELE WishAI — Architecture moderne (2024)
=================================================================
Inspiré de LLaMA / Mistral :
  • RMSNorm        au lieu de LayerNorm  (plus stable, plus rapide)
  • Pre-norm       (inchangé par rapport à GPT-2)
  • RoPE           au lieu d'embeddings de position appris
                   (meilleure généralisation sur les séquences longues)
  • SwiGLU         au lieu de ReLU/GeLU dans le FFN
                   (meilleure loss à paramètres égaux)
  • Sans biais     sur les couches linéaires (comme LLaMA)

Nombre de paramètres MINI (n_embd=384, n_head=6, n_layer=10) : ~20M
=================================================================
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ConfigModele:
    """Hyperparamètres d'architecture du modèle."""
    vocab_size: int
    n_embd:     int
    n_head:     int
    n_layer:    int
    block_size: int
    dropout:    float = 0.0


# ================================================================
#  RMSN ORM
#  Remplace LayerNorm : pas de biais, calcul plus simple.
#  x_norm = weight * x / sqrt(mean(x²) + eps)
# ================================================================

class RMSNorm(nn.Module):
    """RMSNorm : normalisation par la racine de la moyenne des carrés."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # rsqrt = 1 / sqrt
        norme = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norme


# ================================================================
#  ROPE — ROTARY POSITION EMBEDDING
#  Applique une rotation aux vecteurs Q et K selon leur position.
#  Avantage vs embeddings appris : généralise mieux à des longueurs
#  non vues à l'entraînement, et encode la distance relative.
# ================================================================

def precomputer_rope(taille_tete, block_size, device="cpu"):
    """
    Précalcule les fréquences cos/sin pour RoPE.

    Retourne deux tenseurs de forme [block_size, taille_tete // 2].
    """
    assert taille_tete % 2 == 0, "taille_tete doit etre paire pour RoPE"
    # Fréquences : theta_i = 1 / 10000^(2i / d)
    theta     = 1.0 / (10000 ** (
        torch.arange(0, taille_tete, 2, device=device).float() / taille_tete
    ))
    positions = torch.arange(block_size, device=device).float()
    freqs     = torch.outer(positions, theta)   # [T, d//2]
    return freqs.cos(), freqs.sin()


def appliquer_rope(x, cos, sin):
    """
    Applique RoPE au tenseur x de forme [B, n_head, T, taille_tete].

    Paires de dimensions (0,1), (2,3), ... → rotation par (cos, sin).
    """
    T   = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)   # [1, 1, T, d//2]
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1  = x[..., ::2]    # dimensions paires
    x2  = x[..., 1::2]   # dimensions impaires
    # Rotation dans le plan complexe :
    # x_new[2i]   = x[2i]   * cos - x[2i+1] * sin
    # x_new[2i+1] = x[2i]   * sin + x[2i+1] * cos
    x_rot = torch.stack([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)   # [B, n_head, T, taille_tete]


# ================================================================
#  ATTENTION MULTI-TETES (avec RoPE)
#  Implémentation fusionnée (toutes les têtes en un seul tenseur).
# ================================================================

class AttentionMultiTetes(nn.Module):
    """
    Multi-Head Attention avec RoPE.

    Différences vs GPT-2 :
      - Projections Q/K/V sans biais
      - RoPE appliqué sur Q et K avant le produit scalaire
      - Projection de sortie sans biais
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd doit etre divisible par n_head"
        self.n_head      = cfg.n_head
        self.taille_tete = cfg.n_embd // cfg.n_head

        # Projections sans biais (comme LLaMA)
        self.wq          = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk          = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wv          = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj_sortie = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        self.dropout_att = nn.Dropout(cfg.dropout)
        self.dropout_res = nn.Dropout(cfg.dropout)

        # Masque causal (triangulaire inférieur)
        self.register_buffer(
            'masque_causal',
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        H = self.n_head
        D = self.taille_tete

        # Projections Q, K, V → forme [B, H, T, D]
        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)

        # RoPE sur Q et K
        q = appliquer_rope(q, cos, sin)
        k = appliquer_rope(k, cos, sin)

        # Scores d'attention (divisés par √D pour stabilité)
        scores = q @ k.transpose(-2, -1) * (D ** -0.5)
        # Masque causal : bloque l'accès aux positions futures
        scores = scores.masked_fill(self.masque_causal[:T, :T] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_att(scores)

        # Combinaison des valeurs puis projection
        out = (scores @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout_res(self.proj_sortie(out))


# ================================================================
#  FEED-FORWARD SWIGLU
#  Remplace ReLU/GeLU par SwiGLU : deux branches + porte SiLU.
#
#  FFN(x) = W3( SiLU(W1(x)) * W2(x) )
#
#  Taille cachée : 8/3 × n_embd (arrondi au multiple de 64).
#  Pour n_embd=384 : 8/3 × 384 = 1024 exactement.
#  Paramètres FFN = 3 × n_embd × hidden (même ordre que GPT-2 ×4).
# ================================================================

class CoucheFFN(nn.Module):
    """Feed-Forward SwiGLU : porte SiLU sur deux projections parallèles."""

    def __init__(self, cfg):
        super().__init__()
        # 8/3 × n_embd, arrondi au multiple de 64
        hidden = int(8 / 3 * cfg.n_embd)
        hidden = ((hidden + 63) // 64) * 64

        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)   # projection "gate"
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)   # projection "up"
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)   # projection "down"
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # SwiGLU : SiLU(gate) * up → down
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ================================================================
#  BLOC TRANSFORMER MODERNE
#  Pre-norm + RMSNorm + AttentionMultiTetes(RoPE) + SwiGLU
# ================================================================

class BlocTransformer(nn.Module):
    """Bloc Transformer moderne : pre-norm, RMSNorm, RoPE, SwiGLU."""

    def __init__(self, cfg):
        super().__init__()
        self.attention = AttentionMultiTetes(cfg)
        self.ffn       = CoucheFFN(cfg)
        self.norm1     = RMSNorm(cfg.n_embd)
        self.norm2     = RMSNorm(cfg.n_embd)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.norm1(x), cos, sin)   # attention + résiduel
        x = x + self.ffn(self.norm2(x))                   # FFN + résiduel
        return x


# ================================================================
#  MODELE COMPLET : WishAI_BPE
# ================================================================

class WishAI_BPE(nn.Module):
    """
    Modèle WishAI complet — architecture moderne (RoPE + RMSNorm + SwiGLU).

    Entrée  : séquence d'IDs de tokens (entiers)
    Sortie  : logits sur le vocabulaire + perte (si targets fournis)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg             = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Pas d'embedding de position — RoPE s'en charge
        self.blocs           = nn.ModuleList([BlocTransformer(cfg) for _ in range(cfg.n_layer)])
        self.norm_finale     = RMSNorm(cfg.n_embd)
        self.tete_lm         = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_poids)

        # Précalcul des fréquences RoPE (buffers → suivent le .to(device))
        taille_tete = cfg.n_embd // cfg.n_head
        cos, sin    = precomputer_rope(taille_tete, cfg.block_size)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def _init_poids(self, module):
        """Initialisation des poids (std=0.02, sans biais)."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x    = self.token_embedding(idx)   # [B, T, n_embd]

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for bloc in self.blocs:
            x = bloc(x, cos, sin)

        x      = self.norm_finale(x)
        logits = self.tete_lm(x)

        perte = None
        if targets is not None:
            B, T, V = logits.shape
            perte = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, perte

    def generer(self, idx, max_nouveaux_tokens, temperature=1.0):
        """Génère des tokens un à un (autorégressif)."""
        for _ in range(max_nouveaux_tokens):
            idx_contexte = idx[:, -self.cfg.block_size:]
            logits, _    = self(idx_contexte)
            logits       = logits[:, -1, :] / temperature
            probs        = F.softmax(logits, dim=-1)
            idx_next     = torch.multinomial(probs, num_samples=1)
            idx          = torch.cat((idx, idx_next), dim=1)
        return idx
