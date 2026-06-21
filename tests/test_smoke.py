"""
=================================================================
  SMOKE TESTS — WishAI
=================================================================
Filet de sécurité minimal : 5 tests rapides qui vérifient que les
briques essentielles ne sont pas cassées. À lancer après chaque
modification du code.

  python -m unittest tests/test_smoke.py        (depuis la racine)
  python tests/test_smoke.py

Couverture :
  1. Tokenizer : round-trip encode → decode
  2. Tokenizer : détection de changement de dataset (signature)
  3. Modèle    : forward pass + calcul de la loss
  4. Modèle    : save / load d'un checkpoint (poids identiques)
  5. Modèle    : génération sans erreur
  (bonus) Tous les fichiers .py compilent (attrape les SyntaxError)
=================================================================
"""
import os
import sys
import json
import tempfile
import unittest
import py_compile
import contextlib
import io

# ── Rendre src/ importable ───────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch                                   # noqa: E402
from tokenizer import (                        # noqa: E402
    TokenizerBPE, signature_dataset, tokenizer_a_jour,
)
from model import WishAI_BPE, ConfigModele     # noqa: E402


def _petit_modele():
    """Un modèle minuscule pour des tests rapides."""
    cfg = ConfigModele(vocab_size=40, n_embd=32, n_head=4,
                        n_layer=2, block_size=16, dropout=0.0)
    return WishAI_BPE(cfg), cfg


class TestTokenizer(unittest.TestCase):

    def test_1_round_trip_encode_decode(self):
        """decode(encode(texte)) doit redonner le texte d'origine."""
        texte = "the cat sat on the mat and the dog ran "
        tok = TokenizerBPE()
        with contextlib.redirect_stdout(io.StringIO()):   # silence les logs d'entraînement
            tok.entrainer(texte * 200, vocab_size=80)
        phrase = "the cat ran on the mat"
        redecode = tok.decoder(tok.encoder(phrase))
        self.assertEqual(redecode.strip(), phrase.strip())

    def test_2_signature_detecte_changement_dataset(self):
        """La signature doit changer quand le dataset change."""
        with tempfile.TemporaryDirectory() as d:
            data = os.path.join(d, "data.txt")
            tokj = os.path.join(d, "tokenizer.json")

            with open(data, "w", encoding="utf-8") as f:
                f.write("hello world " * 1000)
            sig = signature_dataset(data)
            with open(tokj, "w", encoding="utf-8") as f:
                json.dump({"vocab_size": 100, "dataset_sig": sig,
                           "dataset_file": data}, f)

            ok, _ = tokenizer_a_jour(tokj, data)
            self.assertTrue(ok, "doit être à jour juste après sauvegarde")

            with open(data, "w", encoding="utf-8") as f:
                f.write("bonjour le monde " * 1000)
            ok, _ = tokenizer_a_jour(tokj, data)
            self.assertFalse(ok, "doit détecter le changement de dataset")


class TestModele(unittest.TestCase):

    def test_3_forward_pass_calcule_la_loss(self):
        """Un forward avec targets doit produire une loss finie."""
        m, cfg = _petit_modele()
        x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        logits, loss = m(x, y)
        self.assertEqual(tuple(logits.shape), (2, cfg.block_size, cfg.vocab_size))
        self.assertTrue(torch.isfinite(loss), "la loss doit être un nombre fini")

    def test_4_save_load_checkpoint(self):
        """Les poids rechargés doivent être identiques aux poids sauvegardés."""
        m, cfg = _petit_modele()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "modele.pt")
            torch.save(m.state_dict(), ckpt)

            m2 = WishAI_BPE(cfg)
            m2.load_state_dict(torch.load(ckpt, weights_only=True))

            for (n1, p1), (n2, p2) in zip(m.state_dict().items(),
                                          m2.state_dict().items()):
                self.assertEqual(n1, n2)
                self.assertTrue(torch.equal(p1, p2), f"poids différents : {n1}")

    def test_5_generation_sans_erreur(self):
        """generer() doit retourner une séquence plus longue, sans erreur."""
        m, cfg = _petit_modele()
        debut = torch.zeros((1, 1), dtype=torch.long)
        out = m.generer(debut, max_nouveaux_tokens=12)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 13)          # 1 départ + 12 générés
        self.assertEqual(out.dtype, torch.long)


class TestCompilation(unittest.TestCase):

    def test_6_tous_les_fichiers_compilent(self):
        """Aucun fichier .py du projet ne doit contenir de SyntaxError."""
        fichiers = []
        for nom in os.listdir(ROOT):
            if nom.endswith(".py"):
                fichiers.append(os.path.join(ROOT, nom))
        for nom in os.listdir(SRC):
            if nom.endswith(".py"):
                fichiers.append(os.path.join(SRC, nom))
        for f in fichiers:
            with self.subTest(fichier=os.path.basename(f)):
                try:
                    py_compile.compile(f, doraise=True)
                except py_compile.PyCompileError as e:
                    self.fail(f"{os.path.basename(f)} ne compile pas : {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
