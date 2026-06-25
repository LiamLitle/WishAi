"""\n=================================================================\n  TÉLÉCHARGEUR UNIFIÉ — WishAI by Liam\n  Toutes les sources de données au même endroit.\n=================================================================\n\nSTRUCTURE DES FICHIERS GÉNÉRÉS :\n  data/fr/data.txt     ← données françaises\n  data/en/data.txt     ← données anglaises\n  data/multi/data.txt  ← données multilingues\n  data/data.txt        ← dernier téléchargement (utilisé par nanogpt.py)\n\nSOURCES DISPONIBLES :\n  [1-8]  HuggingFace  (Wikipedia, Wikitext, OpenWebText, OSCAR...)\n  [9-11] Common Crawl (scraping direct du web, 3 bots en parallèle)\n=================================================================\n"""

import os
import re
import sys
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import gzip
import time
import shutil
import threading
import unicodedata
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# langdetect (optionnel — améliore la détection de langue, ~2 Mo)
try:
    from langdetect import detect as _langdetect_detect
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

# Logger + auto-réparation
try:
    from bot_logger import log, log_telechargement, auto_repair
except ImportError:
    def log(n, m, source=""): pass
    def log_telechargement(*a, **k): pass
    def auto_repair(p, pip_n=None): return False

# ================================================================
# CONFIG
# ================================================================
DATA_ROOT  = "data"
TEMP_DIR   = os.path.join(DATA_ROOT, "cc_temp")
CRAWL_ID   = "CC-MAIN-2025-13"
NB_WORKERS = 3       # bots parallèles pour Common Crawl
MIN_LEN    = 200     # longueur minimale d'une page

write_lock  = threading.Lock()
stop_event  = threading.Event()

# ================================================================
# FICHIERS CUSTOM & CACHE DE TAILLES
# ================================================================
_SRC_DIR             = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_SOURCES_FILE = os.path.join(_SRC_DIR, "custom_sources.json")
_SIZES_CACHE_FILE    = os.path.join(_SRC_DIR, "sizes_cache.json")
_sizes_cache         = {}

def _charger_sizes_cache():
    global _sizes_cache
    try:
        if os.path.exists(_SIZES_CACHE_FILE):
            with open(_SIZES_CACHE_FILE, 'r', encoding='utf-8') as f:
                _sizes_cache = json.load(f)
    except Exception:
        _sizes_cache = {}

def _sauvegarder_sizes_cache():
    try:
        with open(_SIZES_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_sizes_cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def obtenir_taille_hf(src):
    """Retourne la taille en Mo depuis le cache ou l'API HuggingFace (avec fallback)."""
    if src["type"] != "hf":
        return src.get("taille_max_mo")
    cache_key = f"{src['hf_path']}::{src.get('hf_config', '')}::{src.get('hf_split', 'train')}"
    entree    = _sizes_cache.get(cache_key)
    if entree and (time.time() - entree.get("ts", 0)) < 30 * 86400:
        return entree["mo"]
    try:
        from datasets import load_dataset_builder
        builder    = load_dataset_builder(src["hf_path"], src.get("hf_config"))
        info       = builder.info
        split      = src.get("hf_split", "train")
        size_bytes = None
        if info.splits and split in info.splits:
            size_bytes = getattr(info.splits[split], 'num_bytes', None)
        if not size_bytes:
            size_bytes = (getattr(info, 'download_size', None)
                         or getattr(info, 'dataset_size', None))
        if size_bytes and size_bytes > 0:
            mo = round(size_bytes / 1_000_000)
            _sizes_cache[cache_key] = {"mo": mo, "ts": time.time()}
            _sauvegarder_sizes_cache()
            return mo
    except Exception:
        pass
    return src.get("taille_max_mo")

_charger_sizes_cache()

# ================================================================
# SOURCES DISPONIBLES (La Bibliothèque)
# ================================================================

CATEGORIES = {
    "fr": "🇫🇷 Français (Général, Web, Littérature)",
    "en": "🇬🇧 Anglais (Général, Web, Littérature)",
    "code": "💻 Code & Programmation (GitHub, StackOverflow)",
    "science": "🔬 Mathématiques & Sciences",
    "instruct": "🤖 Instructions & Q/A (Pour faire un Assistant)",
    "dialog": "💬 Conversations & Dialogues",
    "multi": "🌍 Multilingue (Mélange)"
}

SOURCES = {
    # 🇫🇷 FRANÇAIS
    "fr_wiki":    {"nom": "Wikipedia Français",                               "cat": "fr",      "type": "hf", "hf_path": "wikimedia/wikipedia",                         "hf_config": "20231101.fr",             "hf_split": "train", "champ": "text",              "taille_max_mo": 4_000,   "taille_typique_mo": 800},
    "fr_mc4":     {"nom": "C4 Français (Texte web de Google)",                "cat": "fr",      "type": "hf", "hf_path": "allenai/c4",                                  "hf_config": "fr",                      "hf_split": "train", "champ": "text",  "slow": True, "taille_max_mo": 500_000, "taille_typique_mo": 2_000},
    "fr_culturax":{"nom": "CulturaX (Haute qualité fr)",                      "cat": "fr",      "type": "hf", "hf_path": "uonlp/CulturaX",                              "hf_config": "fr",                      "hf_split": "train", "champ": "text",  "gated": True,"taille_max_mo": 100_000, "taille_typique_mo": 5_000},
    "fr_oscar":   {"nom": "OSCAR FR (Nécessite compte HF)",                   "cat": "fr",      "type": "hf", "hf_path": "oscar-corpus/oscar",                          "hf_config": "unshuffled_deduplicated_fr","hf_split": "train", "champ": "text",  "gated": True,"taille_max_mo": 50_000,  "taille_typique_mo": 2_000},
    "fr_gutenberg":{"nom": "Project Gutenberg (Livres classiques domaine public)","cat": "fr",  "type": "hf", "hf_path": "manu/project_gutenberg",                       "hf_config": None,                      "hf_split": "fr",    "champ": "TEXT",              "taille_max_mo": 200,     "taille_typique_mo": 100},
    "fr_cc":      {"nom": "Common Crawl Français (Web brut, énorme)",         "cat": "fr",      "type": "cc", "filtre": ["fr"],                                                                                                                                          "taille_max_mo": None,    "taille_typique_mo": 2_000},

    # 🇬🇧 ANGLAIS
    "en_fineweb": {"nom": "FineWeb (Le meilleur dataset web actuel)",         "cat": "en",      "type": "hf", "hf_path": "HuggingFaceFW/fineweb",                       "hf_config": "sample-10BT",             "hf_split": "train", "champ": "text",  "slow": True, "taille_max_mo": 50_000,  "taille_typique_mo": 5_000},
    "en_finewebedu":{"nom": "FineWeb-Edu (Web éducatif, très haute qualité)", "cat": "en",      "type": "hf", "hf_path": "HuggingFaceFW/fineweb-edu",                   "hf_config": "sample-10BT",             "hf_split": "train", "champ": "text",              "taille_max_mo": 20_000,  "taille_typique_mo": 2_000},
    "en_wiki":    {"nom": "Wikipedia Anglais",                                 "cat": "en",      "type": "hf", "hf_path": "wikimedia/wikipedia",                         "hf_config": "20231101.en",             "hf_split": "train", "champ": "text",              "taille_max_mo": 20_000,  "taille_typique_mo": 2_000},
    "en_wikitext":{"nom": "Wikitext-103 (Classique)",                         "cat": "en",      "type": "hf", "hf_path": "Salesforce/wikitext",                         "hf_config": "wikitext-103-raw-v1",     "hf_split": "train", "champ": "text",              "taille_max_mo": 500,     "taille_typique_mo": 500},
    "en_openweb": {"nom": "OpenWebText (Données de GPT-2)",                   "cat": "en",      "type": "hf", "hf_path": "Skylion007/openwebtext",                      "hf_config": None,                      "hf_split": "train", "champ": "text",              "taille_max_mo": 40_000,  "taille_typique_mo": 2_000},
    "en_c4":      {"nom": "C4 (Texte web propre de Google)",                  "cat": "en",      "type": "hf", "hf_path": "allenai/c4",                                  "hf_config": "en",                      "hf_split": "train", "champ": "text",              "taille_max_mo": 500_000, "taille_typique_mo": 5_000},
    "en_pile_hn": {"nom": "CNN/DailyMail (Articles de presse anglais)",       "cat": "en",      "type": "hf", "hf_path": "abisee/cnn_dailymail",                        "hf_config": "3.0.0",                   "hf_split": "train", "champ": "article",           "taille_max_mo": 1_500,   "taille_typique_mo": 500},
    "en_pile_enron":{"nom": "Enron Emails (Emails professionnels)",           "cat": "en",      "type": "hf", "hf_path": "SetFit/enron_spam",                           "hf_config": None,                      "hf_split": "train", "champ": "text",              "taille_max_mo": 100,     "taille_typique_mo": 50},
    "en_cc":      {"nom": "Common Crawl Anglais (Web brut, énorme)",          "cat": "en",      "type": "cc", "filtre": ["en"],                                                                                                                                          "taille_max_mo": None,    "taille_typique_mo": 2_000},

    # 💻 CODE
    "code_stack":   {"nom": "The Stack (Python) - Code propre (Compte HF requis)",    "cat": "code", "type": "hf", "hf_path": "bigcode/the-stack-smol",                  "hf_config": "data/python",             "hf_split": "train", "champ": "content","gated": True, "taille_max_mo": 30_000,  "taille_typique_mo": 1_000},
    "code_stack_js":{"nom": "The Stack (JavaScript/HTML/CSS) (Compte HF requis)",     "cat": "code", "type": "hf", "hf_path": "bigcode/the-stack-smol",                  "hf_config": "data/javascript",         "hf_split": "train", "champ": "content","gated": True, "taille_max_mo": 30_000,  "taille_typique_mo": 1_000},
    "code_search":  {"nom": "CodeSearchNet Python (Code + Docstrings)",                "cat": "code", "type": "hf", "hf_path": "code-search-net/code_search_net",         "hf_config": "python",                  "hf_split": "train", "champ": "whole_func_string",  "taille_max_mo": 500,     "taille_typique_mo": 200},
    "code_stackoverflow":{"nom": "StackExchange / StackOverflow Q&A",                 "cat": "code", "type": "hf", "hf_path": "flax-sentence-embeddings/stackexchange_title_body_jsonl","hf_config": None,     "hf_split": "train", "champ": "body",  "slow": True,  "taille_max_mo": 10_000,  "taille_typique_mo": 500},
    "code_starcoder":{"nom": "StarCoder Data (Nécessite compte HF)",                  "cat": "code", "type": "hf", "hf_path": "bigcode/starcoderdata",                   "hf_config": "python",                  "hf_split": "train", "champ": "content","gated": True, "taille_max_mo": 100_000, "taille_typique_mo": 2_000},

    # 🔬 SCIENCES & MATHS
    "sci_openwebmath": {"nom": "OpenWebMath (Haut niveau mathématique)",      "cat": "science", "type": "hf", "hf_path": "open-web-math/open-web-math",                 "hf_config": None,                      "hf_split": "train", "champ": "text",              "taille_max_mo": 20_000,  "taille_typique_mo": 500},
    "sci_proofpile":   {"nom": "MetaMath QA (Mathématiques avancées)",        "cat": "science", "type": "hf", "hf_path": "meta-math/MetaMathQA",                        "hf_config": None,                      "hf_split": "train", "champ": "query", "champ2": "response", "slow": True, "taille_max_mo": 1_000, "taille_typique_mo": 200},
    "sci_arxiv":       {"nom": "ArXiv (Articles scientifiques)",              "cat": "science", "type": "hf", "hf_path": "CShorten/ML-ArXiv-Papers",                    "hf_config": None,                      "hf_split": "train", "champ": "abstract",          "taille_max_mo": 1_000,   "taille_typique_mo": 500},
    "sci_pubmed":      {"nom": "PubMed (Médecine et Biologie)",               "cat": "science", "type": "hf", "hf_path": "ccdv/pubmed-summarization",                   "hf_config": None,                      "hf_split": "train", "champ": "article",           "taille_max_mo": 10_000,  "taille_typique_mo": 500},
    "sci_phil":        {"nom": "ArXiv Articles Complets (Sciences diverses)", "cat": "science", "type": "hf", "hf_path": "ccdv/arxiv-summarization",                    "hf_config": None,                      "hf_split": "train", "champ": "article",           "taille_max_mo": 5_000,   "taille_typique_mo": 300},

    # 🤖 INSTRUCTIONS
    "inst_hermes":     {"nom": "OpenHermes 2.5 (Excellent dataset d'instructions)", "cat": "instruct","type": "hf","hf_path": "teknium/OpenHermes-2.5",               "hf_config": None,                      "hf_split": "train", "champ": "conversations","slow": True, "taille_max_mo": 2_000, "taille_typique_mo": 500},
    "inst_alpaca":     {"nom": "Alpaca Clean (Instructions de base de Stanford)",   "cat": "instruct","type": "hf","hf_path": "yahma/alpaca-cleaned",                  "hf_config": None,                      "hf_split": "train", "champ": "instruction","champ2": "output",          "taille_max_mo": 50,    "taille_typique_mo": 50},
    "inst_dolly":      {"nom": "Dolly 15k (Instructions écrites par des humains)",  "cat": "instruct","type": "hf","hf_path": "databricks/databricks-dolly-15k",       "hf_config": None,                      "hf_split": "train", "champ": "instruction","champ2": "response",        "taille_max_mo": 20,    "taille_typique_mo": 20},
    "inst_squad_fr":   {"nom": "OpenAssistant 2 (Conversations multilingues FR/EN)","cat": "instruct","type": "hf","hf_path": "OpenAssistant/oasst2",                  "hf_config": None,                      "hf_split": "train", "champ": "text",  "langue_cible": "fr",          "taille_max_mo": 100,   "taille_typique_mo": 50},
    "inst_tinystories":{"nom": "TinyStories (Histoires courtes faciles, idéal débutants)","cat":"instruct","type":"hf","hf_path":"roneneldan/TinyStories",             "hf_config": None,                      "hf_split": "train", "champ": "text",              "taille_max_mo": 3_000,   "taille_typique_mo": 500},

    # 💬 DIALOGUES
    "dial_daily":   {"nom": "DailyDialog (Conversations quotidiennes)",       "cat": "dialog",  "type": "hf", "hf_path": "frankdarkluo/DailyDialog",                    "hf_config": None,                      "hf_split": "train", "champ": "context", "champ2": "response",        "taille_max_mo": 10,    "taille_typique_mo": 5},
    "dial_movies":  {"nom": "OpenSubtitles (Dialogues films/séries)",         "cat": "dialog",  "type": "hf", "hf_path": "frankdarkluo/OpenSubtitles",                  "hf_config": None,                      "hf_split": "train", "champ": "context", "champ2": "response",        "taille_max_mo": 500,   "taille_typique_mo": 100},
    "dial_ubuntu":  {"nom": "HH-RLHF Anthropic (Dialogues utiles et inoffensifs)","cat": "dialog","type": "hf","hf_path": "Anthropic/hh-rlhf",                         "hf_config": None,                      "hf_split": "train", "champ": "chosen",            "taille_max_mo": 100,   "taille_typique_mo": 50},
    "dial_reddit":  {"nom": "XSum BBC (Articles de presse + résumés)",        "cat": "dialog",  "type": "hf", "hf_path": "EdinburghNLP/xsum",                           "hf_config": None,                      "hf_split": "train", "champ": "document",          "taille_max_mo": 200,   "taille_typique_mo": 50},
    "dial_persona": {"nom": "SAMSum (Dialogues de chat résumés)",             "cat": "dialog",  "type": "hf", "hf_path": "knkarthick/samsum",                           "hf_config": None,                      "hf_split": "train", "champ": "dialogue",          "taille_max_mo": 10,    "taille_typique_mo": 5},

    # 🌍 MULTI
    "multi_cc":     {"nom": "Common Crawl Multilingue (Extrêmement massif)", "cat": "multi",   "type": "cc", "filtre": None,                                                                                                                                            "taille_max_mo": None,    "taille_typique_mo": 2_000},
    "multi_madlad": {"nom": "MADLAD-400 (Google multilingue)",               "cat": "multi",   "type": "hf", "hf_path": "allenai/MADLAD-400",                          "hf_config": None,                      "hf_split": "train", "champ": "text",  "gated": True, "taille_max_mo": 200_000, "taille_typique_mo": 5_000},
}

# ── Chargement des sources personnalisées (custom_sources.json) ───
def _charger_sources_custom():
    if not os.path.exists(_CUSTOM_SOURCES_FILE):
        return
    try:
        with open(_CUSTOM_SOURCES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for src in data.get("sources_custom", []):
            cle = src.get("id", "")
            if cle and cle not in SOURCES:
                src_copy = {k: v for k, v in src.items() if k != "id"}
                src_copy["_custom"] = True
                SOURCES[cle] = src_copy
    except Exception as e:
        print(f"  ⚠️  Erreur chargement custom_sources.json : {e}")

_charger_sources_custom()

def get_langue_by_cat(cat):
    if cat in ["fr", "fr_instruct"]: return "fr"
    if cat in ["multi"]: return "multi"
    return "en" # Par défaut, le code, les maths, etc. vont dans "en" car la majorité du vocab est anglais

FLAG = {"fr": "🇫🇷", "en": "🇬🇧", "multi": "🌍"}

# ================================================================
# UTILITAIRES SOURCES
# ================================================================

def slug_source(nom):
    """Transforme un nom de source en nom de fichier propre."""
    nom = unicodedata.normalize('NFD', nom)
    nom = ''.join(c for c in nom if unicodedata.category(c) != 'Mn')
    nom = nom.split('(')[0].strip()
    nom = nom.lower()
    nom = re.sub(r'[^a-z0-9]+', '_', nom)
    return nom.strip('_')

def lister_sources(langue):
    """Liste les fichiers dans data/{langue}/sources/."""
    sources_dir = os.path.join(DATA_ROOT, langue, "sources")
    if not os.path.exists(sources_dir):
        return []
    return sorted([f for f in os.listdir(sources_dir) if f.endswith('.txt')])

def afficher_sources_dispo(langue):
    """Affiche les sources téléchargées avec leurs tailles."""
    fichiers = lister_sources(langue)
    if not fichiers:
        print(f"  (aucune source pour '{langue}')")
        return {}
    tailles = {}
    for i, f in enumerate(fichiers):
        chemin = os.path.join(DATA_ROOT, langue, "sources", f)
        mo = os.path.getsize(chemin) / 1_000_000
        tailles[f] = mo
        print(f"    [{i+1}] {f:<40} {mo:.1f} Mo")
    return tailles

def combiner_sources(langue=None):
    """Combine les sources en data/{langue}/data.txt avec proportions custom."""
    if not langue:
        print("\nLangue à combiner ?")
        print("  [1] en (anglais)  [2] fr (français)  [3] multi")
        c = input("  > ").strip()
        langue = {"1":"en","2":"fr","3":"multi"}.get(c, "en")

    sources_dir = os.path.join(DATA_ROOT, langue, "sources")
    output_file = os.path.join(DATA_ROOT, langue, "data.txt")
    fichiers    = lister_sources(langue)

    if not fichiers:
        print(f"\n❌ Pas de sources pour '{langue}'. Lance d'abord un téléchargement.")
        return

    print(f"\n{'='*62}")
    print(f"  SOURCES DISPONIBLES — {langue.upper()}")
    print(f"{'='*62}")
    tailles = afficher_sources_dispo(langue)
    total   = sum(tailles.values())
    print(f"  {'─'*52}")
    print(f"    Total : {total:.1f} Mo")
    print(f"{'='*62}")

    if len(fichiers) > 1:
        print("\n  Proportions ? (Entrée = égales)")
        print(f"  Tape {len(fichiers)} nombres séparés par espaces.")
        print("  Ex: '70 30' → 70% source 1, 30% source 2")
        prop_str = input("  > ").strip()
        if not prop_str:
            poids = [1.0] * len(fichiers)
        else:
            try:
                poids = [float(x) for x in prop_str.split()]
                if len(poids) != len(fichiers):
                    print(f"❌ Il faut {len(fichiers)} valeurs.")
                    return
            except:
                print("❌ Format invalide."); return
    else:
        poids = [1.0]

    print("\n  Taille totale du data.txt final (Mo) ? (0 = tout)")
    try:
        cible_mo = int(input("  Mo > ").strip() or "0")
    except:
        cible_mo = 0

    cible_chars    = cible_mo * 1_000_000 if cible_mo > 0 else int(total * 1_000_000)
    total_poids    = sum(poids)
    chars_par_src  = {fichiers[i]: int(cible_chars * poids[i] / total_poids)
                      for i in range(len(fichiers))}

    print(f"\n{'='*62}")
    print(f"  COMBINAISON → {output_file}")
    print(f"{'='*62}")
    for f in fichiers:
        print(f"  {f:<40} → {chars_par_src[f]//1_000_000:.1f} Mo")
    print()

    with open(output_file, 'w', encoding='utf-8') as out:
        for f in fichiers:
            chemin    = os.path.join(sources_dir, f)
            limite    = chars_par_src[f]
            chars_lus = 0
            with open(chemin, 'r', encoding='utf-8') as sf:
                while chars_lus < limite:
                    chunk = sf.read(65536)
                    if not chunk:
                        break
                    restant = limite - chars_lus
                    out.write(chunk[:restant])
                    chars_lus += min(len(chunk), restant)
            print(f"  ✅ {f}  ({chars_lus//1_000_000:.1f} Mo)")

    taille_finale = os.path.getsize(output_file) / 1_000_000
    shutil.copy2(output_file, os.path.join(DATA_ROOT, "data.txt"))

    print(f"\n{'='*62}")
    print(f"  ✅ data.txt combiné : {taille_finale:.1f} Mo")
    print("  ⚠️  Supprime bpe_cache.pt avant de relancer l'entraînement !")
    print("     Remove-Item data\\en\\bpe_cache.pt")
    print(f"{'='*62}")

# ================================================================
# MANIFEST — Remplace data.txt (pas de copie, économise 50% espace)
# ================================================================

def maj_manifest(langue=None):
    """Met à jour data/{langue}/manifest.json sans copier les fichiers sources."""
    if not langue:
        print("\nLangue à mettre à jour ?")
        print("  [1] en (anglais)  [2] fr (français)  [3] multi")
        c = input("  > ").strip()
        langue = {"1": "en", "2": "fr", "3": "multi"}.get(c, "en")

    sources_dir   = os.path.join(DATA_ROOT, langue, "sources")
    manifest_path = os.path.join(DATA_ROOT, langue, "manifest.json")
    fichiers      = lister_sources(langue)

    if not fichiers:
        print(f"\n  (aucune source pour '{langue}' — manifest non créé)")
        return

    sources_manifest = []
    total_mo = 0.0
    for f in fichiers:
        chemin = os.path.join(sources_dir, f)
        mo = round(os.path.getsize(chemin) / 1_000_000, 1)
        total_mo += mo
        sources_manifest.append({"fichier": f, "mo": mo, "poids": 1})

    manifest = {
        "langue"      : langue,
        "sources"     : sources_manifest,
        "total_mo"    : round(total_mo, 1),
        "derniere_maj": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"  ✅ manifest.json ({langue}) mis à jour — {total_mo:.1f} Mo "
          f"({len(fichiers)} source(s))")
    print(f"     → {manifest_path}")

def _maj_manifests(cles_sources):
    """Met à jour les manifests pour toutes les langues concernées (post-téléchargement auto)."""
    print("  Téléchargements terminés — mise à jour des manifests...\n")
    langues = set(get_langue_by_cat(SOURCES[cle]["cat"]) for cle in cles_sources)
    for lang in langues:
        maj_manifest(lang)
    print()

def _creer_flag_reload(langue):
    """Crée le fichier signal pour que nanogpt_bpe.py recharge les données à chaud."""
    try:
        flag = os.path.join(DATA_ROOT, langue, "reload_requested.flag")
        open(flag, 'w').close()
    except Exception:
        pass

# ================================================================
# AJOUT SOURCE CUSTOM
# ================================================================

def ajouter_source_custom():
    """Guide interactif pour ajouter une source HuggingFace dans custom_sources.json."""
    print("\n  ── Ajouter une source HuggingFace personnalisée ──")
    print("  (Sauvegardée dans src/custom_sources.json)")
    print()
    hf_path = input("  Path HuggingFace (ex: mon-user/mon-dataset) > ").strip()
    if not hf_path:
        print("  Annulé.")
        return
    nom     = input(f"  Nom affiché [{hf_path}] > ").strip() or hf_path
    cle_id  = input("  Identifiant court (ex: ma_source) > ").strip()
    if not cle_id:
        cle_id = re.sub(r'[^a-z0-9]+', '_', hf_path.lower()).strip('_')

    print("\n  Catégorie :")
    cats = list(CATEGORIES.keys())
    for i, cat in enumerate(cats, 1):
        print(f"    [{i}] {CATEGORIES[cat]}")
    try:
        idx_c = int(input("  Choix [1] > ").strip() or "1") - 1
        cat   = cats[idx_c] if 0 <= idx_c < len(cats) else "fr"
    except Exception:
        cat = "fr"

    hf_config  = input("  Config HF (vide si aucune) > ").strip() or None
    hf_split   = input("  Split [train] > ").strip() or "train"
    champ      = input("  Champ texte [text] > ").strip() or "text"
    try:
        taille_max = int(input("  Taille max estimée en Mo [500] > ").strip() or "500")
    except Exception:
        taille_max = 500

    nouvelle_src = {
        "id"          : cle_id,
        "nom"         : nom,
        "cat"         : cat,
        "type"        : "hf",
        "hf_path"     : hf_path,
        "hf_config"   : hf_config,
        "hf_split"    : hf_split,
        "champ"       : champ,
        "taille_max_mo": taille_max,
    }

    # Charger / créer custom_sources.json
    if os.path.exists(_CUSTOM_SOURCES_FILE):
        try:
            with open(_CUSTOM_SOURCES_FILE, 'r', encoding='utf-8') as f:
                data_custom = json.load(f)
        except Exception:
            data_custom = {"sources_custom": []}
    else:
        data_custom = {"sources_custom": []}
    data_custom["sources_custom"].append(nouvelle_src)
    with open(_CUSTOM_SOURCES_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_custom, f, indent=2, ensure_ascii=False)

    # Ajouter en mémoire
    src_mem = {k: v for k, v in nouvelle_src.items() if k != "id"}
    src_mem["_custom"] = True
    SOURCES[cle_id] = src_mem

    print(f"\n  ✅ Source '{nom}' ajoutée [CUSTOM] !")
    print(f"     Catégorie : {CATEGORIES.get(cat, cat)}")
    print(f"     Identifiant : {cle_id}")

# ================================================================
# DÉTECTION LANGUE + NETTOYAGE
# ================================================================
MOTS_FR = {"le","la","les","de","du","des","et","en","un","une",
            "que","est","je","il","nous","vous","dans","sur","avec"}
MOTS_EN = {"the","of","and","to","in","is","it","that","was","for",
            "are","with","his","they","this","have","from","but"}

def detecter_langue(texte):
    """Détection : langdetect si disponible (~95% précision), sinon mots-clés."""
    if not texte:
        return "autre"
    # Filtre non-latin rapide (chinois, arabe, etc.)
    echantillon = texte[:500]
    nb_latin = sum(1 for c in echantillon if ord(c) < 0x0250)
    if nb_latin / len(echantillon) < 0.75:
        return "autre"
    # Détection via langdetect si disponible
    if _HAS_LANGDETECT and len(texte) >= 50:
        try:
            lang = _langdetect_detect(texte[:2000])
            if lang == "fr": return "fr"
            if lang == "en": return "en"
            return "autre"
        except Exception:
            pass
    # Fallback : mots-clés
    mots = set(texte.lower().split()[:80])
    score_fr = len(mots & MOTS_FR)
    score_en = len(mots & MOTS_EN)
    if score_fr >= score_en:
        return "fr"
    return "en"

def nettoyer_texte(texte):
    """
    Nettoyage qualité Common Crawl — plusieurs passes :
    1. Supprime les balises HTML résiduelles
    2. Filtre ligne par ligne : longueur, densité de ponctuation, URLs
    3. Déduplique les lignes (spam répétitif)
    4. Retire les caractères non-latins
    """
    import re as _re

    # 1. Balises HTML (< ... >) et entités (&amp; &nbsp; etc.)
    texte = _re.sub(r'<[^>]{1,200}>', ' ', texte)
    texte = _re.sub(r'&[a-z]{2,6};', ' ', texte)

    # 2. Filtrage ligne par ligne
    _RE_URL   = _re.compile(r'https?://\S+|www\.\S+', _re.I)
    _RE_EMAIL = _re.compile(r'\S+@\S+\.\S+')
    _RE_PONCT = _re.compile(r'[|{}\[\]<>#*_=~^]')

    lignes_ok  = []
    vues       = set()

    for ligne in texte.splitlines():
        ligne = ligne.strip()

        # Trop courte (menu, bouton, navigation)
        if len(ligne) < 40:
            continue

        # Trop d'URLs → page de liens / pub
        if len(_RE_URL.findall(ligne)) >= 2:
            continue

        # Contient un email → RGPD / spam
        if _RE_EMAIL.search(ligne):
            continue

        # Densité de ponctuation parasite > 15 % → HTML mal parsé
        nb_ponct = len(_RE_PONCT.findall(ligne))
        if nb_ponct / len(ligne) > 0.15:
            continue

        # Déduplique (lignes répétées = boilerplate / SEO spam)
        cle = ligne[:80].lower()
        if cle in vues:
            continue
        vues.add(cle)

        lignes_ok.append(ligne)

    texte = "\n".join(lignes_ok)

    # 3. Retire les caractères non-latins (garde \n \t)
    texte = "".join(c for c in texte if ord(c) < 0x0250 or c in "\n\t")

    return texte

# ================================================================
# COMMON CRAWL — TÉLÉCHARGEMENT + EXTRACTION (STREAMING)
# ================================================================

def get_liste_wet():
    url = f"https://data.commoncrawl.org/crawl-data/{CRAWL_ID}/wet.paths.gz"
    print("  Récupération de la liste des fichiers...")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    contenu = gzip.decompress(r.content).decode("utf-8")
    chemins = [l.strip() for l in contenu.splitlines() if l.strip()]
    print(f"  {len(chemins):,} fichiers WET disponibles")
    return chemins

def lire_wet_stream(chemin_local, filtre_langue=None, limite_chars=None, numero_bot=0):
    """Extraction en streaming — ligne par ligne, pas tout en RAM."""
    textes        = []
    chars_cumules = 0
    pages_gardees = 0
    nb_blocs      = 0
    corps_lignes  = []
    in_body       = False
    is_conversion = False
    headers_done  = False

    try:
        with gzip.open(chemin_local, "rt", encoding="utf-8", errors="ignore") as gz:
            for ligne in gz:
                if stop_event.is_set():
                    break
                if limite_chars and chars_cumules >= limite_chars:
                    break

                if ligne.startswith("WARC/1.0"):
                    if corps_lignes and is_conversion:
                        corps = "".join(corps_lignes).strip()
                        if len(corps) >= MIN_LEN:
                            if not filtre_langue or detecter_langue(corps) in filtre_langue:
                                corps = nettoyer_texte(corps)
                                if len(corps) >= MIN_LEN:
                                    textes.append(corps)
                                    chars_cumules += len(corps)
                                    pages_gardees += 1

                    corps_lignes  = []
                    in_body       = False
                    is_conversion = False
                    headers_done  = False
                    nb_blocs     += 1

                    if nb_blocs % 2000 == 0:
                        print(f"  ⏳ Bot {numero_bot} : {nb_blocs:,} blocs — "
                              f"{pages_gardees:,} pages — "
                              f"{chars_cumules/1_000_000:.1f} Mo")

                elif not headers_done:
                    if ligne.strip() == "":
                        headers_done = True
                        in_body      = True
                    elif "WARC-Type: conversion" in ligne:
                        is_conversion = True
                elif in_body:
                    corps_lignes.append(ligne)

    except Exception as e:
        print(f"\n  ❌ Bot {numero_bot} erreur extraction : {e}")

    return textes

def bot_cc(args):
    """Worker : télécharge ET extrait un fichier WET."""
    chemin, numero, total, filtre, limite_chars = args
    if stop_event.is_set():
        return []

    url          = f"https://data.commoncrawl.org/{chemin}"
    nom_fichier  = os.path.basename(chemin)
    chemin_local = os.path.join(TEMP_DIR, nom_fichier)

    # Téléchargement
    if not os.path.exists(chemin_local):
        print(f"  🤖 Bot {numero}/{total} → téléchargement...")
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
            debut = time.time()
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total_octets = 0
            with open(chemin_local, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if stop_event.is_set():
                        break
                    f.write(chunk)
                    total_octets += len(chunk)
            duree   = time.time() - debut
            vitesse = total_octets / 1_000_000 / duree if duree > 0 else 0
            print(f"  ✅ Bot {numero} : {total_octets/1_000_000:.1f} Mo "
                  f"en {duree:.1f}s ({vitesse:.1f} Mo/s)")
        except Exception as e:
            print(f"  ❌ Bot {numero} erreur téléchargement : {e}")
            if os.path.exists(chemin_local):
                os.remove(chemin_local)
            return []
    else:
        print(f"  ⏩ Bot {numero} : déjà présent, skip téléchargement")

    if stop_event.is_set():
        return []

    # Extraction streaming
    print(f"  📄 Bot {numero} → extraction...")
    textes = lire_wet_stream(chemin_local, filtre_langue=filtre,
                             limite_chars=limite_chars, numero_bot=numero)
    print(f"  ✅ Bot {numero} : {len(textes):,} pages extraites")

    # Suppression immédiate
    if os.path.exists(chemin_local):
        mb = os.path.getsize(chemin_local) / 1_000_000
        os.remove(chemin_local)
        print(f"  🗑️  Bot {numero} : supprimé ({mb:.1f} Mo libérés)")

    return textes

def telecharger_common_crawl(output_file, filtre, limite_chars, mode="w"):
    global stop_event
    stop_event = threading.Event()
    total_chars = 0
    total_pages = 0

    try:
        chemins = get_liste_wet()
    except Exception as e:
        print(f"❌ Impossible de récupérer la liste : {e}")
        return False

    nb_fichiers = max(1, int(limite_chars / 800_000_000) + 2)
    chemins = chemins[:nb_fichiers]
    print(f"  → {len(chemins)} fichier(s) WET à traiter")
    print(f"\n🤖 {NB_WORKERS} bots en parallèle\n")

    args_list = [(c, i+1, len(chemins), filtre, limite_chars)
                 for i, c in enumerate(chemins)]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode, encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=NB_WORKERS) as executor:
            futures = {executor.submit(bot_cc, a): a for a in args_list}
            for future in as_completed(futures):
                if stop_event.is_set():
                    break
                textes = future.result()
                if not textes:
                    continue
                with write_lock:
                    for texte in textes:
                        if total_chars >= limite_chars:
                            stop_event.set()
                            break
                        f_out.write(texte + "\n\n")
                        total_chars += len(texte)
                        total_pages += 1
                print(f"  📊 Progression : {total_chars/1_000_000:.1f} / "
                      f"{limite_chars/1_000_000:.0f} Mo  ({total_pages:,} pages)")
                if stop_event.is_set():
                    print(f"\n  ✅ {total_chars/1_000_000:.1f} Mo atteints !")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    return total_chars > 0

# ================================================================
# HUGGINGFACE — TÉLÉCHARGEMENT
# ================================================================

def _lire_hf_token():
    """Lit le token HuggingFace depuis env ou system/hf_token.txt."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    token_file = os.path.join(_root, "system", "hf_token.txt")
    if os.path.isfile(token_file):
        with open(token_file, "r", encoding="utf-8") as f:
            t = f.read().strip()
            if t:
                return t
    return None


def _charger_dataset_hf(src, tentative=1, max_tentatives=4):
    """Charge un dataset HuggingFace avec retry exponentiel + auto-réparation."""
    # Vérification / réparation du module datasets
    try:
        from datasets import load_dataset
    except ImportError:
        log("ERROR", "Module 'datasets' absent — tentative de réinstallation...", source="telecharger")
        repare = auto_repair("datasets", "datasets")
        if not repare:
            log("FATAL", "Impossible de réinstaller 'datasets'. Téléchargement annulé.", source="telecharger")
            return None
        try:
            from datasets import load_dataset
        except ImportError:
            log("FATAL", "'datasets' toujours non importable après réinstallation.", source="telecharger")
            return None
    except Exception as e:
        log("ERROR", f"Import 'datasets' corrompu : {e} — réinstallation...", source="telecharger")
        repare = auto_repair("datasets", "datasets")
        if not repare:
            return None
        from datasets import load_dataset

    hf_token = _lire_hf_token()
    kwargs = {"split": src["hf_split"], "streaming": True}
    if hf_token:
        kwargs["token"] = hf_token

    delais = [0, 3, 8, 20]
    for i in range(tentative - 1, max_tentatives):
        if i > 0:
            attente = delais[min(i, len(delais)-1)]
            print(f"  Retry {i}/{max_tentatives-1} dans {attente}s...")
            log("WARNING", f"Retry {i} sur '{src['hf_path']}' dans {attente}s", source="telecharger")
            time.sleep(attente)
        try:
            if src.get("hf_config"):
                return load_dataset(src["hf_path"], src["hf_config"], **kwargs)
            else:
                return load_dataset(src["hf_path"], **kwargs)
        except Exception as e:
            log("WARNING", f"Tentative {i+1}/{max_tentatives} échouée sur '{src['hf_path']}' : {e}", source="telecharger")
            print(f"  Tentative {i+1}/{max_tentatives} échouée : {e}")
    log("ERROR", f"Toutes les tentatives ont échoué pour '{src['hf_path']}'", source="telecharger")
    return None


def _extraire_texte(exemple, src):
    """Extrait et normalise le texte d'un exemple HF."""
    texte = exemple.get(src["champ"], "")
    if isinstance(texte, dict):
        texte = "\n".join(str(v) for v in texte.values() if v)
    if isinstance(texte, list):
        texte = "\n".join(str(t) for t in texte if t)
    if src.get("champ2"):
        texte2 = exemple.get(src["champ2"], "")
        if isinstance(texte2, list):
            texte2 = "\n".join(str(t) for t in texte2 if t)
        texte = f"{texte}\n{texte2}".strip()
    return texte


def telecharger_hf(src, output_file, limite_chars, mode="w"):
    log("INFO", f"Début téléchargement : {src['nom']} ({limite_chars//1_000_000} Mo max)", source="telecharger")
    print("  Connexion à HuggingFace...")
    ds = _charger_dataset_hf(src)
    if ds is None:
        print("❌ Impossible de charger le dataset après plusieurs tentatives.")
        return False

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    total_chars    = 0
    total_articles = 0
    debut          = time.time()
    MAX_RETRY_ITER = 3   # retries en cas d'erreur réseau pendant l'itération

    print(f"  Écriture dans {output_file}...")
    with open(output_file, mode, encoding="utf-8") as f:
        iter_retry = 0
        it = iter(ds)
        while True:
            try:
                exemple = next(it)
                iter_retry = 0  # reset compteur après un succès
            except StopIteration:
                break
            except Exception as e:
                msg = str(e)
                if iter_retry < MAX_RETRY_ITER and (
                    "10054" in msg or "client" in msg.lower() or
                    "connection" in msg.lower() or "reset" in msg.lower()
                ):
                    iter_retry += 1
                    attente = iter_retry * 5
                    print(f"  ⚠️  Erreur réseau ({msg[:60]}) — retry {iter_retry}/{MAX_RETRY_ITER} dans {attente}s...")
                    time.sleep(attente)
                    # Recharge le dataset depuis le début (streaming ne supporte pas seek)
                    ds2 = _charger_dataset_hf(src)
                    if ds2 is None:
                        break
                    it = iter(ds2)
                    # Skip les articles déjà traités
                    print(f"  ↩️  Reprise — skip {total_articles:,} articles déjà écrits...")
                    for _ in range(total_articles):
                        try:
                            next(it)
                        except StopIteration:
                            break
                    continue
                else:
                    print(f"  ❌ Erreur itération : {e}")
                    break

            texte = _extraire_texte(exemple, src)
            if not texte or len(texte) < 20:
                continue
            texte = nettoyer_texte(texte)
            if not texte:
                continue
            # Filtre langue si configuré (ex: oasst2 contient FR+EN+DE...)
            langue_cible = src.get("langue_cible")
            if langue_cible and detecter_langue(texte) != langue_cible:
                continue
            f.write(texte + "\n\n")
            total_chars    += len(texte)
            total_articles += 1
            if total_chars >= limite_chars:
                break
            if total_articles % 5000 == 0:
                duree = time.time() - debut
                print(f"  ⏳ {total_articles:,} articles — "
                      f"{total_chars/1_000_000:.1f} Mo — {duree:.0f}s")

    mo = total_chars / 1_000_000
    if total_chars > 0:
        print(f"  {total_articles:,} articles — {mo:.1f} Mo ecrits")
        log_telechargement(
            source_key=src.get("hf_path", "?"),
            nom=src["nom"],
            mo=mo,
            ok=True,
            detail=f"{total_articles} articles"
        )
    else:
        print("  Aucun texte recupere.")
        log_telechargement(
            source_key=src.get("hf_path", "?"),
            nom=src["nom"],
            mo=0,
            ok=False,
            detail="0 articles recuperes"
        )
    return total_chars > 0

# ================================================================
# MENU PRINCIPAL (LA BIBLIOTHÈQUE)
# ================================================================

def afficher_categories():
    print("\n" + "="*62)
    print("  📚 BIBLIOTHÈQUE DE DONNÉES")
    print("="*62)
    print("\n  Choisis une catégorie :\n")
    cats = list(CATEGORIES.keys())
    for i, cat in enumerate(cats, 1):
        print(f"  [{i}] {CATEGORIES[cat]}")
    print("\n  ── Gestion ──")
    print("  [ a]  🤖 Bot automatique (détecte l'espace disque, propose un preset)")
    print("  [ +]  Ajouter une source HuggingFace personnalisée")
    print("  [ c]  Combiner les sources (choisir proportions)")
    print("  [ l]  Lister les sources téléchargées")
    print("  [ s]  Supprimer des données (sources, cache, tout)")
    print("="*62)
    return cats

def demander_mo():
    print("\n  Combien de données veux-tu télécharger ?")
    print("  Écris la taille avec 'Mo' (Megaoctets) ou 'Go' (Gigaoctets).")
    print("  Exemples : '500 Mo', '2 Go', '50 Mo'")
    
    while True:
        choix = input("  Taille > ").strip().lower()
        if not choix:
            return 200_000_000 # 200 Mo par defaut
        
        try:
            if "go" in choix or "g" in choix:
                val = float(choix.replace("go", "").replace("g", "").strip())
                return int(val * 1_000_000_000)
            elif "mo" in choix or "m" in choix:
                val = float(choix.replace("mo", "").replace("m", "").strip())
                return int(val * 1_000_000)
            else:
                val = float(choix)
                # S'il ne met pas d'unité, on assume Mo
                return int(val * 1_000_000)
        except:
            print("  ❌ Format invalide. Utilise 'Mo' ou 'Go' (ex: 2 Go).")

def supprimer_donnees():
    """Supprime des sources téléchargées (ou le data.txt combiné) par langue."""
    print("\n  ── Suppression de données ──")
    langues_dispo = []
    for lang in ["en", "fr", "multi"]:
        fichiers = lister_sources(lang)
        data_txt = os.path.join(DATA_ROOT, lang, "data.txt")
        if fichiers or os.path.exists(data_txt):
            langues_dispo.append(lang)

    if not langues_dispo:
        print("  (aucune donnée à supprimer)")
        return

    for lang in langues_dispo:
        print(f"\n  {FLAG.get(lang,'')} {lang.upper()} :")
        afficher_sources_dispo(lang)
        data_txt = os.path.join(DATA_ROOT, lang, "data.txt")
        if os.path.exists(data_txt):
            mo = os.path.getsize(data_txt) / 1_000_000
            print(f"    [data.txt] {'(combiné)':<40} {mo:.1f} Mo")

    lang = input("\n  Langue à nettoyer (en/fr/multi) > ").strip().lower()
    if lang not in langues_dispo:
        print("  ❌ Langue invalide.")
        return

    fichiers = lister_sources(lang)
    data_txt = os.path.join(DATA_ROOT, lang, "data.txt")
    print(f"\n  Que supprimer pour {lang.upper()} ?")
    print("    [t] tout (sources + data.txt)")
    print("    [d] seulement data.txt combiné")
    print("    [numéro] une source précise")
    cible = input("  Choix > ").strip().lower()

    libere = 0.0
    sources_dir = os.path.join(DATA_ROOT, lang, "sources")

    if cible == "t":
        for f in fichiers:
            chemin = os.path.join(sources_dir, f)
            libere += os.path.getsize(chemin) / 1_000_000
            os.remove(chemin)
        if os.path.exists(data_txt):
            libere += os.path.getsize(data_txt) / 1_000_000
            os.remove(data_txt)
        print(f"  🗑️  Tout supprimé pour {lang.upper()} ({libere:.1f} Mo libérés)")
    elif cible == "d":
        if os.path.exists(data_txt):
            libere = os.path.getsize(data_txt) / 1_000_000
            os.remove(data_txt)
            print(f"  🗑️  data.txt supprimé ({libere:.1f} Mo libérés)")
        else:
            print("  (aucun data.txt à supprimer)")
    else:
        try:
            idx = int(cible) - 1
            if 0 <= idx < len(fichiers):
                chemin = os.path.join(sources_dir, fichiers[idx])
                libere = os.path.getsize(chemin) / 1_000_000
                os.remove(chemin)
                print(f"  🗑️  {fichiers[idx]} supprimé ({libere:.1f} Mo libérés)")
            else:
                print("  ❌ Numéro invalide.")
        except ValueError:
            print("  ❌ Choix invalide.")

# ================================================================
# AUTO-BOT — Téléchargement automatique intelligent
# ================================================================

BOT_PRESETS = [
    ("nano",   "Nano   — ~100 Mo",  "test rapide, idéal pour ~20M params",   100),
    ("small",  "Small  — ~300 Mo",  "bon départ, ~50-100M params",            300),
    ("medium", "Medium — ~700 Mo",  "bonne qualité, ~100-300M params",        700),
    ("large",  "Large  — ~1.5 Go",  "haute qualité, ~300M+ params",          1500),
]

# Nb params → Mo de données recommandés
PARAM_VERS_MO = [
    (20,   150,  "20M"),
    (50,   300,  "50M"),
    (100,  600,  "100M"),
    (300, 1200,  "300M"),
    (999, 2000,  "1B+"),
]

# Sources choisies automatiquement par (type_ia, langue_bot)
# langue_bot : "fr" | "en" | "multi"
# Le bot répartit le budget Mo également entre les sources
TYPE_SOURCES = {
    ("general",   "fr"):    ["fr_wiki", "fr_mc4", "fr_gutenberg"],
    ("general",   "en"):    ["en_fineweb", "en_finewebedu", "en_wiki"],
    ("general",   "multi"): ["fr_wiki", "fr_mc4", "en_fineweb", "en_finewebedu"],

    ("code",      "fr"):    ["code_search", "code_stackoverflow", "fr_wiki"],
    ("code",      "en"):    ["code_search", "code_stackoverflow", "en_openweb"],
    ("code",      "multi"): ["code_search", "code_stackoverflow", "fr_wiki"],

    ("science",   "fr"):    ["sci_openwebmath", "sci_arxiv", "fr_wiki"],
    ("science",   "en"):    ["sci_openwebmath", "sci_proofpile", "sci_arxiv"],
    ("science",   "multi"): ["sci_openwebmath", "sci_arxiv", "fr_wiki", "en_fineweb"],

    ("chat",      "fr"):    ["dial_daily", "dial_movies", "fr_wiki"],
    ("chat",      "en"):    ["dial_daily", "dial_movies", "dial_ubuntu", "dial_reddit"],
    ("chat",      "multi"): ["dial_daily", "dial_movies", "fr_wiki", "en_wiki"],

    ("assistant", "fr"):    ["inst_alpaca", "inst_squad_fr", "inst_tinystories", "fr_wiki"],
    ("assistant", "en"):    ["inst_hermes", "inst_alpaca", "inst_dolly", "inst_tinystories"],
    ("assistant", "multi"): ["inst_hermes", "inst_alpaca", "inst_squad_fr", "fr_wiki"],
}

LANGUES_BOT = [
    ("fr",    "\U0001f1eb\U0001f1f7 Francais"),
    ("en",    "\U0001f1ec\U0001f1e7 Anglais"),
    ("multi", "\U0001f30d Multilingue (FR + EN melanges)"),
]

TYPES_IA = [
    ("general",   "General          - texte varie, encyclopedique"),
    ("code",      "Code             - Python, JavaScript, StackOverflow"),
    ("science",   "Sciences & Maths - ArXiv, formules, demonstrations"),
    ("chat",      "Chat             - dialogues, conversations quotidiennes"),
    ("assistant", "Assistant        - instructions, questions / reponses"),
]


def _demander_liste(titre, options, defaut=0):
    """Affiche une liste numerotee et retourne l'index choisi.
    defaut=None → choix obligatoire (Entree sans valeur refusee).
    """
    print(f"\n  {titre}\n")
    for i, (_, lbl) in enumerate(options):
        marker = " (defaut)" if i == defaut else ""
        print(f"  [{i+1}]  {lbl}{marker}")
    print()
    if defaut is None:
        prompt = f"  Choix [1-{len(options)}] > "
    else:
        prompt = f"  Choix [1-{len(options)}] (Entree = {defaut+1}) > "
    while True:
        raw = input(prompt).strip()
        if not raw:
            if defaut is not None:
                return defaut
            print("  Choix obligatoire — tape un numero.")
            continue
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("  Choix invalide, reessaie.")


def _combiner_et_copier(cles_sources):
    """Combine les sources telechargees et copie dans data/data.txt."""
    print("  Telechargements termines - combinaison des sources...\n")

    langues_utilisees = set(get_langue_by_cat(SOURCES[cle]["cat"]) for cle in cles_sources)

    for lang_comb in langues_utilisees:
        sources_dir  = os.path.join(DATA_ROOT, lang_comb, "sources")
        output_final = os.path.join(DATA_ROOT, lang_comb, "data.txt")
        fichiers     = lister_sources(lang_comb)
        if not fichiers:
            continue
        with open(output_final, "w", encoding="utf-8") as out:
            for f in fichiers:
                chemin = os.path.join(sources_dir, f)
                with open(chemin, encoding="utf-8", errors="ignore") as sf:
                    out.write(sf.read())
        mo_final = os.path.getsize(output_final) / 1_000_000
        print(f"  {lang_comb}/data.txt : {mo_final:.1f} Mo")

    for prio in ["fr", "en", "multi"]:
        src_path = os.path.join(DATA_ROOT, prio, "data.txt")
        if os.path.exists(src_path) and os.path.getsize(src_path) > 1000:
            dst = os.path.join(DATA_ROOT, "data.txt")
            shutil.copy2(src_path, dst)
            print(f"\n  data/data.txt mis a jour ({os.path.getsize(dst)/1_000_000:.1f} Mo)"
                  f" - pret pour l'entrainement !")
            break
    print()


def _telecharger_plan(sources_plan):
    """Telecharge une liste [(cle_source, mo), ...]."""
    for i, (cle, mo) in enumerate(sources_plan, 1):
        src         = SOURCES[cle]
        langue_src  = get_langue_by_cat(src["cat"])
        sources_dir = os.path.join(DATA_ROOT, langue_src, "sources")
        os.makedirs(sources_dir, exist_ok=True)
        slug        = slug_source(src["nom"])
        output_file = os.path.join(sources_dir, f"{slug}.txt")

        print(f"  [{i}/{len(sources_plan)}] {src['nom']}  (~{mo} Mo)")

        if src.get("gated"):
            print("  (!) Source protegee (compte HuggingFace requis) — ignoree par le bot.")
            print("      Lance './wish telecharger' pour la telecharger manuellement.")
            continue

        if src["type"] == "cc":
            ok = telecharger_common_crawl(output_file, src.get("filtre"), mo * 1_000_000, "w")
        else:
            ok = telecharger_hf(src, output_file, mo * 1_000_000, "w")
        if ok:
            _creer_flag_reload(langue_src)
        print()


def auto_bot():
    """Bot de telechargement : preset OU configuration par nb de parametres."""

    print("\n" + "="*62)
    print("  BOT DE TELECHARGEMENT AUTOMATIQUE")
    print("="*62 + "\n")

    # Espace disque
    try:
        usage    = shutil.disk_usage(os.path.abspath("."))
        total_mo = usage.total / 1_000_000
        libre_mo = usage.free  / 1_000_000
    except Exception:
        total_mo = 10_000
        libre_mo = 2000
    plafond = min(total_mo * 0.30, 2000)   # 30 % de la capacite totale, max 2 Go

    print(f"  Capacite totale du disque : {total_mo:,.0f} Mo")
    print(f"  Espace libre              : {libre_mo:,.0f} Mo")
    print(f"  Plafond bot (30 %)        : {plafond:.0f} Mo (max 2 Go)\n")

    if libre_mo < 80:
        print("  Moins de 100 Mo libres - libere de l'espace et reessaie.")
        return

    # Presets disponibles
    presets_dispo = [(k, lbl, desc, mo) for k, lbl, desc, mo in BOT_PRESETS if mo <= plafond]

    print("  Presets disponibles :\n")
    for i, (_, lbl, desc, _) in enumerate(presets_dispo):
        print(f"  [{i+1}]  {lbl:<22}  {desc}")
    print()

    while True:
        mode = input("  Utiliser un preset ? [o/n] > ").strip().lower()
        if mode in ("o", "n", ""):
            break

    if mode == "n":
        # MODE CONFIGURATION MANUELLE
        print("\n  -- Taille du modele --\n")
        for i, (_, mo_rec, label) in enumerate(PARAM_VERS_MO):
            print(f"  [{i+1}]  {label:<8}  -> ~{mo_rec} Mo de donnees recommandes")
        print(f"  [{len(PARAM_VERS_MO)+1}]  Autre (saisie libre)\n")

        while True:
            raw = input(f"  Nb params [1-{len(PARAM_VERS_MO)+1}] (Entree = 1) > ").strip()
            if not raw:
                idx_p = 0; break
            try:
                idx_p = int(raw) - 1
                if 0 <= idx_p <= len(PARAM_VERS_MO):
                    break
            except ValueError:
                pass
            print("  Choix invalide.")

        if idx_p == len(PARAM_VERS_MO):
            while True:
                raw = input("  Nombre de parametres (ex: 200M, 500M, 1B) > ").strip().upper()
                try:
                    mul = 1000 if raw.endswith("B") else 1
                    val = float(raw.rstrip("MB").rstrip("G")) * mul
                    mo_recommande = min(int(val * 6), 2000)
                    break
                except Exception:
                    print("  Format invalide (ex: 100M, 1B).")
        else:
            _, mo_recommande, label_p = PARAM_VERS_MO[idx_p]
            print(f"\n  -> {label_p} params - donnees recommandees : ~{mo_recommande} Mo")

        mo_max_safe   = min(int(plafond), int(libre_mo * 0.90))  # ne pas depasser l'espace libre
        mo_recommande = min(mo_recommande, mo_max_safe)

        print(f"\n  -- Quantite de donnees --\n")
        print(f"  Recommande : {mo_recommande} Mo   |   Plafond (30% disque) : {int(plafond)} Mo   |   Libre : {int(libre_mo)} Mo\n")
        while True:
            raw = input(f"  Mo maximum [Entree = {mo_recommande}] > ").strip()
            if not raw:
                mo_total = mo_recommande; break
            try:
                mo_total = int(raw)
                if 10 <= mo_total <= mo_max_safe:
                    break
                print(f"  Entre 10 et {mo_max_safe} Mo (espace disponible).")
            except ValueError:
                print("  Nombre entier attendu.")

        idx_l  = _demander_liste("Langue des donnees :", LANGUES_BOT, defaut=None)
        langue = LANGUES_BOT[idx_l][0]
        print(f"  -> {LANGUES_BOT[idx_l][1]}\n")

        idx_t   = _demander_liste("Type d'intelligence artificielle :", TYPES_IA, defaut=0)
        type_ia = TYPES_IA[idx_t][0]
        print(f"  -> {TYPES_IA[idx_t][1]}\n")

        cles = TYPE_SOURCES.get((type_ia, langue), TYPE_SOURCES.get(("general", langue), ["fr_wiki"]))
        mo_par_source = max(10, mo_total // len(cles))
        sources_plan  = [(cle, mo_par_source) for cle in cles]

    else:
        # MODE PRESET
        while True:
            raw = input(f"  Preset [1-{len(presets_dispo)}] (Entree = 1) > ").strip()
            if not raw:
                idx_p = 0; break
            try:
                idx_p = int(raw) - 1
                if 0 <= idx_p < len(presets_dispo):
                    break
            except ValueError:
                pass
            print("  Choix invalide, reessaie.")

        _, preset_lbl, _, mo_total = presets_dispo[idx_p]
        print(f"\n  -> Preset : {preset_lbl}\n")

        idx_l  = _demander_liste("Langue des donnees :", LANGUES_BOT, defaut=None)
        langue = LANGUES_BOT[idx_l][0]
        print(f"  -> {LANGUES_BOT[idx_l][1]}\n")

        idx_t   = _demander_liste("Type d'intelligence artificielle :", TYPES_IA, defaut=0)
        type_ia = TYPES_IA[idx_t][0]
        print(f"  -> {TYPES_IA[idx_t][1]}\n")

        cles = TYPE_SOURCES.get((type_ia, langue), TYPE_SOURCES.get(("general", langue), ["fr_wiki"]))
        mo_par_source = max(10, mo_total // len(cles))
        sources_plan  = [(cle, mo_par_source) for cle in cles]

    # Recapitulatif
    total_prevu = sum(mo for _, mo in sources_plan)
    print(f"  {'-'*58}")
    print(f"  Plan de telechargement :\n")
    for cle, mo in sources_plan:
        print(f"    - {SOURCES[cle]['nom']:<44} ~{mo} Mo")
    print(f"\n    Total prevu : ~{total_prevu} Mo")
    print(f"  {'-'*58}\n")

    while True:
        confirm = input("  Lancer le telechargement ? [o/n] > ").strip().lower()
        if confirm in ("o", "n"):
            break

    if confirm != "o":
        print("  Annule.")
        return

    print()
    _telecharger_plan(sources_plan)
    cles_dl = [cle for cle, _ in sources_plan]
    _maj_manifests(cles_dl)
    _combiner_et_copier(cles_dl)


def main():
    cats = afficher_categories()

    while True:
        choix = input("\n  Choix > ").strip().lower()

        if not choix:
            continue

        if choix == "a":
            auto_bot()
            return
        if choix == "+":
            ajouter_source_custom()
            return
        if choix == "c":
            combiner_sources()
            return
        if choix == "s":
            supprimer_donnees()
            return
        if choix == "l":
            print("\n  -- Sources telechargees --")
            for lang in ["en", "fr", "multi"]:
                fichiers = lister_sources(lang)
                if fichiers:
                    print(f"\n  {FLAG.get(lang,'')} {lang.upper()} :")
                    afficher_sources_dispo(lang)
            return
        if choix == "q":
            return

        try:
            idx = int(choix) - 1
            if not (0 <= idx < len(cats)):
                print("  Choix invalide, reessaie.")
                continue
            cat_choisie = cats[idx]
            break
        except ValueError:
            print("  Choix invalide, reessaie.")
            continue

    # Afficher les sources de la categorie
    print(f"\n{'='*62}")
    print(f"  {CATEGORIES[cat_choisie]}")
    print(f"{'='*62}\n")

    sources_cat = {k: v for k, v in SOURCES.items() if v["cat"] == cat_choisie}
    cles = list(sources_cat.keys())

    for i, cle in enumerate(cles, 1):
        src = sources_cat[cle]
        mo  = obtenir_taille_hf(src) if src["type"] == "hf" else src.get("taille_max_mo")
        if mo is None:
            taille_str = "illimitée"
        elif mo >= 1_000_000:
            taille_str = f"~{mo // 1_000_000} To"
        elif mo >= 1_000:
            taille_str = f"~{mo // 1_000} Go"
        else:
            taille_str = f"~{mo} Mo"
        badges = ""
        if src.get("gated"):  badges += " 🔒"
        if src.get("slow"):   badges += " 🐢"
        if src.get("_custom"):badges += " [CUSTOM]"
        typ_mo = src.get("taille_typique_mo")
        rec_str = f"  Typique: {typ_mo} Mo" if typ_mo else ""
        print(f"  [{i:>2}]  {src['nom']:<50} Max: {taille_str}{rec_str}{badges}")

    print("\n  [ q]  Retour")

    while True:
        choix_src = input("\n  Dataset > ").strip().lower()

        if not choix_src:
            continue
        if choix_src == "q":
            return

        try:
            idx_src = int(choix_src) - 1
            if not (0 <= idx_src < len(cles)):
                print("  Choix invalide, reessaie.")
                continue
            cle_finale = cles[idx_src]
            break
        except ValueError:
            print("  Choix invalide, reessaie.")
            continue

    src = SOURCES[cle_finale]

    if src.get("gated"):
        print("  Ce dataset necessite un compte HuggingFace.")
        if input("   Continuer quand meme ? [o/N] > ").strip().lower() != "o":
            return

    langue = get_langue_by_cat(src["cat"])
    limite = demander_mo()

    # Validation : cap à la taille max si connue
    taille_max = src.get("taille_max_mo")
    if taille_max and limite > taille_max * 1_000_000:
        print(f"\n  ⚠️  {src['nom']} ne contient que ~{taille_max} Mo.")
        print(f"     Téléchargement plafonné à {taille_max} Mo.")
        limite = taille_max * 1_000_000

    sources_dir = os.path.join(DATA_ROOT, langue, "sources")
    os.makedirs(sources_dir, exist_ok=True)
    slug        = slug_source(src["nom"])
    output_file = os.path.join(sources_dir, f"{slug}.txt")

    mode = "w"
    if os.path.exists(output_file):
        taille = os.path.getsize(output_file) / 1_000_000
        print(f"\n  Source '{slug}.txt' existe deja ({taille:.1f} Mo)")
        print("  [1] Ajouter  [2] Remplacer")
        choix_mode = input("  Choix > ").strip()
        mode = "w" if choix_mode == "2" else "a"

    ok = False
    if src["type"] == "cc":
        ok = telecharger_common_crawl(output_file, src.get("filtre"), limite, mode)
    else:
        ok = telecharger_hf(src, output_file, limite, mode)

    if ok:
        maj_manifest(langue)
        _creer_flag_reload(langue)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Telecharger des donnees pour WishAI")
    parser.add_argument("--download",  default=None,  help="ID de la source (ex: fr_culturax)")
    parser.add_argument("--mo",        type=int, default=200, help="Mo a telecharger")
    parser.add_argument("--combine",   action="store_true",   help="Combiner les sources apres dl")
    parser.add_argument("--lang",      default="multi")
    args = parser.parse_args()

    if args.download:
        if args.download not in SOURCES:
            print("Source inconnue : " + args.download)
            sys.exit(1)
        src    = SOURCES[args.download]
        langue = get_langue_by_cat(src["cat"])
        sources_dir  = os.path.join(DATA_ROOT, langue, "sources")
        os.makedirs(sources_dir, exist_ok=True)
        slug         = slug_source(src["nom"])
        output_file  = os.path.join(sources_dir, f"{slug}.txt")
        ok = telecharger_hf(src, output_file, args.mo * 1_000_000, "w")
        if ok and args.combine:
            combiner_sources(langue)
    else:
        main()
