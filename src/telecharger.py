"""\n=================================================================\n  TÉLÉCHARGEUR UNIFIÉ — WishAI by Liam\n  Toutes les sources de données au même endroit.\n=================================================================\n\nSTRUCTURE DES FICHIERS GÉNÉRÉS :\n  data/fr/data.txt     ← données françaises\n  data/en/data.txt     ← données anglaises\n  data/multi/data.txt  ← données multilingues\n  data/data.txt        ← dernier téléchargement (utilisé par nanogpt.py)\n\nSOURCES DISPONIBLES :\n  [1-8]  HuggingFace  (Wikipedia, Wikitext, OpenWebText, OSCAR...)\n  [9-11] Common Crawl (scraping direct du web, 3 bots en parallèle)\n=================================================================\n"""

import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import gzip
import time
import shutil
import threading
import unicodedata
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    "fr_wiki":    {"nom": "Wikipedia Français", "cat": "fr", "type": "hf", "hf_path": "wikimedia/wikipedia", "hf_config": "20231101.fr", "hf_split": "train", "champ": "text"},
    "fr_mc4":     {"nom": "mC4 Français (Texte web de Google)", "cat": "fr", "type": "hf", "hf_path": "mc4", "hf_config": "fr", "hf_split": "train", "champ": "text"},
    "fr_culturax":{"nom": "CulturaX (Haute qualité fr)", "cat": "fr", "type": "hf", "hf_path": "uonlp/CulturaX", "hf_config": "fr", "hf_split": "train", "champ": "text"},
    "fr_oscar":   {"nom": "OSCAR FR (Nécessite compte HF)", "cat": "fr", "type": "hf", "hf_path": "oscar-corpus/oscar", "hf_config": "unshuffled_deduplicated_fr", "hf_split": "train", "champ": "text", "gated": True},
    "fr_gutenberg":{"nom": "Project Gutenberg FR (Livres classiques)", "cat": "fr", "type": "hf", "hf_path": "graelo/gutenberg", "hf_config": "fr", "hf_split": "train", "champ": "text"},
    "fr_cc":      {"nom": "Common Crawl Français (Web brut, énorme)", "cat": "fr", "type": "cc", "filtre": ["fr"]},

    # 🇬🇧 ANGLAIS
    "en_fineweb": {"nom": "FineWeb (Le meilleur dataset web actuel)", "cat": "en", "type": "hf", "hf_path": "HuggingFaceFW/fineweb", "hf_config": "sample-10BT", "hf_split": "train", "champ": "text"},
    "en_finewebedu":{"nom": "FineWeb-Edu (Web éducatif, très haute qualité)", "cat": "en", "type": "hf", "hf_path": "HuggingFaceFW/fineweb-edu", "hf_config": "sample-10BT", "hf_split": "train", "champ": "text"},
    "en_wiki":    {"nom": "Wikipedia Anglais", "cat": "en", "type": "hf", "hf_path": "wikimedia/wikipedia", "hf_config": "20231101.en", "hf_split": "train", "champ": "text"},
    "en_wikitext":{"nom": "Wikitext-103 (Classique)", "cat": "en", "type": "hf", "hf_path": "Salesforce/wikitext", "hf_config": "wikitext-103-raw-v1", "hf_split": "train", "champ": "text"},
    "en_openweb": {"nom": "OpenWebText (Données de GPT-2)", "cat": "en", "type": "hf", "hf_path": "Skylion007/openwebtext", "hf_config": None, "hf_split": "train", "champ": "text"},
    "en_c4":      {"nom": "C4 (Texte web propre de Google)", "cat": "en", "type": "hf", "hf_path": "allenai/c4", "hf_config": "en", "hf_split": "train", "champ": "text"},
    "en_pile_hn": {"nom": "The Pile - HackerNews (Conversations tech)", "cat": "en", "type": "hf", "hf_path": "EleutherAI/pile", "hf_config": "hacker_news", "hf_split": "train", "champ": "text"},
    "en_pile_enron":{"nom": "The Pile - Enron Emails (Pour apprendre à écrire des emails)", "cat": "en", "type": "hf", "hf_path": "EleutherAI/pile", "hf_config": "enron_emails", "hf_split": "train", "champ": "text"},
    "en_cc":      {"nom": "Common Crawl Anglais (Web brut, énorme)", "cat": "en", "type": "cc", "filtre": ["en"]},

    # 💻 CODE
    "code_stack":   {"nom": "The Stack (Python) - Code propre", "cat": "code", "type": "hf", "hf_path": "bigcode/the-stack-smol", "hf_config": "data/python", "hf_split": "train", "champ": "content"},
    "code_stack_js":{"nom": "The Stack (JavaScript/HTML/CSS)", "cat": "code", "type": "hf", "hf_path": "bigcode/the-stack-smol", "hf_config": "data/javascript", "hf_split": "train", "champ": "content"},
    "code_search":  {"nom": "CodeSearchNet Python (Code + Docstrings)", "cat": "code", "type": "hf", "hf_path": "code_search_net", "hf_config": "python", "hf_split": "train", "champ": "whole_func_string"},
    "code_stackoverflow":{"nom": "StackExchange / StackOverflow Q&A", "cat": "code", "type": "hf", "hf_path": "flax-sentence-embeddings/stackexchange_title_body_jsonl", "hf_config": None, "hf_split": "train", "champ": "body"},
    "code_starcoder":{"nom": "StarCoder Data (Nécessite compte HF)", "cat": "code", "type": "hf", "hf_path": "bigcode/starcoderdata", "hf_config": "python", "hf_split": "train", "champ": "content", "gated": True},

    # 🔬 SCIENCES & MATHS
    "sci_openwebmath": {"nom": "OpenWebMath (Haut niveau mathématique)", "cat": "science", "type": "hf", "hf_path": "open-web-math/open-web-math", "hf_config": None, "hf_split": "train", "champ": "text"},
    "sci_proofpile":   {"nom": "Proof-Pile (Démonstrations mathématiques)", "cat": "science", "type": "hf", "hf_path": "hoskinson-center/proof-pile", "hf_config": None, "hf_split": "train", "champ": "text"},
    "sci_arxiv":       {"nom": "ArXiv (Articles scientifiques)", "cat": "science", "type": "hf", "hf_path": "CShorten/ML-ArXiv-Papers", "hf_config": None, "hf_split": "train", "champ": "abstract"},
    "sci_pubmed":      {"nom": "PubMed (Médecine et Biologie)", "cat": "science", "type": "hf", "hf_path": "pubmed", "hf_config": None, "hf_split": "train", "champ": "text"},
    "sci_phil":        {"nom": "PhilPapers (Philosophie)", "cat": "science", "type": "hf", "hf_path": "EleutherAI/pile", "hf_config": "philpapers", "hf_split": "train", "champ": "text"},

    # 🤖 INSTRUCTIONS
    "inst_hermes":  {"nom": "OpenHermes 2.5 (Excellent dataset d'instructions)", "cat": "instruct", "type": "hf", "hf_path": "teknium/OpenHermes-2.5", "hf_config": None, "hf_split": "train", "champ": "conversations"},
    "inst_alpaca":  {"nom": "Alpaca Clean (Instructions de base de Stanford)", "cat": "instruct", "type": "hf", "hf_path": "yahma/alpaca-cleaned", "hf_config": None, "hf_split": "train", "champ": "instruction", "champ2": "output"},
    "inst_dolly":   {"nom": "Dolly 15k (Instructions écrites par des humains)", "cat": "instruct", "type": "hf", "hf_path": "databricks/databricks-dolly-15k", "hf_config": None, "hf_split": "train", "champ": "instruction", "champ2": "response"},
    "inst_squad_fr":{"nom": "SQuAD Français (Compréhension de texte)", "cat": "instruct", "type": "hf", "hf_path": "pragnakalp/squad_v2_french_translated", "hf_config": None, "hf_split": "train", "champ": "context", "champ2": "question"},
    "inst_tinystories":{"nom": "TinyStories (Histoires courtes faciles, idéal débutants)", "cat": "instruct", "type": "hf", "hf_path": "roneneldan/TinyStories", "hf_config": None, "hf_split": "train", "champ": "text"},

    # 💬 DIALOGUES
    "dial_daily":   {"nom": "DailyDialog (Conversations quotidiennes)", "cat": "dialog", "type": "hf", "hf_path": "frankdarkluo/DailyDialog", "hf_config": None, "hf_split": "train", "champ": "context", "champ2": "response"},
    "dial_movies":  {"nom": "OpenSubtitles (Dialogues films/séries)", "cat": "dialog", "type": "hf", "hf_path": "frankdarkluo/OpenSubtitles", "hf_config": None, "hf_split": "train", "champ": "context", "champ2": "response"},
    "dial_ubuntu":  {"nom": "Ubuntu Dialogue Corpus (Chat technique)", "cat": "dialog", "type": "hf", "hf_path": "ubuntu_dialogs", "hf_config": None, "hf_split": "train", "champ": "Context", "champ2": "Utterance"},
    "dial_reddit":  {"nom": "Reddit (Discussions et réponses)", "cat": "dialog", "type": "hf", "hf_path": "webis/tldr-17", "hf_config": None, "hf_split": "train", "champ": "content"},
    "dial_persona": {"nom": "PersonaChat (Personnalités multiples)", "cat": "dialog", "type": "hf", "hf_path": "bavard/personachat_truecased", "hf_config": None, "hf_split": "train", "champ": "history", "champ2": "candidates"},

    # 🌍 MULTI
    "multi_cc":     {"nom": "Common Crawl Multilingue (Extrêmement massif)", "cat": "multi", "type": "cc", "filtre": None},
    "multi_madlad": {"nom": "MADLAD-400 (Google multilingue)", "cat": "multi", "type": "hf", "hf_path": "allenai/MADLAD-400", "hf_config": None, "hf_split": "train", "champ": "text", "gated": True},
}

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
# DÉTECTION LANGUE + NETTOYAGE
# ================================================================
MOTS_FR = {"le","la","les","de","du","des","et","en","un","une",
            "que","est","je","il","nous","vous","dans","sur","avec"}
MOTS_EN = {"the","of","and","to","in","is","it","that","was","for",
            "are","with","his","they","this","have","from","but"}

def detecter_langue(texte):
    """Détection fiable : ratio latin d'abord, puis mots-clés."""
    if not texte:
        return "autre"
    echantillon = texte[:500]
    nb_latin = sum(1 for c in echantillon if ord(c) < 0x0250)
    if nb_latin / len(echantillon) < 0.75:
        return "autre"  # chinois, arabe, etc.
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

def telecharger_hf(src, output_file, limite_chars, mode="w"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Module 'datasets' manquant. Lance : install_data.bat")
        return False

    print("  Connexion à HuggingFace...")
    try:
        if src.get("hf_config"):
            ds = load_dataset(src["hf_path"], src["hf_config"],
                              split=src["hf_split"], streaming=True)
        else:
            ds = load_dataset(src["hf_path"],
                              split=src["hf_split"], streaming=True)
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")
        return False

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    total_chars   = 0
    total_articles = 0
    debut = time.time()

    print(f"  Écriture dans {output_file}...")
    with open(output_file, mode, encoding="utf-8") as f:
        for exemple in ds:
            texte = exemple.get(src["champ"], "")
            # Gérer les dicts (ex: traductions {"fr": "...", "en": "..."})
            if isinstance(texte, dict):
                texte = "\n".join(str(v) for v in texte.values() if v)
            # Toujours gérer les listes (ex: liste d'utterances de dialogue)
            if isinstance(texte, list):
                texte = "\n".join(str(t) for t in texte if t)
            # Deux champs (ex: context + response)
            if src.get("champ2"):
                texte2 = exemple.get(src["champ2"], "")
                if isinstance(texte2, list):
                    texte2 = "\n".join(str(t) for t in texte2 if t)
                texte  = f"{texte}\n{texte2}".strip()
            if not texte or len(texte) < 20:
                continue
            texte = nettoyer_texte(texte)
            if not texte:
                continue
            f.write(texte + "\n\n")
            total_chars   += len(texte)
            total_articles += 1
            if total_chars >= limite_chars:
                break
            if total_articles % 5000 == 0:
                duree = time.time() - debut
                print(f"  ⏳ {total_articles:,} articles — "
                      f"{total_chars/1_000_000:.1f} Mo — {duree:.0f}s")

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

def main():
    cats = afficher_categories()
    choix = input("\n  Choix > ").strip().lower()

    # ── Options de gestion ──
    if choix == "c":
        combiner_sources()
        return
    if choix == "s":
        supprimer_donnees()
        return
    if choix == "l":
        print("\n  ── Sources téléchargées ──")
        for lang in ["en", "fr", "multi"]:
            fichiers = lister_sources(lang)
            if fichiers:
                print(f"\n  {FLAG.get(lang,'')} {lang.upper()} :")
                afficher_sources_dispo(lang)
        return

    try:
        idx = int(choix) - 1
        if not (0 <= idx < len(cats)):
            print("❌ Choix invalide.")
            return
        cat_choisie = cats[idx]
    except:
        print("❌ Choix invalide.")
        return

    # ── Afficher les sources de la catégorie ──
    print(f"\n{'='*62}")
    print(f"  📁 {CATEGORIES[cat_choisie]}")
    print(f"{'='*62}\n")
    
    sources_cat = {k: v for k, v in SOURCES.items() if v["cat"] == cat_choisie}
    cles = list(sources_cat.keys())
    
    for i, cle in enumerate(cles, 1):
        src = sources_cat[cle]
        print(f"  [{i:>2}]  {src['nom']}")
        
    print("\n  [ q]  Retour")
    
    choix_src = input("\n  Dataset > ").strip().lower()
    if choix_src == 'q' or not choix_src:
        return
        
    try:
        idx_src = int(choix_src) - 1
        if not (0 <= idx_src < len(cles)):
            print("❌ Choix invalide.")
            return
        cle_finale = cles[idx_src]
    except:
        print("❌ Choix invalide.")
        return
        
    src = SOURCES[cle_finale]

    if src.get("gated"):
        print("⚠️  Ce dataset nécessite un compte HuggingFace (authentification locale).")
        if input("   Continuer quand même ? [o/N] > ").strip().lower() != 'o':
            return

    langue = get_langue_by_cat(src["cat"])
    flag   = FLAG.get(langue, "")
    limite = demander_mo()

    # ── Dossier sources par langue ──
    sources_dir = os.path.join(DATA_ROOT, langue, "sources")
    os.makedirs(sources_dir, exist_ok=True)
    slug        = slug_source(src["nom"])
    output_file = os.path.join(sources_dir, f"{slug}.txt")

    # ── Mode ajout ou remplacement (dans le fichier source) ──
    mode = "w"
    if os.path.exists(output_file):
        taille = os.path.getsize(output_file) / 1_000_000
        print(f"\n  ⚠️  Source '{slug}.txt' existe déjà ({taille:.1f} Mo)")
        print("  [1] Ajouter à ce fichier source")
        print("  [2] Remplacer ce fichier source")
        choix_mode = input("  Choix > ").strip()
        if choix_mode == "2":
            mode = "w"
            print("  → Remplacement")
        else:
            mode = "a"
            print("  -> Ajout")

    ok = False

    if src["type"] == "cc":
        ok = telecharger_common_crawl(output_file, src.get("filtre"), limite, mode)
    else:
        ok = telecharger_hf(src, output_file, limite, mode)

    if ok:
        combiner_sources(langue)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Telecharger des donnees pour WishAI")
    parser.add_argument("--download",  default=None,  help="ID de la source (ex: fr_culturax)")
    parser.add_argument("--mo",        type=int, default=200, help="Mo a telecharger")
    parser.add_argument("--combine",   action="store_true",   help="Combiner les sources apres dl")
    parser.add_argument("--is_raw_hf", action="store_true",   help="Dataset HF brut")
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
        output_file  = os.path.join(sources_dir, slug + ".txt")
        limite       = args.mo * 1_000_000

        ok = False
        if src["type"] == "cc":
            ok = telecharger_common_crawl(output_file, src.get("filtre"), limite)
        else:
            ok = telecharger_hf(src, output_file, limite)

        if ok and args.combine:
            combiner_sources(langue)
    else:
        main()
