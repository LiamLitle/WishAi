# catalogue.py - Decouverte automatique de datasets via l'API HuggingFace
# - 100 requetes couvrant toutes les langues et domaines
# - Fetch simultane avec ThreadPoolExecutor (14 threads)
# - Deduplication par raw_id ET par nom normalise
# - Relancable plusieurs fois sans creer de doublons

import json, os, re, threading
import urllib.request, urllib.parse, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DS_FILE = os.path.join(SRC_DIR, "datasets.json")
HF_API  = "https://huggingface.co/api/datasets"
WORKERS = 14

LANG_DISPLAY = {
    "fr":("Francais","\U0001f1eb\U0001f1f7"),"en":("Anglais","\U0001f1ec\U0001f1e7"),
    "de":("Allemand","\U0001f1e9\U0001f1ea"),"es":("Espagnol","\U0001f1ea\U0001f1f8"),
    "zh":("Chinois","\U0001f1e8\U0001f1f3"),"ja":("Japonais","\U0001f1ef\U0001f1f5"),
    "ar":("Arabe","\U0001f1f8\U0001f1e6"),"ru":("Russe","\U0001f1f7\U0001f1fa"),
    "pt":("Portugais","\U0001f1e7\U0001f1f7"),"it":("Italien","\U0001f1ee\U0001f1f9"),
    "ko":("Coreen","\U0001f1f0\U0001f1f7"),"nl":("Neerlandais","\U0001f1f3\U0001f1f1"),
    "pl":("Polonais","\U0001f1f5\U0001f1f1"),"tr":("Turc","\U0001f1f9\U0001f1f7"),
    "hi":("Hindi","\U0001f1ee\U0001f1f3"),"vi":("Vietnamien","\U0001f1fb\U0001f1f3"),
    "sv":("Suedois","\U0001f1f8\U0001f1ea"),"no":("Norvegien","\U0001f1f3\U0001f1f4"),
    "fi":("Finlandais","\U0001f1eb\U0001f1ee"),"cs":("Tcheque","\U0001f1e8\U0001f1ff"),
    "ro":("Roumain","\U0001f1f7\U0001f1f4"),"hu":("Hongrois","\U0001f1ed\U0001f1fa"),
    "uk":("Ukrainien","\U0001f1fa\U0001f1e6"),"he":("Hebreu","\U0001f1ee\U0001f1f1"),
    "th":("Thai","\U0001f1f9\U0001f1ed"),"id":("Indonesien","\U0001f1ee\U0001f1e9"),
    "fa":("Persan","\U0001f1ee\U0001f1f7"),"da":("Danois","\U0001f1e9\U0001f1f0"),
    "el":("Grec","\U0001f1ec\U0001f1f7"),"bn":("Bengali","\U0001f1e7\U0001f1e9"),
    "ca":("Catalan","\U0001f3f3\U0000fe0f"),"sk":("Slovaque","\U0001f1f8\U0001f1f0"),
    "bg":("Bulgare","\U0001f1e7\U0001f1ec"),"hr":("Croate","\U0001f1ed\U0001f1f7"),
    "lt":("Lituanien","\U0001f1f1\U0001f1f9"),"lv":("Letton","\U0001f1f1\U0001f1fb"),
    "et":("Estonien","\U0001f1ea\U0001f1ea"),"sl":("Slovene","\U0001f1f8\U0001f1ee"),
    "sr":("Serbe","\U0001f1f7\U0001f1f8"),"ms":("Malais","\U0001f1f2\U0001f1fe"),
    "ur":("Ourdou","\U0001f1f5\U0001f1f0"),"sw":("Swahili","\U0001f1f0\U0001f1ea"),
    "ta":("Tamoul","\U0001f1ee\U0001f1f3"),"te":("Telugu","\U0001f1ee\U0001f1f3"),
}

TASK_MAP = {
    "text-generation":"Generation de texte","text-classification":"Classification",
    "question-answering":"Question-Reponse","extractive-qa":"Question-Reponse",
    "open-domain-qa":"Question-Reponse","translation":"Traduction",
    "summarization":"Resume","conversational":"Dialogues",
    "dialogue-modeling":"Dialogues","fill-mask":"Completion",
    "token-classification":"Analyse de texte","named-entity-recognition":"Analyse de texte",
    "natural-language-inference":"Raisonnement","text2text-generation":"Transformation",
    "multiple-choice":"QCM","language-modeling":"Langage",
    "sentence-similarity":"Similarite","zero-shot-classification":"Classification",
    "text-ranking":"Classement","code-generation":"Programmation","other":"Divers",
}

REQUETES = [
    # Langues europeennes (20)
    ("french text corpus",         "Web",              "fr",  "FR"),
    ("english text nlp",           "Web",              "en",  "GB"),
    ("german language corpus",     "Web",              "de",  "DE"),
    ("spanish language corpus",    "Web",              "es",  "ES"),
    ("italian language",           "Web",              "it",  "IT"),
    ("portuguese language",        "Web",              "pt",  "BR"),
    ("dutch language",             "Web",              "nl",  "NL"),
    ("polish language",            "Web",              "pl",  "PL"),
    ("russian language",           "Web",              "ru",  "RU"),
    ("ukrainian language",         "Web",              "uk",  "UA"),
    ("romanian language",          "Web",              "ro",  "RO"),
    ("czech language",             "Web",              "cs",  "CZ"),
    ("hungarian language",         "Web",              "hu",  "HU"),
    ("swedish language",           "Web",              "sv",  "SE"),
    ("norwegian language",         "Web",              "no",  "NO"),
    ("danish language",            "Web",              "da",  "DK"),
    ("finnish language",           "Web",              "fi",  "FI"),
    ("greek language",             "Web",              "el",  "GR"),
    ("bulgarian language",         "Web",              "bg",  "BG"),
    ("croatian serbian",           "Web",              "hr",  "HR"),
    # Langues asiatiques & autres (15)
    ("chinese mandarin nlp",       "Web",              "zh",  "CN"),
    ("japanese language corpus",   "Web",              "ja",  "JP"),
    ("korean language corpus",     "Web",              "ko",  "KR"),
    ("arabic language corpus",     "Web",              "ar",  "SA"),
    ("hindi language corpus",      "Web",              "hi",  "IN"),
    ("turkish language corpus",    "Web",              "tr",  "TR"),
    ("vietnamese language",        "Web",              "vi",  "VN"),
    ("thai language",              "Web",              "th",  "TH"),
    ("indonesian malay",           "Web",              "id",  "ID"),
    ("bengali language",           "Web",              "bn",  "BD"),
    ("persian farsi",              "Web",              "fa",  "IR"),
    ("urdu language",              "Web",              "ur",  "PK"),
    ("tamil telugu",               "Web",              "ta",  "IN"),
    ("swahili african",            "Web",              "sw",  "KE"),
    ("hebrew language",            "Web",              "he",  "IL"),
    # Multilingue (5)
    ("multilingual corpus",        "Web",              "multi","xx"),
    ("cross-lingual transfer",     "Traduction",       "multi","xx"),
    ("parallel corpus translation","Traduction",       "multi","xx"),
    ("machine translation",        "Traduction",       "multi","xx"),
    ("low-resource language",      "Web",              "multi","xx"),
    # Sciences & Recherche (10)
    ("arxiv scientific papers",    "Science",          "en",  "GB"),
    ("pubmed biomedical",          "Medecine",         "en",  "GB"),
    ("clinical medical health",    "Medecine",         "en",  "GB"),
    ("mathematics proof formal",   "Science",          "en",  "GB"),
    ("chemistry molecular",        "Science",          "en",  "GB"),
    ("physics quantum",            "Science",          "en",  "GB"),
    ("biology genomics protein",   "Science",          "en",  "GB"),
    ("astronomy astrophysics",     "Science",          "en",  "GB"),
    ("neuroscience cognitive",     "Science",          "en",  "GB"),
    ("environmental climate",      "Science",          "en",  "GB"),
    # Code & Programmation (10)
    ("code programming github",    "Programmation",    "en",  "GB"),
    ("python code functions",      "Programmation",    "en",  "GB"),
    ("javascript typescript web",  "Programmation",    "en",  "GB"),
    ("java c++ systems code",      "Programmation",    "en",  "GB"),
    ("competitive programming",    "Programmation",    "en",  "GB"),
    ("sql database queries",       "Programmation",    "en",  "GB"),
    ("bash shell linux",           "Programmation",    "en",  "GB"),
    ("cybersecurity vulnerability","Programmation",    "en",  "GB"),
    ("software documentation api", "Programmation",    "en",  "GB"),
    ("machine learning code",      "Programmation",    "en",  "GB"),
    # Instructions & Alignment (7)
    ("instruction following",      "Instructions",     "en",  "GB"),
    ("chat assistant helpful",     "Instructions",     "en",  "GB"),
    ("rlhf human feedback",        "Instructions",     "en",  "GB"),
    ("synthetic instruct gpt",     "Instructions",     "en",  "GB"),
    ("preference reward model",    "Instructions",     "en",  "GB"),
    ("roleplay character fiction", "Instructions",     "en",  "GB"),
    ("safety alignment refusal",   "Instructions",     "en",  "GB"),
    # Raisonnement & Q&A (7)
    ("question answering open",    "Question-Reponse", "en",  "GB"),
    ("reading comprehension",      "Question-Reponse", "en",  "GB"),
    ("commonsense reasoning",      "Raisonnement",     "en",  "GB"),
    ("logical math reasoning",     "Raisonnement",     "en",  "GB"),
    ("visual question answering",  "Question-Reponse", "en",  "GB"),
    ("knowledge graph facts",      "Encyclopedie",     "en",  "GB"),
    ("trivia quiz knowledge",      "Encyclopedie",     "en",  "GB"),
    # Dialogues & Social (5)
    ("dialogue conversation chat", "Dialogues",        "en",  "GB"),
    ("social media reddit forum",  "Dialogues",        "en",  "GB"),
    ("customer support service",   "Dialogues",        "en",  "GB"),
    ("debate argument opinion",    "Dialogues",        "en",  "GB"),
    ("storytelling narrative",     "Litterature",      "en",  "GB"),
    # Texte & Analyse (5)
    ("sentiment opinion review",   "Classification",   "multi","xx"),
    ("text summarization news",    "Resume",           "en",  "GB"),
    ("named entity relation",      "Analyse de texte", "en",  "GB"),
    ("information extraction",     "Analyse de texte", "en",  "GB"),
    ("text classification topic",  "Classification",   "en",  "GB"),
    # Domaines specialises (16)
    ("legal law court",            "Droit",            "en",  "GB"),
    ("finance stock economics",    "Finance",          "en",  "GB"),
    ("news journalism press",      "Actualites",       "multi","xx"),
    ("books literature novels",    "Litterature",      "en",  "GB"),
    ("wikipedia encyclopedia",     "Encyclopedie",     "multi","xx"),
    ("philosophy ethics morality", "Philosophie",      "en",  "GB"),
    ("education pedagogy school",  "Education",        "en",  "GB"),
    ("history ancient modern",     "Encyclopedie",     "multi","xx"),
    ("psychology mental therapy",  "Medecine",         "en",  "GB"),
    ("religion theology culture",  "Encyclopedie",     "multi","xx"),
    ("sports fitness game",        "Divers",           "en",  "GB"),
    ("cooking food recipe",        "Divers",           "en",  "GB"),
    ("music lyrics audio",         "Divers",           "multi","xx"),
    ("travel geography tourism",   "Divers",           "en",  "GB"),
    ("creative writing poetry",    "Litterature",      "en",  "GB"),
    ("agriculture environment",    "Science",          "en",  "GB"),
]

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def normalise(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fetch_hf(query, limit=100):
    params = urllib.parse.urlencode({
        "search": query, "sort": "downloads",
        "direction": "-1", "limit": limit, "full": "false",
    })
    try:
        req = urllib.request.Request(
            HF_API + "?" + params,
            headers={"User-Agent": "wishai-catalogue/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        log("  WARN " + repr(query) + ": " + str(e))
        return []


def parse_tags(tags):
    lang_code = "multi"
    domaine   = "Divers"
    for t in tags:
        if t.startswith("language:"):
            code = t.split(":", 1)[1]
            if code in LANG_DISPLAY:
                lang_code = code
                break
    for t in tags:
        if t.startswith("task_categories:"):
            task = t.split(":", 1)[1]
            if task in TASK_MAP:
                domaine = TASK_MAP[task]
                break
    return lang_code, domaine


def make_entry(d, domaine_defaut, lang_defaut):
    tags               = d.get("tags") or []
    lang_code, domaine = parse_tags(tags)

    if lang_code == "multi" and lang_defaut not in ("multi", "xx"):
        lang_code = lang_defaut
    if domaine == "Divers" and domaine_defaut != "Divers":
        domaine = domaine_defaut

    if lang_code in LANG_DISPLAY:
        lang_nom, flag = LANG_DISPLAY[lang_code]
    else:
        lang_nom = "Multilingue"
        flag     = "\U0001f30d"

    desc = (d.get("description") or "").strip()
    if len(desc) > 180:
        desc = desc[:177] + "..."
    if not desc:
        author = d.get("author") or d["id"].split("/")[0]
        desc   = "Dataset heberge par " + author + " sur HuggingFace Hub."

    downloads = d.get("downloads", 0)
    dl_str    = (str(downloads // 1000) + "k") if downloads >= 1000 else str(downloads)
    nom       = d["id"].split("/")[-1].replace("-", " ").replace("_", " ").title()

    return {
        "id":      d["id"].replace("/", "__"),
        "nom":     nom,
        "desc":    desc,
        "lang":    lang_nom,
        "hf_lang": lang_code,
        "flag":    flag,
        "domain":  domaine,
        "sval":    200,
        "raw":     True,
        "raw_id":  d["id"],
        "dl":      dl_str,
        "likes":   d.get("likes", 0),
        "curated": False,
    }


def worker(args):
    query, domaine_defaut, lang_defaut, _ = args
    items   = fetch_hf(query, limit=100)
    entries = []
    for d in items:
        if d.get("id"):
            entries.append((d["id"], make_entry(d, domaine_defaut, lang_defaut)))
    return query, entries


def run():
    print("\n" + "="*64)
    print("  WishAI -- Catalogue HuggingFace")
    print("  " + str(len(REQUETES)) + " requetes  /  " + str(WORKERS) + " threads")
    print("="*64)

    existing = []
    try:
        with open(DS_FILE, "rb") as f:
            raw = f.read().replace(b"\x00", b"")
        existing = json.loads(raw.decode("utf-8"))
    except Exception:
        pass

    # Double cle de deduplication
    seen_raw  = set()
    seen_noms = set()
    for e in existing:
        seen_raw.add(e.get("raw_id") or e.get("id"))
        seen_noms.add(normalise(e.get("nom", "")))

    curated_count = sum(1 for e in existing if e.get("curated"))
    auto_avant    = sum(1 for e in existing if not e.get("curated"))
    print("\n  Curates : " + str(curated_count) + "  |  Auto existants : " + str(auto_avant))
    print("  Fetch en cours...\n")

    nouveaux  = []
    completed = 0
    total     = len(REQUETES)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, r): r[0] for r in REQUETES}
        for fut in as_completed(futures):
            query, entries = fut.result()
            completed += 1
            added = 0
            for hf_id, entry in entries:
                nom_key = normalise(entry.get("nom", ""))
                if hf_id not in seen_raw and nom_key not in seen_noms:
                    seen_raw.add(hf_id)
                    seen_noms.add(nom_key)
                    nouveaux.append(entry)
                    added += 1
            log("  [" + str(completed).rjust(3) + "/" + str(total) + "] "
                + query.ljust(38) + " +" + str(added).rjust(3)
                + "  total: " + str(len(nouveaux)))

    nouveaux.sort(key=lambda x: x.get("likes", 0), reverse=True)

    curated = [e for e in existing if e.get("curated")]
    auto    = [e for e in existing if not e.get("curated")]
    auto    = sorted(auto + nouveaux, key=lambda x: x.get("likes", 0), reverse=True)
    final   = curated + auto

    # Deduplication finale
    seen_f = set()
    dedup  = []
    for e in final:
        key = e.get("raw_id") or e.get("id")
        if key not in seen_f:
            seen_f.add(key)
            dedup.append(e)
    final = dedup

    with open(DS_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    auto_apres = len([e for e in final if not e.get("curated")])
    print("\n  OK: " + str(len(nouveaux)) + " nouveaux datasets ajoutes")
    print("     Auto avant : " + str(auto_avant) + "  ->  apres : " + str(auto_apres))
    print("     Total datasets.json : " + str(len(final)))
    print("="*64 + "\n")
    return len(nouveaux)


if __name__ == "__main__":
    run()
