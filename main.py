# Locale
# python -m uvicorn main:app --host 0.0.0.0 --port 8080
# http://localhost:8080/ai

# Remoto
# python -m uvicorn main:app --host 80.88.88.48 --port 11000
# http://80.88.88.48:11000/ai

# Libreria che consente la rappresentazione vettoriale (embedded) di frasi anziché parole
from sentence_transformers import SentenceTransformer
# Libreria per calcolare la similarità coseno tra vettori (la similiarità coseno misura quanto due vettori sono simili)
from sklearn.metrics.pairwise import cosine_similarity
# Libreria per il rilevamento della lingua del testo
from langdetect import detect
# Libreria per creare API web
from fastapi import FastAPI
# Libreria per la definizione di modelli di dati
from pydantic import BaseModel
# Libreria per tipi dinamici
from typing import Dict, Any 
# Libreria per le espressioni regolari
import re 
# Libreria per l'elaborazione del linguaggio naturale
import spacy 
# Componente spacy per il riconoscimento di entità basato su pattern
from spacy.pipeline import EntityRuler 
# Libreria per mescolare i dati di addestramento
import random 
# Libreria per la normalizzazione dei caratteri Unicode
from unidecode import unidecode
# Libreria per il download e l'uso di modelli NLP (Stanza)
import stanza
""" # Libreria per connettersi a PostgreSQL
import psycopg2 """

# Inizializza l'app FastAPI
app = FastAPI()

# --- Modello sentence-transformers
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

#token = 'YOUR_HUGGING'
#model = SentenceTransformer('google/embeddinggemma-300m', use_auth_token=token)

""" # Funzione per caricare le API da un database PostgreSQL
def carica_api_da_db():
    conn = psycopg2.connect(
        host="HOST",
        database="DBNAME",
        user="USER",
        password="PASSWORD"
    )
    cur = conn.cursor()
    cur.execute("SELECT descrizione, parametri, path FROM api_catalog")
    risultati = cur.fetchall()
    cur.close()
    conn.close()
    api_catalog_db = [{"descrizione": r[0], "parametri": r[1], "path": r[2]} for r in risultati]
    return api_catalog_db

# Carica le API dal database
api_catalog = carica_api_da_db() """

# ---- Lista statica di API con descrizioni, parametri ed endpoint
api_catalog = [
    {
        "descrizione": "Mostra la home page dell'applicazione",
        "parametri": {},
        "path": "home",
        "keywords": ["home", "inizio", "pagina principale", "schermata iniziale"]
    },
    {
        "descrizione": "Crea un nuovo finanziamento",
        "parametri": {},
        "path": "nuovo-finanziamento",
        "keywords": ["nuovo", "creare", "aggiungere", "aprire", "attivare", "inserire", "avviare"]
    },
    {
        "descrizione": "Dettagli del finanziamento",
        "parametri": {"id_finanziamento": "string"},
        "path": "dettaglio-finanziamento",
        "keywords": ["dettaglio", "dettagli", "info", "informazioni", "vedere", "visualizzare", "mostrare", "informare", "andare"]
    },
    {
        "descrizione": "Gestisci i dati della perizia",
        "parametri": {"id_finanziamento": "string"},
        "path": "dati-perizia",
        "keywords": ["perizia"]
    },
    {
        "descrizione": "Dettagli dell'attività",
        "parametri": {"id_finanziamento": "string", "id_attivita": "string"},
        "path": "dettaglio-attivita",
        "keywords": ["attività", "task"]
    },
    {
        "descrizione": "Dettagli della rata",
        "parametri": {"id_finanziamento": "string", "id_rata": "string"},
        "path": "dettaglio-rata",
        "keywords": ["rata"]
    },
    {
        "descrizione": "Gestisci i finanziamenti",
        "parametri": {},
        "path": "gestisci-finanziamenti",
        "keywords": ["gestione", "lista", "finanziamenti", "elenco"] 
        #la keyword "finanziamenti" non verrà mai presa perché nella normalizzazione viene rimosso il plurale
        #mettere il singolare "finanziamento" è troppo generico e porta a falsi positivi
    },
    {
        "descrizione": "Stipula il finanziamento",
        "parametri": {"id_finanziamento": "string"},
        "path": "stipula-finanziamento",
        "keywords": ["stipula", "stipulare", "firma", "contratto", "sottoscrivere", "firmare"]
    },
    {
        "descrizione": "Eroga il finanziamento",
        "parametri": {"id_finanziamento": "string"},
        "path": "erogazione-finanziamento",
        "keywords": ["eroga", "erogazione", "erogare"]
        #esempio: "erogazione 12345" non viene preso a causa dell'embedding troppo basso
    }
]

#  --- Modello spaCy per l'italiano ed EntityRuler
try:
    nlp = spacy.load("it_core_news_sm")
except Exception:
    # se non disponibile, crea un blank 'it' (meno performante ma funziona)
    nlp = spacy.blank("it")

def init_entity_ruler(nlp_obj):
    """
    Aggiunge un EntityRuler con pattern utili per riconoscere contesti tipo:
    'finanziamento 12345', 'id finanziamento: 12345', 'id 12345', ecc.
    """
    # Se esiste già un entity_ruler, riutilizzalo; altrimenti crealo
    if "entity_ruler" not in nlp_obj.pipe_names:
        if "ner" in nlp_obj.pipe_names:
            ruler = nlp_obj.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
        else:
            ruler = nlp_obj.add_pipe("entity_ruler", config={"overwrite_ents": True})
    else:
        ruler = nlp_obj.get_pipe("entity_ruler")
    patterns = [
        # === PATTERN 1: caso tokenizzato ===
        # es: "finanziamento 25-81", "fin 25 / 81"
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": {"REGEX": "^fin"}},       # "fin" o "finanziamento"
                {"IS_SPACE": True, "OP": "*"},      # spazi opzionali
                {"IS_DIGIT": True},                 # primo numero
                {"TEXT": {"REGEX": r"^[\-\./]$"}, "OP": "?"},  # separatore opzionale
                {"IS_SPACE": True, "OP": "?"},      # spazio opzionale
                {"IS_DIGIT": True, "OP": "?"}       # secondo numero opzionale
            ]
        },
        # === PATTERN 2: caso NON tokenizzato ===
        # es: "finanziamento 25/81", "fin 25.81", "finanziamento 25-81"
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": {"REGEX": "^fin"}},       # "fin" o "finanziamento"
                {"IS_SPACE": True, "OP": "*"},
                {"TEXT": {"REGEX": r"^\d+(?:[\-\./\s]?\d+)*$"}}  # numeri concatenati o con separatori
            ]
        },
        # === (Facoltativo) pattern per "id finanziamento" ===
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": "id"},
                {"IS_SPACE": True, "OP": "*"},
                {"LOWER": {"REGEX": "^fin"}},
                {"IS_PUNCT": True, "OP": "?"},
                {"IS_SPACE": True, "OP": "*"},
                {"TEXT": {"REGEX": r"^\d+(?:[\-\./\s]?\d+)*$"}}
            ]
        },

        # === Pattern 3: "stipula" o "eroga" seguiti da un numero ===
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": {"REGEX": "^(stipula|eroga)$"}},  # "stipula" o "eroga"
                {"IS_SPACE": True, "OP": "*"},              # spazi opzionali
                {"IS_DIGIT": True}                          # numero
            ]
        },

        # === Pattern 4: "stipula" o "eroga" seguiti da numeri con separatori ===
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": {"REGEX": "^(stipula|eroga)$"}},  # "stipula" o "eroga"
                {"IS_SPACE": True, "OP": "*"},              # spazi opzionali
                {"TEXT": {"REGEX": r"^\d+(?:[\-\./\s]?\d+)*$"}}  # numeri concatenati o con separatori
            ]
        },

        {"label": "ID_RATA", "pattern": [{"TEXT": {"REGEX": "^rat.*"}}, {"IS_SPACE": True, "OP": "?"}, {"TEXT": {"REGEX": "^[0-9]+([\\s\\-\\./][0-9]+)*$"}}]},
        {"label": "ID_RATA", "pattern": [{"LOWER": "rata"}, {"IS_DIGIT": True}]},
        {"label": "ID_RATA", "pattern": [{"LOWER": "id"}, {"LOWER": "rata"}, {"IS_PUNCT": True, "OP": "?"}, {"IS_DIGIT": True}]},
        {"label": "ID_RATA", "pattern": [{"LOWER": "id"}, {"IS_DIGIT": True}]},
        
        {"label": "ID_ATTIVITA", "pattern": [{"TEXT": {"REGEX": "^att.*"}}, {"IS_SPACE": True, "OP": "?"}, {"TEXT": {"REGEX": "^[0-9]+([\\s\\-\\./][0-9]+)*$"}}]},
        {"label": "ID_ATTIVITA", "pattern": [{"LOWER": "attivita"}, {"IS_DIGIT": True}]},
        {"label": "ID_ATTIVITA", "pattern": [{"LOWER": "id"}, {"LOWER": "attivita"}, {"IS_PUNCT": True, "OP": "?"}, {"IS_DIGIT": True}]},
        {"label": "ID_ATTIVITA", "pattern": [{"LOWER": "id"}, {"IS_DIGIT": True}]},

        # pattern semplici per numeri che seguono ':' o '=' o 'n.'
        {"label": "NUMBER_GENERIC", "pattern": [{"IS_ASCII": True, "OP": "?"}, {"IS_DIGIT": True}]},

        #TODO: aggiungere altri patterns specifici del tuo dominio
    ]
    ruler.add_patterns(patterns)

init_entity_ruler(nlp)

# --- Funzione per estrarre id basandosi su NER + regex come fallback
def extract_params_from_text(text: str, api_params: Dict[str, str]) -> Dict[str, Any]:
    """
    Restituisce un dict con i parametri trovati: {param_name: value}
    Strategia:
      1) usa nlp per trovare entità con label tipo ID_FINANZIAMENTO, ID_RATA, ...
      2) se non trovi, cerca con regex numeriche vicino a parole chiave rilevanti
      3) se ancora nulla, tenta un fallback: primo numero trovato (opzionale)
    """
    doc = nlp(text)
    found = {}
    # 1)
    for ent in doc.ents:
        if ent.label_ in ("ID_FINANZIAMENTO", "ID_RATA", "ID_ATTIVITA"):
            print(f"Trovata entità: {ent.text} con label {ent.label_}")
            clean_text = re.sub(r'\D+', '', ent.text)
            if clean_text:
                if ent.label_ == "ID_FINANZIAMENTO" and "id_finanziamento" in api_params:
                    found["id_finanziamento"] = clean_text.zfill(8)
                elif ent.label_ == "ID_RATA" and "id_rata" in api_params:
                    found["id_rata"] = clean_text
                elif ent.label_ == "ID_ATTIVITA" and "id_attivita" in api_params:
                    found["id_attivita"] = clean_text
    if all(p in found for p in api_params):
        return found
    # 2A) cerca pattern specifici "stipula/eroga/fin <numero>"
    text_low = text.lower()
    m = re.search(r"\b(stipula|eroga|fin|finanziamento)\b[\s\:\-]*([0-9]+(?:[\/\-\.\s][0-9]+)*)", text_low)
    if m:
        verb = m.group(1)
        num = re.sub(r'\D+', '', m.group(2))
        if num:
            if "id_finanziamento" in api_params and "id_finanziamento" not in found:
                found["id_finanziamento"] = num.zfill(8)
            # possibile estensione: mappare ad altri parametri in base al verbo
    # 2B) cerca pattern generici "id <parametro> <numero>"
    for pname in api_params:
        if pname in found:
            continue
        keyword = pname.replace("id_", "").replace("_", " ")
        pattern_context = rf"(?:{keyword})\W{{0,6}}?(\d{{1,20}})|(?:id\W*{keyword})\W{{0,6}}?(\d{{1,20}})"
        m2 = re.search(pattern_context, text_low, flags=re.IGNORECASE)
        if m2:
            num = m2.group(1) or m2.group(2)
            if num:
                if pname == "id_finanziamento":
                    found[pname] = re.sub(r'\D+', '', num).zfill(8)
                else:
                    found[pname] = re.sub(r'\D+', '', num)
    # 3)
    for pname in api_params:
        if pname not in found:
            m3 = re.search(r"(\d{1,20})", text_low)
            if m3:
                if pname == "id_finanziamento":
                    found[pname] = m3.group(1).zfill(8)
                else:
                    found[pname] = m3.group(1)
    return found

# --- Funzione per (ri)allenare il modello NER su esempi etichettati
def train_spacy_ner(base_model="it_core_news_sm", training_data=None, n_iter=30):
    """
    training_data: lista di (text, {"entities": [(start, end, label), ...]})
    Esempio:
      training_data = [
        ("voglio fare la stipula del finanziamento 123456", {"entities": [(31, 37, "ID_FINANZIAMENTO")]}),
        ...
      ]
    Nota: esegui questa funzione localmente per migliorare il riconoscimento.
    """
    if training_data is None or len(training_data) == 0:
        raise ValueError("Serve training_data con esempi annotati")
    # carica modello di base o blank
    try:
        nlp_train = spacy.load(base_model)
    except Exception:
        nlp_train = spacy.blank("it")
    # controlla se il modello ha già un NER, altrimenti lo aggiunge
    if "ner" not in nlp_train.pipe_names:
        ner = nlp_train.add_pipe("ner", last=True)
    else:
        ner = nlp_train.get_pipe("ner")
    # estrae tutte le label nuove presenti negli esempi di training
    labels = set([ent[2] for _, ann in training_data for ent in ann.get("entities", [])])
    # le aggiunge al NER, così il modello saprà che deve imparare a riconoscerle
    for lbl in labels:
        ner.add_label(lbl)
    # disattiva temporaneamente tutti i componenti del pipeline tranne il NER, per addestrare solo il modello di riconoscimento entità senza aggiornare gli altri moduli
    other_pipes = [p for p in nlp_train.pipe_names if p != "ner"]
    with nlp_train.disable_pipes(*other_pipes):
        # un optimizer è un oggetto che aggiorna i pesi del modello per minimizzare la funzione di perdita
        optimizer = nlp_train.begin_training()
        for itn in range(n_iter): # n_iter → numero di epoche
            # mescola gli esempi ad ogni epoca, così il modello non si abitua all’ordine dei dati
            random.shuffle(training_data)
            losses = {}
            # crea mini-batch dal training set, partendo da 4 esempi e aumentando gradualmente fino a 32
            # (piccoli batch all’inizio per precisione → grandi batch alla fine per velocità, senza sacrificare la qualità dell’addestramento)
            batches = spacy.util.minibatch(training_data, size=spacy.util.compounding(4.0, 32.0, 1.001))
            # aggiornamento del modello
            for batch in batches:
                # separa dai batch di training i testi dalle annotazioni per poterli passare separatamente al modello NER
                texts, annotations = zip(*batch)
                # aggiorna il modello NER con i nuovi dati
                nlp_train.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
                # sgd → Stochastic Gradient Descent, algoritmo di ottimizzazione
                # drop → tasso di dropout, cioè la probabilità di ignorare casualmente alcune unità durante l’addestramento per evitare overfitting
                # losses → dizionario che tiene traccia delle perdite durante l'addestramento
            # Opzionale: monitorare le perdite: print(itn, losses)
    # salva modello su disco per riutilizzare
    nlp_train.to_disk("./ner_finanziamenti_model")
    return "./ner_finanziamenti_model"

# --- Pydantic model per la richiesta
class Richiesta(BaseModel):
    richiesta_utente: str
    id_finanziamento: str

stanza.download('it')  # eseguila una sola volta
nlp_stanza = stanza.Pipeline('it', processors='tokenize,mwt,pos,lemma', use_gpu=False)

def normalizza_testo(testo: str) -> str:
    """
    Pipeline completa di normalizzazione di un testo in italiano.
    Passaggi:
    1. Pulizia (rimozione di URL, emoji, punteggiatura, numeri, ecc.)
    2. Normalizzazione accenti
    3. Tokenizzazione
    4. Lemmatizzazione
    5. Rimozione stopwords
    6. Lowercase
    """
    print(f"Testo originale: {testo}")

    testo = re.sub(r"http\S+|www\S+|https\S+", " ", testo)  # URL
    testo = re.sub(r"@\w+", " ", testo)                     # menzioni
    testo = re.sub(r"#\w+", " ", testo)                     # hashtag
    testo = re.sub(r"[^\w\sàèéìòóùÀÈÉÌÒÓÙ]", " ", testo)  # rimuove simboli, emoji, numeri
    testo = re.sub(r"\s+", " ", testo).strip()              # spazi multipli

    testo = unidecode(testo)  # converte tutti gli accenti a forme canoniche ASCII

    doc = nlp_stanza(testo)
    lemmi = [word.lemma.lower() for sent in doc.sentences for word in sent.words]

    stopwords_it = set([
        "il", "lo", "la", "i", "gli", "le", "un", "una", "uno",
        "di", "a", "da", "in", "su", "per", "con", "come", "tra", "fra",
        "del", "della", "dell", "dei", "degli", "delle",
        "al", "allo", "alla", "ai", "agli", "alle",
        "dal", "dallo", "dalla", "dai", "dagli", "dalle",
        "nel", "nello", "nella", "nei", "negli", "nelle",
        "sul", "sullo", "sulla", "sui", "sugli", "sulle",
        "e", "ed", "o", "od", "ma", "anche", "se", "che", "quando", "dove", "come", "cui",
        "non", "piu", "meno", "molto", "tanto", "troppo", "tutti", "tutto",
        "questo", "quello", "questa", "quella", "questi", "quelle",
        "io", "tu", "lui", "lei", "noi", "voi", "loro",
        "mi", "ti", "si", "ci", "vi", "ne", "gli", "le", "li",
        "sono", "era", "stato", "stare", "essere", "avere",
        "ho", "hai", "ha", "abbiamo", "avete", "hanno",
        "sia", "siano", "sarà", "saranno", "può", "puoi", "può", "posso", "possono"
    ])
    tokens = [lemma for lemma in lemmi if lemma not in stopwords_it and lemma.strip() != ""]

    testo_normalizzato = " ".join(tokens)
    testo_normalizzato = testo_normalizzato.lower()
    print(f"Testo normalizzato: {testo_normalizzato}")

    return testo_normalizzato

# --- Funzione principale che cerca l'API e, se necessario, estrae parametri
def get_api(testo_input: str, id_finanziamento, soglia_similarita=0.5, peso_keyword=0.3, peso_embedding=0.9) -> Dict[str, Any]:
    """
    Restituisce l'API più probabile in base a:
    1) keyword specifiche nel testo
    2) similarità embedding tra testo e descrizione API
    3) estrazione parametri richiesti
    """
    try:
        """ # Controllo lingua
        if detect(testo_input) != "it":
            return {"codice_risposta": "KO", "risposta_app": "Per favore fornisci il testo in italiano."} """
        testo_originale = testo_input
        testo_input = normalizza_testo(testo_input)

        # Embedding del testo utente
        embedding_input = model.encode([testo_input])

        migliori_match = {"path": None, "score": 0.0, "api": None, "match_type": None}

        for api in api_catalog:  
            keywords = api.get("keywords", [])
            # punteggio keyword
            if keywords:
                kw_score = 1.0 if any(kw.lower() in testo_input for kw in keywords) else 0.0 # presenza/assenza
                # kw_score = sum(1 for kw in keywords if kw.lower() in testo_input) / len(keywords) # frazione di keyword trovate
                # kw_bonus = max((len(kw) / 20 for kw in keywords if kw.lower() in testo_input), default=0) # bonus per keyword più lunghe
                # kw_score += kw_bonus
            else:
                kw_score = 0.0

            # punteggio embedding
            embedding_descrizione = normalizza_testo(api["descrizione"])
            embedding_descrizione = model.encode([embedding_descrizione])
            emb_score = cosine_similarity(embedding_input, embedding_descrizione)[0][0]

            # punteggio combinato
            score_combinato = peso_keyword * kw_score + peso_embedding * emb_score

            print(f"  kw_score: {kw_score}")
            print(f"  emb_score: {emb_score}")
            print(f"  score_combinato: {score_combinato}")

            if score_combinato > migliori_match["score"]:
                match_type = "keyword+embedding" if kw_score > 0 else "embedding"
                migliori_match = {
                    "path": api["path"],
                    "score": float(score_combinato),
                    "api": api,
                    "match_type": match_type
                }

        # Se superiamo la soglia, estraiamo parametri
        if migliori_match["score"] >= soglia_similarita:
            api = migliori_match["api"]
            response = {
                "codice_risposta": "OK",
                "path": api["path"],
                "score": round(migliori_match["score"], 4),
                "match_type": migliori_match["match_type"]
            }

            if api.get("parametri"):
                estratti = extract_params_from_text(testo_originale, api["parametri"])
                mancanti = [p for p in api["parametri"].keys() if p not in estratti or not estratti[p]]
                if len(mancanti) == 0:
                    response["parametri"] = estratti
                else:
                    #TODO: gestione specifica per id_finanziamento passato separatamente
                    if "id_finanziamento" in mancanti and id_finanziamento:
                        print("Usando id_finanziamento passato separatamente.")
                        estratti["id_finanziamento"] = id_finanziamento
                        mancanti.remove("id_finanziamento")
                        response["parametri"] = estratti
                    else:
                        response["parametri_parziali"] = estratti
                        response["mancanti"] = mancanti
                        response["avviso"] = "Parametri mancanti o incerti. Fornisci gli id richiesti."
            return response

        return {"codice_risposta": "KO", "risposta_app": "La richiesta non trova corrispondenza con nessuna API.", "score": round(migliori_match["score"], 4)}

    except Exception as e:
        return {"codice_risposta": "KO", "risposta_app": f"Errore durante l'elaborazione: {str(e)}"}

# --- Endpoint FastAPI
@app.post("/ai")
def get_api_endpoint(richiesta: Richiesta):
    return get_api(richiesta.richiesta_utente, richiesta.id_finanziamento)
