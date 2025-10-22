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
""" # Libreria per connettersi a PostgreSQL
import psycopg2 """

# Inizializza l'app FastAPI
app = FastAPI()

# --- Modello sentence-transformers
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

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
        "keywords": ["home", "inizio", "pagina principale", "dashboard", "schermata iniziale"]
    },
    {
        "descrizione": "Crea un nuovo finanziamento",
        "parametri": {},
        "path": "nuovo-finanziamento",
        "keywords": ["nuovo", "crea", "aggiungi", "apri", "attiva", "inserisci"]
    },
    {
        "descrizione": "Mostra i dettagli del finanziamento",
        "parametri": {"id_finanziamento": "string"},
        "path": "dettaglio-finanziamento",
        "keywords": ["dettaglio", "dettagli", "vedi", "visualizza", "mostra", "info", "informazioni", "vai"]
    },
    {
        "descrizione": "Gestisci i dati della perizia",
        "parametri": {"id_finanziamento": "string"},
        "path": "dati-perizia",
        "keywords": ["gestisci", "modifica", "aggiorna", "perizia", "dati", "informazioni perizia", "gestione perizia"]
    },
    {
        "descrizione": "Mostra i dettagli dell'attività",
        "parametri": {"id_finanziamento": "string", "id_attivita": "string"},
        "path": "dettaglio-attivita",
        "keywords": ["attività", "dettaglio", "vedi", "visualizza", "info attività", "informazioni attività"]
    },
    {
        "descrizione": "Mostra i dettagli della rata",
        "parametri": {"id_finanziamento": "string", "id_rata": "string"},
        "path": "dettaglio-rata",
        "keywords": ["rata", "dettaglio", "vedi", "mostra", "visualizza", "informazioni rata", "pagamento"]
    },
    {
        "descrizione": "Gestisci i finanziamenti",
        "parametri": {},
        "path": "gestisci-finanziamenti",
        "keywords": ["gestisci", "gestione", "amministra", "modifica", "controlla", "lista", "catalogo"]
    },
    {
        "descrizione": "Stipula il finanziamento",
        "parametri": {"id_finanziamento": "string"},
        "path": "stipula-finanziamento",
        "keywords": ["stipula", "firma", "contratto", "sottoscrivi", "attiva"]
    },
    {
        "descrizione": "Eroga il finanziamento",
        "parametri": {"id_finanziamento": "string"},
        "path": "erogazione-finanziamento",
        "keywords": ["eroga", "erogazione", "paga", "rilascia", "disponi", "invio"]
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
    # crea l'EntityRuler tramite la factory di spaCy
    if "entity_ruler" not in nlp_obj.pipe_names:
        ruler = nlp_obj.add_pipe(
            "entity_ruler", 
            config={"overwrite_ents": True}, 
            first=True  # lo mette in testa al pipeline
        )
    else:
        ruler = nlp_obj.get_pipe("entity_ruler")
    patterns = [
        # pattern che catturano vicino a parole chiave

        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": {"REGEX": "^fin"}},  # "fin" o "finanziamento"
                {"IS_SPACE": True, "OP": "*"}, # spazio opzionale
                {"IS_DIGIT": True},  # primo numero
                {"TEXT": {"REGEX": r"^[\-\./]$"}, "OP": "?"},  # opzionale separatore singolo
                {"IS_SPACE": True, "OP": "?"},  # opzionale spazio
                {"IS_DIGIT": True, "OP": "?"}   # secondo numero (facoltativo)
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
    # 2)
    for pname in api_params:
        if pname in found:
            continue
        # tentativi di trovare numero vicino a keyword del parametro
        keyword = pname.replace("id_", "").replace("_", " ")
        # es. cerca 'finanziamento 12345', 'id finanziamento 12345', 'finanziamento:12345'
        pattern_context = rf"(?:{keyword})\D{{0,6}}?(\d{{3,20}})|(?:id\W*{keyword})\D{{0,6}}?(\d{{3,20}})"
        m = re.search(pattern_context, text, flags=re.IGNORECASE)
        if m:
            num = m.group(1) or m.group(2)
            if num:
                found[pname] = num
    """ # 3)
    for pname in api_params:
        if pname not in found:
            m = re.search(r"(\d{3,20})", text)
            if m:
                found[pname] = m.group(1) """
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

# --- Funzione principale che cerca l'API e, se necessario, estrae parametri
def get_api(testo_input: str, soglia_similarita=0.5, peso_keyword=0.4, peso_embedding=0.6) -> Dict[str, Any]:
    """
    Restituisce l'API più probabile in base a:
    1) keyword specifiche nel testo
    2) similarità embedding tra testo e descrizione API
    3) estrazione parametri richiesti
    """
    try:
        testo_lower = testo_input.lower()

        print(testo_lower)

        """ # Controllo lingua
        if detect(testo_input) != "it":
            return {"codice_risposta": "KO", "risposta_app": "Per favore fornisci il testo in italiano."} """

        # Embedding del testo utente
        embedding_input = model.encode([testo_input])

        migliori_match = {"path": None, "score": 0.0, "api": None, "match_type": None}

        for api in api_catalog:
            keywords = api.get("keywords", [])
            # punteggio keyword
            if keywords:
                kw_score = 1.0 if any(kw.lower() in testo_lower for kw in keywords) else 0.0 # presenza/assenza
                # kw_score = sum(1 for kw in keywords if kw.lower() in testo_lower) / len(keywords) # frazione di keyword trovate
                kw_bonus = max((len(kw) / 20 for kw in keywords if kw.lower() in testo_lower), default=0) # bonus per keyword più lunghe
                kw_score += kw_bonus
            else:
                kw_score = 0.0

            # punteggio embedding
            embedding_descrizione = model.encode([api["descrizione"]])
            emb_score = cosine_similarity(embedding_input, embedding_descrizione)[0][0]

            # punteggio combinato
            score_combinato = peso_keyword * kw_score + peso_embedding * emb_score

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
                estratti = extract_params_from_text(testo_input, api["parametri"])
                mancanti = [p for p in api["parametri"].keys() if p not in estratti or not estratti[p]]
                if len(mancanti) == 0:
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
    return get_api(richiesta.richiesta_utente)
