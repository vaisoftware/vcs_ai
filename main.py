# Avviare il server
# python -m uvicorn main:app --host 0.0.0.0 --port 8080
# Ora l’API sarà disponibile su: http://localhost:8080/ai

# Libreria che consente la rappresentazione vettoriale (embedded) di frasi anziché parole
from sentence_transformers import SentenceTransformer
# Libreria per calcolare la similarità coseno tra vettori (la similiarità coseno misura quanto due vettori sono simili)
from sklearn.metrics.pairwise import cosine_similarity
# Libreria per il rilevamento della lingua del testo
from langdetect import detect
# Libreria per l'elaborazione del linguaggio naturale
import spacy
# Libreria per creare esempi di addestramento
from spacy.training.example import Example 
# Libreria per mescolare i dati di addestramento
import random 
# Libreria per gestire i percorsi dei file
from pathlib import Path 

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch

# Libreria per creare API web
from fastapi import FastAPI
# Libreria per la definizione di modelli di dati
from pydantic import BaseModel
""" # Libreria per connettersi a PostgreSQL
import psycopg2 """
# Inizializza l'app FastAPI
app = FastAPI()
# Definisce il modello di richiesta
class Richiesta(BaseModel):
    richiesta_utente: str
""" # Funzione per caricare le API da un database PostgreSQL
def carica_api_da_db():
    conn = psycopg2.connect(
        host="HOST",
        database="DBNAME",
        user="USER",
        password="PASSWORD"
    )
    cur = conn.cursor()
    cur.execute("SELECT descrizione, endpoint FROM api_catalog")
    risultati = cur.fetchall()
    cur.close()
    conn.close()

    api_catalog_db = [{"descrizione": r[0], "endpoint": r[1], "verbo": r[2]} for r in risultati]
    return api_catalog_db

# Carica le API dal database
api_catalog = carica_api_da_db() """

# Lista statica di API con descrizioni ed endpoint
api_catalog = [
    # implementare la seguente api
    {
        "descrizione": "Crea un nuovo finanziamento con ndg, prodotto, data e convenzione",
        "endpoint": "http://80.88.88.48:8080/accensione-finanziamento",
        "verbo": "GET", # sarebbe una post, ma andrebbe modificato il backend
        "parametri": {"ndg": "string", "prodotto": "string", "data": "string", "convenzione": "string"},
    },
    # api implementata, ma in corrispondenza dell'url http://80.88.88.48:8080/accensione-finanziamento/elaborazione-primaria
    {
        "descrizione": "Crea un nuovo finanziamento con ndg, prodotto, data, convenzione, importo e conto corrente",
        "endpoint": "http://80.88.88.48:8080/accensione-finanziamento/elaborazione-primaria",
        "verbo": "POST",
        "parametri": {"ndg": "string", "prodotto": "string", "data": "string", "convenzione": "string", "importo": "string", "conto_corrente": "string"},
    },
]

# Carica modello multilingua potente che comprende l'italiano
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

""" # Setup del modello Hugging Face
model_id = "m-polignano/ANITA-NEXT-24B-Dolphin-Mistral-UNCENSORED-ITA"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer) """
 
#def estrai_parametri_llm(testo_input, parametri_attesi): 
    #prompt = f"""
    #Estrai i seguenti parametri dal testo: {parametri_attesi}.
    #Rispondi SOLO in JSON con i campi trovati.
    #Testo: "{testo_input}"
    #    """.strip()
    #try:
    #    output = generator(prompt, max_new_tokens=300, do_sample=False, temperature=0)[0]["generated_text"]
    #    # Estrai solo la parte JSON dal testo generato
    #    start = output.find("{")
    #    end = output.rfind("}") + 1
    #    json_text = output[start:end]
    #    return json.loads(json_text)
    #except Exception as e:
    #    print("Errore nella decodifica:", e)
    #    return {}

def get_api(testo_input, soglia_similarita=0.5):
    try:
        # Verifica che la lingua sia italiana
        if detect(testo_input) != "it":
            return "Per favore fornisci il testo in italiano."

        # Calcola embedding del testo utente
        embedding_input = model.encode([testo_input])

        migliori_match = {"endpoint": None, "score": 0.0, "parametri": {}}

        # Confronta con le descrizioni delle API
        for api in api_catalog:
            # Calcola embedding della descrizione dell'API
            embedding_descrizione = model.encode([api["descrizione"]])
            # Calcola la similarità coseno tra l'input e la descrizione dell'API
            sim = cosine_similarity(embedding_input, embedding_descrizione)[0][0]

            # Aggiorna il miglior match se la similarità è maggiore della soglia
            if sim > migliori_match["score"]:
                migliori_match = {"endpoint": api["endpoint"], "score": sim, "parametri": api["parametri"]}

        # Controlla se il miglior match supera la soglia di similarità
        if migliori_match["score"] >= soglia_similarita:
            # Estrai i parametri richiesti usando LLM
            parametri_attesi = migliori_match["parametri"]
            """ parametri_estratti = estrai_parametri_llm(testo_input, parametri_attesi)
            migliori_match["parametri"] = parametri_estratti """
            return f"{migliori_match['endpoint']}"
        else:
            return "La richiesta non trova corrispondenza con nessuna API. Riprova."

    except Exception as e:
        return f"Errore durante l'elaborazione: {str(e)}"


@app.post("/ai")
def get_api_endpoint(richiesta: Richiesta):
    return {"endpoint": get_api(richiesta.richiesta_utente)}
