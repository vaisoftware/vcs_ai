# Libreria che consente la rappresentazione vettoriale (embedded) di frasi anziché parole
from sentence_transformers import SentenceTransformer
# Libreria per calcolare la similarità coseno tra vettori (la similiarità coseno misura quanto due vettori sono simili)
from sklearn.metrics.pairwise import cosine_similarity
# Libreria per il rilevamento della lingua del testo
from langdetect import detect

from .. import schemas

class ApiFinderService:
    
    # carica il modello come variabile statica (troppo grosso caricarlo ad ogni istanza della classe)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    api_catalog = [
        {
            "descrizione": "Creazione nuovo finanziamento. Creazione per tipo e ndg",
            "endpoint": "api/finanziamento",
            "verbo": "POST",
        },
        {
            "descrizione": "Ricerca un finanziamento esistente. Ricerca per tipo e ndg",
            "endpoint": "api/finanziamento&tipo=mutuo&ndg=123456",
            "verbo": "GET",
        }
    ]

    """ # se leggiamo da db allora va tolto api_catalog come variabile statica
    # Funzione per caricare le API da un database PostgreSQL
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
    api_catalog = carica_api_da_db() 
    """

    def __init__(self):
        pass    
        
    def get_api(testo_input: str, soglia_similarita=0.5) -> schemas.Risposta:
        try:
            # Verifica che la lingua sia italiana
            if detect(testo_input) != "it":
                return schemas.Risposta(codice_risposta="KO", risposta_app="Per favore fornisci il testo in italiano.")

            # Calcola embedding del testo utente
            embedding_input = ApiFinderService.model.encode([testo_input])

            migliori_match = {"endpoint": None, "score": 0.0}

            # Confronta con le descrizioni delle API
            for api in ApiFinderService.api_catalog:
                # Calcola embedding della descrizione dell'API
                embedding_descrizione = ApiFinderService.model.encode([api["descrizione"]])
                
                # Calcola la similarità coseno tra l'input e la descrizione dell'API
                sim = cosine_similarity(embedding_input, embedding_descrizione)[0][0]

                # Aggiorna il miglior match se la similarità è maggiore della soglia
                if sim > migliori_match["score"]:
                    migliori_match = {"endpoint": api["endpoint"], "score": sim}

            # Controlla se il miglior match supera la soglia di similarità
            if migliori_match["score"] >= soglia_similarita:
                return schemas.Risposta(codice_risposta="OK", risposta_app=f"{migliori_match['endpoint']}")
            else:
                return schemas.Risposta(codice_risposta="KO", risposta_app="La richiesta non trova corrispondenza con nessuna API. Riprova.")

        except Exception as e:
            return schemas.Risposta(codice_risposta="KO", risposta_app=f"Errore durante l'elaborazione: {str(e)}")
            