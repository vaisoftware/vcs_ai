""" # Libreria per connettersi a PostgreSQL
import psycopg2 """
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
        #la keyword "finanziamenti" non ha senso perché nella normalizzazione il plurale diventa singolare
        #e "finanziamento" è troppo generico e porta a falsi positivi
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
        #esempio: "erogazione 12345" non viene riconosciuto per il valore dell'embedding troppo basso
    }
]