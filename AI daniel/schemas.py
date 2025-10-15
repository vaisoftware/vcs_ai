from pydantic import BaseModel

# Definisce il modello di richiesta
class Richiesta(BaseModel):
    richiesta_utente: str

class Risposta(BaseModel):
    codice_risposta: str
    risposta_app: str
