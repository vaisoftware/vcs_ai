from pydantic import BaseModel

class Richiesta(BaseModel):
    richiesta_utente: str
    id_finanziamento: str