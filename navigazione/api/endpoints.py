from fastapi import APIRouter
from schemas.request_models import Richiesta
from services.api_matcher import get_api

router = APIRouter()

@router.post("/ai")
def get_api_endpoint(richiesta: Richiesta):
    """
    Endpoint principale per elaborare una richiesta utente
    e restituire l'API più rilevante.
    """
    return get_api(richiesta.richiesta_utente, richiesta.id_finanziamento)
