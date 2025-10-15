from fastapi import APIRouter, Depends, HTTPException
#from sqlalchemy.orm import Session
from . import schemas
from .services.api_finder_service import ApiFinderService
#from .database import get_db

# Initialize the router
router = APIRouter()

# Create a new task endpoint
@router.post("/api-finder", response_model=schemas.Risposta)
def get_api_from_prompt(richiesta: schemas.Richiesta):
    prompt = richiesta.richiesta_utente
    
    print(prompt)

    apiFinderService = ApiFinderService()
    
    return apiFinderService.get_api(testo_input=prompt)
