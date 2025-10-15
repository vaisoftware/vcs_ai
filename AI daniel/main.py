# Avviare il server
# python -m uvicorn main:app --host 0.0.0.0 --port 8080
# Ora l’API sarà disponibile su: http://localhost:8080/ai

from fastapi import FastAPI
#from .database import engine
#from .models import Base
from .routes import router as ai_router

# Create all database tables
#Base.metadata.create_all(bind=engine)

# Initialize the FastAPI app
app = FastAPI()

# Include the task router with the specified prefix and tags
app.include_router(ai_router, prefix="/ai", tags=["ai_router"])

@app.get("/ai")
def read_root():
    return {"message": "Welcome to the AI based router"}