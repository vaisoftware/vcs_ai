from fastapi import FastAPI
from api.endpoints import router as ai_router

app = FastAPI(title="AI API Finder")

# Registra i router
app.include_router(ai_router)