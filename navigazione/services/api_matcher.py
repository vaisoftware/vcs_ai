from sklearn.metrics.pairwise import cosine_similarity
from core.models_loader import model
from services.api_catalog import api_catalog
from services.text_normalizer import normalizza_testo
from services.parameter_extractor import extract_params_from_text
from core.config import SOGLIA_SIM, PESO_KEYWORD, PESO_EMBEDDING

def get_api(testo_input: str, id_finanziamento: str):
    """
    Restituisce l'API più probabile in base a:
    1) keyword specifiche nel testo
    2) similarità embedding tra testo e descrizione API
    3) estrazione parametri richiesti
    """
    try:
        """ # Controllo lingua
        if detect(testo_input) != "it":
            return {"codice_risposta": "KO", "risposta_app": "Per favore fornisci il testo in italiano."} """
        testo_originale = testo_input
        testo_input = normalizza_testo(testo_input)

        # Embedding del testo utente
        embedding_input = model.encode([testo_input])

        migliori_match = {"path": None, "score": 0.0, "api": None, "match_type": None}

        for api in api_catalog:  
            keywords = api.get("keywords", [])
            # punteggio keyword
            if keywords:
                kw_score = 1.0 if any(kw.lower() in testo_input for kw in keywords) else 0.0 # presenza/assenza
                # kw_score = sum(1 for kw in keywords if kw.lower() in testo_input) / len(keywords) # frazione di keyword trovate
                # kw_bonus = max((len(kw) / 20 for kw in keywords if kw.lower() in testo_input), default=0) # bonus per keyword più lunghe
                # kw_score += kw_bonus
            else:
                kw_score = 0.0

            # punteggio embedding
            embedding_descrizione = normalizza_testo(api["descrizione"])
            embedding_descrizione = model.encode([embedding_descrizione])
            emb_score = cosine_similarity(embedding_input, embedding_descrizione)[0][0]

            # punteggio combinato
            score_combinato = PESO_KEYWORD * kw_score + PESO_EMBEDDING * emb_score

            print(f"  kw_score: {kw_score}")
            print(f"  emb_score: {emb_score}")
            print(f"  score_combinato: {score_combinato}")

            if score_combinato > migliori_match["score"]:
                match_type = "keyword+embedding" if kw_score > 0 else "embedding"
                migliori_match = {
                    "path": api["path"],
                    "score": float(score_combinato),
                    "api": api,
                    "match_type": match_type
                }

        # Se superiamo la soglia, estraiamo parametri
        if migliori_match["score"] >= SOGLIA_SIM:
            api = migliori_match["api"]
            response = {
                "codice_risposta": "OK",
                "path": api["path"],
                "score": round(migliori_match["score"], 4),
                "match_type": migliori_match["match_type"]
            }

            if api.get("parametri"):
                estratti = extract_params_from_text(testo_originale, api["parametri"])
                mancanti = [p for p in api["parametri"].keys() if p not in estratti or not estratti[p]]
                if len(mancanti) == 0:
                    response["parametri"] = estratti
                else:
                    #TODO: gestione specifica per id_finanziamento passato separatamente
                    if "id_finanziamento" in mancanti and id_finanziamento:
                        print("Usando id_finanziamento passato separatamente.")
                        estratti["id_finanziamento"] = id_finanziamento
                        mancanti.remove("id_finanziamento")
                        response["parametri"] = estratti
                    else:
                        response["parametri_parziali"] = estratti
                        response["mancanti"] = mancanti
                        response["avviso"] = "Parametri mancanti o incerti. Fornisci gli id richiesti."
            return response

        return {"codice_risposta": "KO", "risposta_app": "La richiesta non trova corrispondenza con nessuna API.", "score": round(migliori_match["score"], 4)}

    except Exception as e:
        return {"codice_risposta": "KO", "risposta_app": f"Errore durante l'elaborazione: {str(e)}"}