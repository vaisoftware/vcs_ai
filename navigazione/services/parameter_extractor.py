import re
from typing import Dict, Any
from core.models_loader import nlp

def extract_params_from_text(text: str, api_params: Dict[str, str]) -> Dict[str, Any]:
    """
    Restituisce un dict con i parametri trovati: {param_name: value}
    Strategia:
      1) usa nlp per trovare entità con label tipo ID_FINANZIAMENTO, ID_RATA, ...
      2) se non trovi, cerca con regex numeriche vicino a parole chiave rilevanti
      3) se ancora nulla, tenta un fallback: primo numero trovato (opzionale)
    """
    doc = nlp(text)
    found = {}
    # 1)
    for ent in doc.ents:
        if ent.label_ in ("ID_FINANZIAMENTO", "ID_RATA", "ID_ATTIVITA"):
            print(f"Trovata entità: {ent.text} con label {ent.label_}")
            clean_text = re.sub(r'\D+', '', ent.text)
            if clean_text:
                if ent.label_ == "ID_FINANZIAMENTO" and "id_finanziamento" in api_params:
                    found["id_finanziamento"] = clean_text.zfill(8)
                elif ent.label_ == "ID_RATA" and "id_rata" in api_params:
                    found["id_rata"] = clean_text
                elif ent.label_ == "ID_ATTIVITA" and "id_attivita" in api_params:
                    found["id_attivita"] = clean_text
    if all(p in found for p in api_params):
        return found
    # 2A) cerca pattern specifici "stipula/eroga/fin <numero>"
    text_low = text.lower()
    m = re.search(r"\b(stipula|eroga|fin|finanziamento)\b[\s\:\-]*([0-9]+(?:[\/\-\.\s][0-9]+)*)", text_low)
    if m:
        verb = m.group(1)
        num = re.sub(r'\D+', '', m.group(2))
        if num:
            if "id_finanziamento" in api_params and "id_finanziamento" not in found:
                found["id_finanziamento"] = num.zfill(8)
            # possibile estensione: mappare ad altri parametri in base al verbo
    # 2B) cerca pattern generici "id <parametro> <numero>"
    for pname in api_params:
        if pname in found:
            continue
        keyword = pname.replace("id_", "").replace("_", " ")
        pattern_context = rf"(?:{keyword})\W{{0,6}}?(\d{{1,20}})|(?:id\W*{keyword})\W{{0,6}}?(\d{{1,20}})"
        m2 = re.search(pattern_context, text_low, flags=re.IGNORECASE)
        if m2:
            num = m2.group(1) or m2.group(2)
            if num:
                if pname == "id_finanziamento":
                    found[pname] = re.sub(r'\D+', '', num).zfill(8)
                else:
                    found[pname] = re.sub(r'\D+', '', num)
    # 3)
    for pname in api_params:
        if pname not in found:
            m3 = re.search(r"(\d{1,20})", text_low)
            if m3:
                if pname == "id_finanziamento":
                    found[pname] = m3.group(1).zfill(8)
                else:
                    found[pname] = m3.group(1)
    return found
