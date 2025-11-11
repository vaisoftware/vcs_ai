from sentence_transformers import SentenceTransformer
import spacy
import stanza

from core.config import MODEL_NAME, SPACY_MODEL, STANZA_LANG
from core.entity_ruler import init_entity_ruler

# SentenceTransformer
model = SentenceTransformer(MODEL_NAME)

# spaCy
try:
    nlp = spacy.load(SPACY_MODEL)
except Exception:
    nlp = spacy.blank("it")

init_entity_ruler(nlp)

# Stanza
stanza.download(STANZA_LANG)
nlp_stanza = stanza.Pipeline(STANZA_LANG, processors="tokenize,mwt,pos,lemma", use_gpu=False)
