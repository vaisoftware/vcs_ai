import re
from unidecode import unidecode
from core.models_loader import nlp_stanza

def normalizza_testo(testo: str) -> str:
    """
    Pipeline completa di normalizzazione di un testo in italiano.
    Passaggi:
    1. Pulizia (rimozione di URL, emoji, punteggiatura, numeri, ecc.)
    2. Normalizzazione accenti
    3. Tokenizzazione
    4. Lemmatizzazione
    5. Rimozione stopwords
    6. Lowercase
    """
    print(f"Testo originale: {testo}")

    testo = re.sub(r"http\S+|www\S+|https\S+", " ", testo)  # URL
    testo = re.sub(r"@\w+", " ", testo)                     # menzioni
    testo = re.sub(r"#\w+", " ", testo)                     # hashtag
    testo = re.sub(r"[^\w\sàèéìòóùÀÈÉÌÒÓÙ]", " ", testo)  # rimuove simboli, emoji, numeri
    testo = re.sub(r"\s+", " ", testo).strip()              # spazi multipli

    testo = unidecode(testo)  # converte tutti gli accenti a forme canoniche ASCII

    doc = nlp_stanza(testo)
    lemmi = [word.lemma.lower() for sent in doc.sentences for word in sent.words]

    stopwords_it = set([
        "il", "lo", "la", "i", "gli", "le", "un", "una", "uno",
        "di", "a", "da", "in", "su", "per", "con", "come", "tra", "fra",
        "del", "della", "dell", "dei", "degli", "delle",
        "al", "allo", "alla", "ai", "agli", "alle",
        "dal", "dallo", "dalla", "dai", "dagli", "dalle",
        "nel", "nello", "nella", "nei", "negli", "nelle",
        "sul", "sullo", "sulla", "sui", "sugli", "sulle",
        "e", "ed", "o", "od", "ma", "anche", "se", "che", "quando", "dove", "come", "cui",
        "non", "piu", "meno", "molto", "tanto", "troppo", "tutti", "tutto",
        "questo", "quello", "questa", "quella", "questi", "quelle",
        "io", "tu", "lui", "lei", "noi", "voi", "loro",
        "mi", "ti", "si", "ci", "vi", "ne", "gli", "le", "li",
        "sono", "era", "stato", "stare", "essere", "avere",
        "ho", "hai", "ha", "abbiamo", "avete", "hanno",
        "sia", "siano", "sarà", "saranno", "può", "puoi", "può", "posso", "possono"
    ])
    tokens = [lemma for lemma in lemmi if lemma not in stopwords_it and lemma.strip() != ""]

    testo_normalizzato = " ".join(tokens)
    testo_normalizzato = testo_normalizzato.lower()
    print(f"Testo normalizzato: {testo_normalizzato}")

    return testo_normalizzato
