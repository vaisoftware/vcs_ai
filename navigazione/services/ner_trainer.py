import spacy, random

def train_spacy_ner(base_model="it_core_news_sm", training_data=None, n_iter=30):
    """
    training_data: lista di (text, {"entities": [(start, end, label), ...]})
    Esempio:
      training_data = [
        ("voglio fare la stipula del finanziamento 123456", {"entities": [(31, 37, "ID_FINANZIAMENTO")]}),
        ...
      ]
    Nota: esegui questa funzione localmente per migliorare il riconoscimento.
    """
    if training_data is None or len(training_data) == 0:
        raise ValueError("Serve training_data con esempi annotati")
    # carica modello di base o blank
    try:
        nlp_train = spacy.load(base_model)
    except Exception:
        nlp_train = spacy.blank("it")
    # controlla se il modello ha già un NER, altrimenti lo aggiunge
    if "ner" not in nlp_train.pipe_names:
        ner = nlp_train.add_pipe("ner", last=True)
    else:
        ner = nlp_train.get_pipe("ner")
    # estrae tutte le label nuove presenti negli esempi di training
    labels = set([ent[2] for _, ann in training_data for ent in ann.get("entities", [])])
    # le aggiunge al NER, così il modello saprà che deve imparare a riconoscerle
    for lbl in labels:
        ner.add_label(lbl)
    # disattiva temporaneamente tutti i componenti del pipeline tranne il NER, per addestrare solo il modello di riconoscimento entità senza aggiornare gli altri moduli
    other_pipes = [p for p in nlp_train.pipe_names if p != "ner"]
    with nlp_train.disable_pipes(*other_pipes):
        # un optimizer è un oggetto che aggiorna i pesi del modello per minimizzare la funzione di perdita
        optimizer = nlp_train.begin_training()
        for itn in range(n_iter): # n_iter → numero di epoche
            # mescola gli esempi ad ogni epoca, così il modello non si abitua all’ordine dei dati
            random.shuffle(training_data)
            losses = {}
            # crea mini-batch dal training set, partendo da 4 esempi e aumentando gradualmente fino a 32
            # (piccoli batch all’inizio per precisione → grandi batch alla fine per velocità, senza sacrificare la qualità dell’addestramento)
            batches = spacy.util.minibatch(training_data, size=spacy.util.compounding(4.0, 32.0, 1.001))
            # aggiornamento del modello
            for batch in batches:
                # separa dai batch di training i testi dalle annotazioni per poterli passare separatamente al modello NER
                texts, annotations = zip(*batch)
                # aggiorna il modello NER con i nuovi dati
                nlp_train.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
                # sgd → Stochastic Gradient Descent, algoritmo di ottimizzazione
                # drop → tasso di dropout, cioè la probabilità di ignorare casualmente alcune unità durante l’addestramento per evitare overfitting
                # losses → dizionario che tiene traccia delle perdite durante l'addestramento
            # Opzionale: monitorare le perdite: print(itn, losses)
    # salva modello su disco per riutilizzare
    nlp_train.to_disk("./ner_finanziamenti_model")
    return "./ner_finanziamenti_model"
