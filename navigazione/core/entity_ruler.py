def init_entity_ruler(nlp_obj):
    """
    Aggiunge un EntityRuler con pattern utili per riconoscere contesti tipo:
    'finanziamento 12345', 'id finanziamento: 12345', 'id 12345', ecc.
    """
    # Se esiste già un entity_ruler, riutilizzalo; altrimenti crealo
    if "entity_ruler" not in nlp_obj.pipe_names:
        if "ner" in nlp_obj.pipe_names: #se c'è la NER, mettiamo l'EntityRuler prima
            ruler = nlp_obj.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
        else: #altrimenti lo aggiungiamo alla fine
            ruler = nlp_obj.add_pipe("entity_ruler", config={"overwrite_ents": True})
    else:
        ruler = nlp_obj.get_pipe("entity_ruler")

    patterns = [
        # === PATTERN 1: caso tokenizzato ===
        # es: "finanziamento 25-81", "fin 25 / 81"
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": {"REGEX": "^fin"}},       # "fin" o "finanziamento"
                {"IS_SPACE": True, "OP": "*"},      # spazi opzionali
                {"IS_DIGIT": True},                 # primo numero
                {"TEXT": {"REGEX": r"^[\-\./]$"}, "OP": "?"},  # separatore opzionale
                {"IS_SPACE": True, "OP": "?"},      # spazio opzionale
                {"IS_DIGIT": True, "OP": "?"}       # secondo numero opzionale
            ]
        },
        # === PATTERN 2: caso NON tokenizzato ===
        # es: "finanziamento 25/81", "fin 25.81", "finanziamento 25-81"
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": {"REGEX": "^fin"}},       # "fin" o "finanziamento"
                {"IS_SPACE": True, "OP": "*"},
                {"TEXT": {"REGEX": r"^\d+(?:[\-\./\s]?\d+)*$"}}  # numeri concatenati o con separatori
            ]
        },
        # === (Facoltativo) pattern per "id finanziamento" ===
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": "id"},
                {"IS_SPACE": True, "OP": "*"},
                {"LOWER": {"REGEX": "^fin"}},
                {"IS_PUNCT": True, "OP": "?"},
                {"IS_SPACE": True, "OP": "*"},
                {"TEXT": {"REGEX": r"^\d+(?:[\-\./\s]?\d+)*$"}}
            ]
        },

        # === Pattern 3: "stipula" o "eroga" seguiti da un numero ===
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": {"REGEX": "^(stipula|eroga)$"}},  # "stipula" o "eroga"
                {"IS_SPACE": True, "OP": "*"},              # spazi opzionali
                {"IS_DIGIT": True}                          # numero
            ]
        },

        # === Pattern 4: "stipula" o "eroga" seguiti da numeri con separatori ===
        {
            "label": "ID_FINANZIAMENTO",
            "pattern": [
                {"LOWER": {"REGEX": "^(stipula|eroga)$"}},  # "stipula" o "eroga"
                {"IS_SPACE": True, "OP": "*"},              # spazi opzionali
                {"TEXT": {"REGEX": r"^\d+(?:[\-\./\s]?\d+)*$"}}  # numeri concatenati o con separatori
            ]
        },

        # da qui in poi, da adattare/aggiungere secondo le esigenze specifiche
        {"label": "ID_RATA", "pattern": [{"TEXT": {"REGEX": "^rat.*"}}, {"IS_SPACE": True, "OP": "?"}, {"TEXT": {"REGEX": "^[0-9]+([\\s\\-\\./][0-9]+)*$"}}]},
        {"label": "ID_RATA", "pattern": [{"LOWER": "rata"}, {"IS_DIGIT": True}]},
        {"label": "ID_RATA", "pattern": [{"LOWER": "id"}, {"LOWER": "rata"}, {"IS_PUNCT": True, "OP": "?"}, {"IS_DIGIT": True}]},
        {"label": "ID_RATA", "pattern": [{"LOWER": "id"}, {"IS_DIGIT": True}]},
        
        {"label": "ID_ATTIVITA", "pattern": [{"TEXT": {"REGEX": "^att.*"}}, {"IS_SPACE": True, "OP": "?"}, {"TEXT": {"REGEX": "^[0-9]+([\\s\\-\\./][0-9]+)*$"}}]},
        {"label": "ID_ATTIVITA", "pattern": [{"LOWER": "attivita"}, {"IS_DIGIT": True}]},
        {"label": "ID_ATTIVITA", "pattern": [{"LOWER": "id"}, {"LOWER": "attivita"}, {"IS_PUNCT": True, "OP": "?"}, {"IS_DIGIT": True}]},
        {"label": "ID_ATTIVITA", "pattern": [{"LOWER": "id"}, {"IS_DIGIT": True}]},

        # pattern semplici per numeri che seguono ':' o '=' o 'n.'
        {"label": "NUMBER_GENERIC", "pattern": [{"IS_ASCII": True, "OP": "?"}, {"IS_DIGIT": True}]},

        #TODO: aggiungere altri patterns specifici del tuo dominio
    ]
    ruler.add_patterns(patterns)
