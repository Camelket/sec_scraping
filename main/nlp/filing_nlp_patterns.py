from main.nlp.filing_nlp_constants import SECUQUANTITY_UNITS

def add_anchor_pattern_to_patterns(
    anchor_pattern: list[dict], patterns: list[list[dict]]
) -> list[list[dict]]:
    if not any(
        [False if entry["RIGHT_ID"] != "anchor" else True for entry in anchor_pattern]
    ):
        raise ValueError(
            "Anchor pattern must contain an anchor token with RIGHT_ID = 'anchor'"
        )
    finished_patterns = [None] * len(patterns)
    for idx, pattern in enumerate(patterns):
        finished_patterns[idx] = anchor_pattern + pattern
    return finished_patterns


ADJ_NEGATION_PATTERNS = [
    [
        {"DEP": "neg"},
        {"POS": "ADJ"}
    ],
    [
        {"POS": "ADJ"},
        {"DEP": "neg"}
    ],
]

VERB_NEGATION_PATTERNS = [
    [
        {"DEP": {"IN": ["auxpass", "aux"]}, "OP": "?"},
        {"DEP": "neg"},
        {"POS": "AUX", "OP": "?"},
        {"POS": "VERB"}
    ]
]




SECU_DEBT_SECURITY_L1_MODIFIERS = [
    "subordinated",
    "senior"
]

SECU_GENERAL_PRE_MODIFIERS = [
    "convertible"
]
SECU_GENERAL_PRE_COMPOUND_MODIFIERS = [
    [
        {"LOWER": "non"},
        {"LOWER": "-"},
        {"LOWER": "convertible"}
    ],
    [
        {"LOWER": "pre"},
        {"LOWER": "-"},
        {"LOWER": "funded"}
    ],
]
SECU_GENERAL_AFFIXES = [
    "series",
    "tranche",
    "class"
]
SECU_PURCHASE_AFFIXES = [
    "stock",
    "shares"
]
SECU_PURCHASE_SUFFIXES = [
    "rights",
    "contracts",
    "units"
]


SECU_ENT_SPECIAL_PATTERNS = [
    [{"LOWER": "common"}, {"LOWER": "units"}, {"LOWER": "of"}, {"LOWER": "beneficial"}, {"LOWER": "interest"}],
    [{"TEXT": "Subsciption"}, {"TEXT": "Rights"}],
]
# exclude particles, conjunctions from regex match 
SECU_ENT_DEPOSITARY_PATTERNS = [
    [   {"LOWER": {"IN": SECU_GENERAL_AFFIXES}}, {"TEXT": {"REGEX": "[a-zA-Z0-9]{1,3}", "NOT_IN": ["of"]}, "OP": "?"},
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["depository", "depositary"]}},
        {"LOWER": {"IN": ["shares"]}}
    ]
        ,
    [
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["preferred", "common", "depository", "depositary", "ordinary"]}},
        {"LOWER": {"IN": ["shares"]}}
    ]
        ,
    *[  
        [
            *general_pre_sec_compound_modifier,
            {"LOWER": {"IN": ["depository", "depositary"]}},
            {"LOWER": {"IN": ["shares"]}, "OP": "?"}
        ] for general_pre_sec_compound_modifier in SECU_GENERAL_PRE_COMPOUND_MODIFIERS
    ]
        ,
    
    [   {"LOWER": {"IN": SECU_GENERAL_AFFIXES}}, {"TEXT": {"REGEX": "[a-zA-Z0-9]{1,3}", "NOT_IN": ["of"]}, "OP": "?"},
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["depository", "depositary"]}, "OP": "?"},
        {"LOWER": {"IN": ["shares", "stock"]}},
        {"LOWER": {"IN": ["options", "option"]}}
    ]
        ,
    [
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["depository", "depositary"]}, "OP": "?"},
        {"LOWER": {"IN": ["shares", "stock"]}},
        {"LOWER": {"IN": ["options", "option"]}}
    ]

]

SECU_ENT_REGULAR_PATTERNS = [
    [   {"LOWER": {"IN": SECU_GENERAL_AFFIXES}}, {"TEXT": {"REGEX": "[a-zA-Z0-9]{1,3}", "NOT_IN": ["of"]}, "OP": "?"},
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["preferred", "common", "ordinary"]}},
        {"LOWER": {"IN": ["stock"]}}
    ]
        ,
    [
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["preferred", "common", "ordinary"]}},
        {"LOWER": {"IN": ["stock"]}}
    ]
        ,
    *[  
        [
            *general_pre_sec_compound_modifier,
            {"LOWER": {"IN": ["preferred", "common", "warrant", "warrants", "ordinary"]}},
            {"LOWER": {"IN": ["stock"]}, "OP": "?"}
        ] for general_pre_sec_compound_modifier in SECU_GENERAL_PRE_COMPOUND_MODIFIERS
    ]
        ,
    
    [   {"LOWER": {"IN": SECU_GENERAL_AFFIXES}}, {"TEXT": {"REGEX": "[a-zA-Z0-9]{1,3}", "NOT_IN": ["of"]}, "OP": "?"},
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["preferred", "common", "ordinary"]}, "OP": "?"},
        {"LOWER": {"IN": ["stock"]}},
        {"LOWER": {"IN": ["options", "option"]}}
    ]
        ,
    [
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["preferred", "common", "ordinary"]}, "OP": "?"},
        {"LOWER": {"IN": ["stock"]}},
        {"LOWER": {"IN": ["options", "option"]}}
    ]

        ,
    
    [   {"LOWER": {"IN": SECU_GENERAL_AFFIXES}}, {"TEXT": {"REGEX": "[a-zA-Z0-9]{1,3}", "NOT_IN": ["of"]}, "OP": "?"},
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["warrant", "warrants"]}}
    ]
        ,
    [
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": ["warrant", "warrants"]}},
        {"LOWER": {"IN": ["stock"]}, "OP": "?"}
    ]
        ,

    [   {"LOWER": {"IN": SECU_GENERAL_AFFIXES}}, {"TEXT": {"REGEX": "[a-zA-Z0-9]{1,3}", "NOT_IN": ["of"]}, "OP": "?"},
        {"LOWER": {"IN": SECU_DEBT_SECURITY_L1_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": "debt"}, {"LOWER": "securities"}
    ]
        ,

    [   {"LOWER": {"IN": SECU_DEBT_SECURITY_L1_MODIFIERS}, "OP": "?"},
        {"LOWER": {"IN": SECU_GENERAL_PRE_MODIFIERS}, "OP": "?"},
        {"LOWER": "debt"}, {"LOWER": "securities"}
    ]
        ,

    [   {"LOWER": {"IN": SECU_PURCHASE_AFFIXES}, "OP": "?"}, {"TEXT": "Purchase"}, {"LOWER": {"IN": SECU_PURCHASE_SUFFIXES}}
    ]
        ,            
]


SECUQUANTITY_BASE_CASES = [
    {"ENT_TYPE": {"IN": ["CARDINAL", "MONEY"]}, "OP": "+"},
    {"POS": "NUM", "DEP": "nummod", "OP": "+"},
]

SECUQUANTITY_ENT_PATTERNS = [
    
        *[[
            base_case,
            {"LOWER": {"IN": ["authorized", "outstanding"]}, "OP": "?"},
            {"LOWER": {"IN": ["share", "shares", "warrant shares"]}}
        ] for base_case in SECUQUANTITY_BASE_CASES],
        *[[
            base_case,
            {"POS": "ADJ", "OP": "?"},
            {"LOWER": {"IN": ["shares"]}},
            {"POS": "ADJ", "OP": "?"},
        ] for base_case in SECUQUANTITY_BASE_CASES],
        *[[   
            base_case,
            {"LOWER": "of", "OP": "?"},
            {"LOWER": "our", "OP": "?"},
            {"ENT_TYPE": {"IN": ["SECU", "SECUREF"]}}
        ] for base_case in SECUQUANTITY_BASE_CASES],
        *[[
            base_case,
            {"LOWER": "shares"},
            {"OP": "*", "IS_SENT_START": False, "POS": {"NOT_IN": ["VERB"]}, "ENT_TYPE": {"NOT_IN": ["SECU", "SECUQUANTITY"]}},
            {"LOWER": "of", "OP": "?"},
            {"LOWER": "our", "OP": "?"},
            {"ENT_TYPE": {"IN": ["SECU"]}}
        ] for base_case in SECUQUANTITY_BASE_CASES],
        *[[
            base_case,
            {"POS": "ADJ", "OP": "*"},
            {"LOWER": "shares"},
            {"LOWER": "of"},
            {"ENT_TYPE": {"IN": ["SECU", "SECUREF"]}}
        ] for base_case in SECUQUANTITY_BASE_CASES],

]



SECU_EXPIRY_PATTERNS = [
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": "expire"}, 
        },
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "on"}, 
        },
        {
            "LEFT_ID": "prep1",
            "REL_OP": ">",
            "RIGHT_ID": "pobj1",
            "RIGHT_ATTRS": {"DEP": "pobj"}, 
        },
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}, 
        },
        {
            "LEFT_ID": "verb1",
            "REL_OP": "<<",
            "RIGHT_ID": "verb2",
            "RIGHT_ATTRS": {"POS": "VERB"}, 
        },
        {
            "LEFT_ID": "verb2",
            "REL_OP": ">",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "on"}, 
        },
        {
            "LEFT_ID": "prep1",
            "REL_OP": ">",
            "RIGHT_ID": "pobj1",
            "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": "DATE"}, 
        },
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}, 
        },
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">>",
            "RIGHT_ID": "verb2",
            "RIGHT_ATTRS": {"POS": "VERB"}, 
        },
        {
            "LEFT_ID": "verb2",
            "REL_OP": ">",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "on"}, 
        },
        {
            "LEFT_ID": "prep1",
            "REL_OP": ">",
            "RIGHT_ID": "pobj1",
            "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": "DATE"}, 
        },
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "verb2",
            "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": "exercise"}, 
        },
        {
            "LEFT_ID": "verb2",
            "REL_OP": ">>",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "up"}, 
        },
        {
            "LEFT_ID": "prep1",
            "REL_OP": ">",
            "RIGHT_ID": "prep2",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "to"}, 
        },
        {
            "LEFT_ID": "prep2",
            "REL_OP": ".*",
            "RIGHT_ID": "prep3",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "on"}, 
        },
        {
            "LEFT_ID": "prep3",
            "REL_OP": ">",
            "RIGHT_ID": "pobj1",
            "RIGHT_ATTRS": {"DEP": "pobj", "LEMMA": "date"}, 
        },
        {
            "LEFT_ID": "pobj1",
            "REL_OP": ">",
            "RIGHT_ID": "verb3",
            "RIGHT_ATTRS": {"LEMMA": "be"}, 
        },
        {
            "LEFT_ID": "verb3",
            "REL_OP": ">",
            "RIGHT_ID": "attr1",
            "RIGHT_ATTRS": {"DEP": "attr"}, 
        },
        {
            "LEFT_ID": "attr1",
            "REL_OP": ">>",
            "RIGHT_ID": "issuance1",
            "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORDINAL", "CARDINAL"]}}, 
        },
    ]
]

PRICE_TRANSFORM_COMPOUND_TO_PRICE_LEMMA = [
    "exercise",
    "convert",
    "conversion",
    "redeem",
    "redemption",
    "issuance",
]

SECU_EXERCISE_PRICE_PREP_MONEY_DOLLAR_VARIANTS = [
    [
        {
            "LEFT_ID": "prep1",
            "REL_OP": ">",
            "RIGHT_ID": "pobj_CD",
            "RIGHT_ATTRS": {"DEP": "pobj", "POS": "NUM"}
        },
        {
            "LEFT_ID": "pobj_CD",
            "REL_OP": ">",
            "RIGHT_ID": "currency_symbol",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nmod", "nummod"]}, "LOWER": {"IN": ["$"]}}
        } 
    ],
    [
        {
            "LEFT_ID": "prep1",
            "REL_OP": ">",
            "RIGHT_ID": "currency_symbol",
            "RIGHT_ATTRS": {"DEP": "pobj", "LOWER": {"IN": ["$"]}}
        },
        {
            "LEFT_ID": "currency_symbol",
            "REL_OP": ">",
            "RIGHT_ID": "pobj_CD",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nmod", "nummod"]}, "POS": "NUM"}
        } 
    ],

]
SECU_EXERCISE_PRICE_FIRST_VERB_VARIANTS = [
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["have", "be", "purchase", "issuable"]}}, 
        },
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["have", "be", "purchase", "issuable"]}}, 
        },
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">>",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["have", "be", "purchase", "issuable"]}}, 
        },
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<<",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["have", "be", "purchase", "issuable"]}}, 
        },
    ]
]
INCOMPLETE_SECU_EXERCISE_PRICE_PATTERNS = [
    [
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">>",
            "RIGHT_ID": "prepverb1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": {"IN": ["of", "at"]}}, 
        },
        {
            "LEFT_ID": "prepverb1",
            "REL_OP": ">",
            "RIGHT_ID": "price",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nobj", "pobj", "dobj"]}, "LOWER": "price"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "transformative",
            "RIGHT_ATTRS": {"DEP": "compound", "LEMMA": {"IN": PRICE_TRANSFORM_COMPOUND_TO_PRICE_LEMMA}}
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "of"}
        }
    ],
    [
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "conj",
            "RIGHT_ATTRS": {"DEP": "conj", "LEMMA": {"IN": ["remain"]}},
        },
        {
            "LEFT_ID": "conj",
            "REL_OP": ">",
            "RIGHT_ID": "prepverb1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": {"IN": ["of", "at"]}}, 
        },
        {
            "LEFT_ID": "prepverb1",
            "REL_OP": ">",
            "RIGHT_ID": "price",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nobj", "pobj", "dobj"]}, "LOWER": "price"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "transformative",
            "RIGHT_ATTRS": {"DEP": "compound", "LEMMA": {"IN": PRICE_TRANSFORM_COMPOUND_TO_PRICE_LEMMA}}
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "of"}
        },
    ],
    [
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "prepverb1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": {"IN": ["of", "at"]}}, 
        },
        {
            "LEFT_ID": "prepverb1",
            "REL_OP": ">",
            "RIGHT_ID": "price",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nobj", "pobj", "dobj"]}, "LOWER": "price"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "transformative",
            "RIGHT_ATTRS": {"DEP": "compound", "LEMMA": {"IN": PRICE_TRANSFORM_COMPOUND_TO_PRICE_LEMMA}}
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "of"}
        },
    ],
    [
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "prepverb1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": {"IN": ["of", "at"]}}, 
        },
        {
            "LEFT_ID": "prepverb1",
            "REL_OP": ">",
            "RIGHT_ID": "price",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nobj", "pobj", "dobj"]}, "LOWER": "price"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "transformative",
            "RIGHT_ATTRS": {"DEP": "compound", "LEMMA": {"IN": PRICE_TRANSFORM_COMPOUND_TO_PRICE_LEMMA}}
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "of"}
        },
    ],
    [
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "price",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nobj", "pobj", "dobj"]}, "LOWER": "price"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "transformative",
            "RIGHT_ATTRS": {"DEP": "compound", "LEMMA": {"IN": PRICE_TRANSFORM_COMPOUND_TO_PRICE_LEMMA}}
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "of"}
        },  
    ]
]

SECU_EXERCISE_PRICE_PATTERNS = [
    first_verb_lemma + incomplete + tail 
    for tail
    in SECU_EXERCISE_PRICE_PREP_MONEY_DOLLAR_VARIANTS
    for first_verb_lemma
    in SECU_EXERCISE_PRICE_FIRST_VERB_VARIANTS
    for incomplete
    in INCOMPLETE_SECU_EXERCISE_PRICE_PATTERNS
]

CORE_DATE_RELATION_CONTEXT_PATTERNS = [
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "of"}
        },
        {
            "LEFT_ID": "prep1",
            "REL_OP": "<",
            "RIGHT_ID": "prep_end",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "as"}
        },
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "prep1",
            "RIGHT_ATTRS": {"DEP": "pobj", "LOWER": "of"}
        },
        {
            "LEFT_ID": "prep1",
            "REL_OP": "<",
            "RIGHT_ID": "prep2",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "period"}
        },
        {
            "LEFT_ID": "prep2",
            "REL_OP": "<",
            "RIGHT_ID": "prep_end",
            "RIGHT_ATTRS": {"DEP": "pobj", "LOWER": "for"}
        },
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "prep_end",
            "RIGHT_ATTRS": {"DEP": "pobj", "LOWER": "on"}
        },
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<<",
            "RIGHT_ID": "prep_end",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "until"}
        },
    ],
]

TAIL_DATE_RELATION_CONTEXT_PATTERNS = [
    [
        {
            "LEFT_ID": "prep_end",
            "REL_OP": "<",
            "RIGHT_ID": "adj_to_aux",
            "RIGHT_ATTRS": {"DEP": "acomp", "POS": "ADJ"}
        },
        {
            "LEFT_ID": "adj_to_aux",
            "REL_OP": "<",
            "RIGHT_ID": "aux_verb",
            "RIGHT_ATTRS": {"LEMMA": "be"}
        },
    ],
    [
        {
            "LEFT_ID": "prep_end",
            "REL_OP": "<",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "aux_verb",
            "RIGHT_ATTRS": {"DEP": "aux"}
        },  
    ],
    [
        {
            "LEFT_ID": "prep_end",
            "REL_OP": "<",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "adj_to_verb",
            "RIGHT_ATTRS": {"DEP": {"IN": ["aux", "acomp"]}, "POS": "ADJ"}
        },  
    ],
    [
        {
            "LEFT_ID": "prep_end",
            "REL_OP": "<",
            "RIGHT_ID": "adj_to_verb",
            "RIGHT_ATTRS": {"DEP": {"IN": ["aux", "acomp"]}, "POS": "ADJ"}
        },
        {
            "LEFT_ID": "adj_to_verb",
            "REL_OP": "<",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },  
    ],
    []
]

SECU_DATE_RELATION_FROM_ROOT_VERB_CONTEXT_PATTERNS = [
    core + tail
    for core in CORE_DATE_RELATION_CONTEXT_PATTERNS
    for tail in TAIL_DATE_RELATION_CONTEXT_PATTERNS
]


SECU_DATE_RELATION_PATTERNS_FROM_ROOT_VERB = [
    [   
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">>",
            "RIGHT_ID": "prep_phrase_start",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER":  "as"}
        },
        {
            "LEFT_ID": "prep_phrase_start",
            "REL_OP": ">",
            "RIGHT_ID": "second_prep",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "of"}
        },
        {
            "LEFT_ID": "second_prep",
            "REL_OP": ">",
            "RIGHT_ID": "date_start",
            "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": "DATE"}
        }
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">>",
            "RIGHT_ID": "prep_phrase_start",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER":  "on"}
        },
        {
            "LEFT_ID": "prep_phrase_start",
            "REL_OP": ">",
            "RIGHT_ID": "date_start",
            "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": "DATE"}
        }
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">>",
            "RIGHT_ID": "prep_end",
            "RIGHT_ATTRS": {"DEP": "prep", "LOWER":  "until"}
        },
        {
            "LEFT_ID": "prep_end",
            "REL_OP": ">",
            "RIGHT_ID": "date_start",
            "RIGHT_ATTRS": {"DEP": "pobj", "ENT_TYPE": "DATE"}
        }
    ],
    
]

# FIXME: which patterns below are to general and cause the wrong assignment to a unassociated SECU during relationship building?
SECU_SECUQUANTITY_PATTERNS = [
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "pobj",
            "RIGHT_ATTRS": {"DEP": "pobj", "LOWER": "of"}
        },
        {
            "LEFT_ID": "pobj",
            "REL_OP": "<",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {"DEP": "prep", "POS": {"IN": ["NOUN", "PROPN"]}, "LOWER": "shares"}
        },
        {
            "LEFT_ID": "prep",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity",
            "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
        }
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {"DEP": "prep"}
        },
        {
            "LEFT_ID": "prep",
            "REL_OP": "<",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": "NOUN", "LOWER": {"IN": ["shares"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity",
            "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
        }

    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity",
            "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
        }

    ],
]

SECU_SOURCE_SECU_SECUQUANTITY_PATTERNS = [
    # anchor is the secu for which we want to check for child security relations
    # eg: warrant (<-anchor) to purchase 5000 shares of common stock (<-child security)
    # [
    #     {
    #         "LEFT_ID": "anchor",
    #         "REL_OP": "<<",
    #         "RIGHT_ID": "verb1",
    #         "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["relate"]}}, 
    #     },
    #     {
    #         "LEFT_ID": "verb1",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "prep",
    #         "RIGHT_ATTRS": {"DEP": "prep"}, 
    #     },
    #     {
    #         "LEFT_ID": "prep",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "noun_relation",
    #         "RIGHT_ATTRS": {"POS": "NOUN"}, 
    #     },
    #     {
    #         "LEFT_ID": "anchor",
    #         "REL_OP": "<<",
    #         "RIGHT_ID": "any",
    #         "RIGHT_ATTRS": {"POS": "NOUN"}
    #     },
    #     {
    #         "LEFT_ID": "any",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "secuquantity",
    #         "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
    #     }
    # ],
    # [
    #     {
    #         "LEFT_ID": "anchor",
    #         "REL_OP": "<<",
    #         "RIGHT_ID": "verb1",
    #         "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["relate"]}}, 
    #     },
    #     {
    #         "LEFT_ID": "verb1",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "prep",
    #         "RIGHT_ATTRS": {"DEP": "prep"}, 
    #     },
    #     {
    #         "LEFT_ID": "prep",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "noun_relation",
    #         "RIGHT_ATTRS": {"POS": "NOUN"}, 
    #     },
    #     {
    #         "LEFT_ID": "anchor",
    #         "REL_OP": "<<",
    #         "RIGHT_ID": "any",
    #         "RIGHT_ATTRS": {"POS": "NOUN"}
    #     },
    #     {
    #         "LEFT_ID": "any",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "secuquantity",
    #         "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
    #     },
    #     {
    #         "LEFT_ID": "verb1",
    #         "REL_OP": "<",
    #         "RIGHT_ID": "source_secu",
    #         "RIGHT_ATTRS": {"ENT_TYPE": "SECU"}, 
    #     }
    # ],
        [
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": "VERB"}, 
        },
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "v2",
            "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["have", "will", "be", "can"]}}, 
        },
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity",
            "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
        }
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": "VERB"}, 
        },
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "v2",
            "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["have", "will", "be", "can"]}}, 
        },
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity_unit",
            "RIGHT_ATTRS": {"LOWER": {"IN": SECUQUANTITY_UNITS}}
        },
        {
            "LEFT_ID": "secuquantity_unit",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity",
            "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
        }
    ],
    [ # eg: As of April 20, 2022, we had[v2] warrants[anchor] outstanding to issue[verb1] 20000[secuquantity] shares[secuquantity_unit] of common stock
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "v2",
            "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": ["have", "will", "be", "can"]}}, 
        },
        {
            "LEFT_ID": "v2",
            "REL_OP": ">",
            "RIGHT_ID": "verb1",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "xcomp", "LEMMA": {"NOT_IN": ["have", "will", "be", "can"]}}, 
        },
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity_unit",
            "RIGHT_ATTRS": {"LOWER": {"IN": SECUQUANTITY_UNITS}}
        },
        {
            "LEFT_ID": "secuquantity_unit",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity",
            "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
        }
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "verb1",
            # "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"NOT_IN": ["purchase", "acquire"]}},
            "RIGHT_ATTRS": {"POS": "VERB"}, 
        },
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity",
            "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
        }
    ],
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": "<",
            "RIGHT_ID": "verb1",
            # "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"NOT_IN": ["purchase", "acquire"]}},
            "RIGHT_ATTRS": {"POS": "VERB"}, 
        },
        {
            "LEFT_ID": "verb1",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity_unit",
            "RIGHT_ATTRS": {"LOWER": {"IN": SECUQUANTITY_UNITS}}
        },
        {
            "LEFT_ID": "secuquantity_unit",
            "REL_OP": ">",
            "RIGHT_ID": "secuquantity",
            "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
        }
    ],
    # [
    #     {
    #         "LEFT_ID": "anchor",
    #         "REL_OP": "<<",
    #         "RIGHT_ID": "verb1",
    #         "RIGHT_ATTRS": {"POS": "VERB"}
    #     },
    #     {
    #         "LEFT_ID": "anchor",
    #         "REL_OP": "<<",
    #         "RIGHT_ID": "any",
    #         "RIGHT_ATTRS": {}
    #     },
    #     {
    #         "LEFT_ID": "any",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "secuquantity",
    #         "RIGHT_ATTRS": {"ENT_TYPE": "SECUQUANTITY"}
    #     },
    #     {
    #         "LEFT_ID": "verb1",
    #         "REL_OP": "<",
    #         "RIGHT_ID": "source_secu",
    #         "RIGHT_ATTRS": {"ENT_TYPE": "SECU"}, 
    #     },

    # ],
]

#TODO: see how i can integrate this approach or are there better ones?
SECU_GET_EXPIRY_DATE_LEMMA_COMBINATIONS = [
    # should i store these as sets so i can faster compare;
    # what are the alternatives and how much time
    # complexity will i get from sets compared to lists 
    # (when length of list is below 100)?
    # at what part of this pipe do I incorperate negation ?
    
    {
        "verb": set(["remain"]),
        "prep": set(["until"]),
        "adj": set(["exercisable"]),
    },
    {
        "verb": set(["expire"]),
        "prep": set(["as of", "on"]),
        "aux_verb": set(["be", "can"]),
    }
]

SECU_GET_EXERCISE_DATE_LEMMA_COMBINATIONS = [
    # should i store these as sets so i can faster compare;
    # what are the alternatives and how much time
    # complexity will i get from sets compared to lists 
    # (when length of list is below 100)?
    # at what part of this pipe do I incorperate negation ?
    
    {
        "prep": set(["as", "of", "on"]),
        "adj": set(["exercisable"]),
    },
    {
        "verb": set(["exercise"]),
        "prep": set(["as", "of", "on"]),
        "aux_verb": set(["can", "be"]),
    }
]

AT_CONTEXT_UNIT_TAIL_PATTERNS = [
    [
        {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "prep_unit",
            "RIGHT_ATTRS": {"DEP": "prep"}
            # "RIGHT_ATTRS": {"LOWER": "per"}, 
        },
        {
            "LEFT_ID": "prep_unit",
            "REL_OP": ">",
            "RIGHT_ID": "noun_unit",
            "RIGHT_ATTRS": {"DEP": "pobj"}, # POS NOUN
            # "RIGHT_ATTRS": {"LOWER": "share"} 
        },
    ]
]

PARENT_VERB_AT_CONTEXT = [
    # anchor should be the parent/root verb of a secu/sentence
    [
       {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "at",
            "RIGHT_ATTRS": {"POS": "ADP", "LOWER": "at", "DEP": "prep"}, 
        },
        {
            "LEFT_ID": "at",
            "REL_OP": ">",
            "RIGHT_ID": "price",
            "RIGHT_ATTRS": {"LOWER": "price"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "det",
            "RIGHT_ATTRS": {"POS": "DET"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "of",
            "RIGHT_ATTRS": {"LOWER": "of"}, 
        },
        {
            "LEFT_ID": "of",
            "REL_OP": ">",
            "RIGHT_ID": "symbol",
            "RIGHT_ATTRS": {"DEP": "pobj", "POS": "SYM"}, 
        },
        {
            "LEFT_ID": "symbol",
            "REL_OP": ">",
            "RIGHT_ID": "price_per_unit",
            "RIGHT_ATTRS": {"POS": "NUM", "DEP": {"IN": ["nmod", "nummod"]}}, 
        }
    ],
    [ # On April 20, 2022, we issued 10000 shares of our common stock at a price of 2$ per share.
       {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "at",
            "RIGHT_ATTRS": {"POS": "ADP", "LOWER": "at", "DEP": "prep"}, 
        },
        {
            "LEFT_ID": "at",
            "REL_OP": ">",
            "RIGHT_ID": "price",
            "RIGHT_ATTRS": {"LOWER": "price"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "det",
            "RIGHT_ATTRS": {"POS": "DET"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": "NOUN", "DEP": "commpound"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "of",
            "RIGHT_ATTRS": {"LOWER": "of"}, 
        },
        {
            "LEFT_ID": "of",
            "REL_OP": ">",
            "RIGHT_ID": "symbol",
            "RIGHT_ATTRS": {"DEP": "pobj", "POS": "SYM"}, 
        },
        {
            "LEFT_ID": "symbol",
            "REL_OP": ">",
            "RIGHT_ID": "price_per_unit",
            "RIGHT_ATTRS": {"POS": "NUM", "DEP": {"IN": ["nmod", "nummod"]}}, 
        }
    ],
    [
       {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "at",
            "RIGHT_ATTRS": {"POS": "ADP", "LOWER": "at", "DEP": "prep"}, 
        },
        {
            "LEFT_ID": "at",
            "REL_OP": ">",
            "RIGHT_ID": "price",
            "RIGHT_ATTRS": {"LOWER": "price"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "det",
            "RIGHT_ATTRS": {"POS": "DET"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "of",
            "RIGHT_ATTRS": {"LOWER": "of"}, 
        },
        {
            "LEFT_ID": "of",
            "REL_OP": ">",
            "RIGHT_ID": "price_per_unit",
            "RIGHT_ATTRS": {"DEP": "pobj", "POS": "NUM"}, 
        },
        {
            "LEFT_ID": "price_per_unit",
            "REL_OP": ">",
            "RIGHT_ID": "symbol",
            "RIGHT_ATTRS": {"POS": "SYM", "DEP": {"IN": ["nmod", "nummod"]}}, 
        }
    ],
    [ # On April 20, 2022, we issued 10000 shares of our common stock at a price of 2$ per share.
       {
            "LEFT_ID": "anchor",
            "REL_OP": ">",
            "RIGHT_ID": "at",
            "RIGHT_ATTRS": {"POS": "ADP", "LOWER": "at", "DEP": "prep"}, 
        },
        {
            "LEFT_ID": "at",
            "REL_OP": ">",
            "RIGHT_ID": "price",
            "RIGHT_ATTRS": {"LOWER": "price"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "det",
            "RIGHT_ATTRS": {"POS": "DET"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": "NOUN", "DEP": "commpound"}, 
        },
        {
            "LEFT_ID": "price",
            "REL_OP": ">",
            "RIGHT_ID": "of",
            "RIGHT_ATTRS": {"LOWER": "of"}, 
        },
        {
            "LEFT_ID": "of",
            "REL_OP": ">",
            "RIGHT_ID": "price_per_unit",
            "RIGHT_ATTRS": {"DEP": "pobj", "POS": "NUM"}, 
        },
        {
            "LEFT_ID": "price_per_unit",
            "REL_OP": ">",
            "RIGHT_ID": "symbol",
            "RIGHT_ATTRS": {"POS": "SYM", "DEP": {"IN": ["nmod", "nummod"]}}, 
        }
    ]
]

        