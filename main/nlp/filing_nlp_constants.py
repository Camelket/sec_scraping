PLURAL_SINGULAR_SECU_TAIL_MAP = {
    "warrants": "warrant"
}
SINGULAR_PLURAL_SECU_TAIL_MAP = {
    "warrant": "warrants"
}

SECUQUANTITY_UNITS = set(["share", "shares", "units", "unit", "tranches"])

ATTRIBUTE_KEY_TO_STRINGKEY = {
        "POS": "pos_",
        "TAG": "tag_",
        "ORTH": "orth_",
        "LOWER": "lower_",
        "DEP": "dep_",
        "LEMMA": "lemma_",
        "ENT_TYPE": "ent_type_",
    }

DEPENDENCY_ATTRIBUTE_MATCHER_IS_OPTIONAL_FLAG = "was_optional"

COMMON_CONTRACT_SUBJECTS = set(
    [
        "investor",
        "investors",
        "accredited investor",
        "accredited investors",
        "placement agent",
        "placement agents",
    ]
)


    