from main.nlp.filing_nlp_certainty_setter import create_certainty_setter, CERTAINTY_LEVEL_MAP_ADV, CERTAINTY_LEVEL_MAP_MD
import spacy
from spacy.tokens import Span, Token, Doc


def test_certainty_setter():
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("certainty_setter")

    text = "We might issue securities after the date of issuance. We may be issuing new securities from time to time."

    doc = nlp(text)
    for marker_idx, scope_idxs in doc._.certainty_marker_map.items():
        scope = [doc[i] for i in scope_idxs]
        marker = doc[marker_idx]
        print(marker, scope)
    assert Doc.has_extension("certainty_marker_map") is True
    assert Token.has_extension("certainty_info") is True
    markers = list(doc._.certainty_marker_map.keys())
    assert doc[markers[0]].lower_ == "might"
    assert doc[markers[1]].lower_ == "may"

def test_determine_level_adv():
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("certainty_setter")

    text = "Perhaps, we won't issue a new security."
    doc = nlp(text)
    level = doc[5]._.certainty_info.determine_level()
    assert level == CERTAINTY_LEVEL_MAP_ADV["perhaps"]

def test_determine_level_md():
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("certainty_setter")

    text = "We may not issue new securities this year."
    doc = nlp(text)
    level = doc[3]._.certainty_info.determine_level()
    assert level == CERTAINTY_LEVEL_MAP_MD["may"]



    