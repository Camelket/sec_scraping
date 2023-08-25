from main.nlp.filing_nlp import SpacyFilingTextSearch
from main.nlp.filing_nlp_utils import WordToNumberConverter
from datetime import timedelta


w2n = WordToNumberConverter()
text_search = SpacyFilingTextSearch()

def test_spacy_token_to_number_conversion():
    token = text_search.nlp("one")[0]
    assert w2n.convert_spacy_token(token) == 1

def test_spacy_token_to_timedelta_conversion():
    tokens = [t for t in text_search.nlp("five weeks")]
    p1 = w2n.convert_spacy_token(tokens[0])
    p2 = w2n.convert_spacy_token(tokens[1])
    assert p2 == timedelta(weeks=1)
    assert p1 == 5
    assert p1*p2 == timedelta(weeks=5)

# TODO: add tests for words to number 

