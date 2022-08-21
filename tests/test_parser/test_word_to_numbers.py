from main.parser.filing_nlp import WordToNumberConverter, SpacyFilingTextSearch
from spacy.tokens import Token, Span
import pytest
from datetime import timedelta


w2n = WordToNumberConverter()
text_search = SpacyFilingTextSearch()

def test_number_conversion():
    token = text_search.nlp("one")[0]
    assert w2n.convert_spacy_token(token) == 1

def test_timedelta_conversion():
    token = [t for t in text_search.nlp("five weeks")][1]
    assert w2n.convert_spacy_token(token) == timedelta(weeks=1)
