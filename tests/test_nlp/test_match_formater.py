
from main.nlp.filing_nlp import SpacyFilingTextSearch
from main.nlp.filing_nlp_utils import MatchFormater
from spacy.tokens import Token, Span
import pytest
from datetime import timedelta
import datetime

text_search = SpacyFilingTextSearch()
f = MatchFormater()

@pytest.mark.parametrize(["input", "expected"], [
    ("10", "10"),
    ("10.0", "10.0"),
    ("250,000.25", "250000.25"),
    ("100,000", "100000"),
    ("100.00,000", None),
    ("100,000.000,000", None),
    ("100.000,000.000", None)
    ])
def test_parse_american_number(input, expected):
    assert f.parse_american_number(input) == expected

@pytest.mark.parametrize(["input", "expected"], [
    ("up to $75,000,000", 75000000),
    ("up to $7.5 million", 7500000),
    ("up to $7,5 million", 7500000),
    ("up to $0.0075 billion", 7500000),
    ("11.5", 11.5),
    ("700,000", 700000),
    ("150,000.25", 150000.25),
    ("150,000,000.01", 150000000.01),
    ])
def test_money_string_to_float(input, expected):
    assert f.quantity_string_to_float(input) == expected

@pytest.mark.parametrize(["input", "expected"], [
    ("no", 0),
    # ("one thousand and one", 1001),
    # ("two million five thousand and seven", 2005007),
    # ("two hundred million", 200000000),
    # ("twenty thousand", 20000)
])
def test_money_words_to_float(input, expected):
    assert f.quantity_string_to_float(input) == expected

def test_timedelta_conversion_with_numeric_tokens():
    tokens = [t for t in text_search.nlp("3 weeks")]
    assert f.coerce_tokens_to_timedelta(tokens)[0][0] == timedelta(weeks=3)

def test_timedelta_conversion_with_numbers_as_words_tokens():
    tokens = [t for t in text_search.nlp("three weeks")]
    assert f.coerce_tokens_to_timedelta(tokens)[0][0] == timedelta(weeks=3)

def test_timedelta_conversion_with_garbage_tokens():
    tokens = [t for t in text_search.nlp("starting on the 3 weeks anniversary after the date which is the issuance date")]
    assert f.coerce_tokens_to_timedelta(tokens)[0][0] == timedelta(weeks=3)

def test_timedelta_conversion_with_multiple_timedeltas():
    tokens = [t for t in text_search.nlp("after 3 weeks and after five weeks")]
    result = f.coerce_tokens_to_timedelta(tokens)
    assert result[0][0] == timedelta(weeks=3)
    assert result[1][0] == timedelta(weeks=5)

@pytest.mark.parametrize(["input", "expected"],
    [
        (
            "April 14th 2020", datetime.datetime(2020, 4, 14)
        ),
        (
            "April 14, 2020", datetime.datetime(2020, 4, 14)
        ),
        (
            "14th April 2020", datetime.datetime(2020, 4, 14)
        ),
        (
            "14 of April 2020", datetime.datetime(2020, 4, 14)
        )
    ]
)
def test_date_conversion_from_tokens(input, expected):
    tokens = [t for t in text_search.nlp(input)]
    assert f.coerce_tokens_to_datetime(tokens) == expected
