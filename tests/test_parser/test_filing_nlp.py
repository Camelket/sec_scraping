import pytest
import spacy
# from spacy.tokens import Span, Token, Doc
# from spacy.matcher import Matcher
from main.parser.filing_nlp import SpacyFilingTextSearch, create_secuquantity_matcher, create_secu_matcher
from main.parser.filing_nlp_dependency_matcher import DependencyAttributeMatcher
from pandas import to_datetime
import datetime

@pytest.fixture
def get_en_core_web_lg_nlp():
    nlp = spacy.load("en_core_web_lg")
    yield nlp
    del nlp

@pytest.fixture
def get_search():
    search = SpacyFilingTextSearch()
    yield search
    del search

@pytest.fixture
def get_dependency_attribute_matcher():
    dep_matcher = DependencyAttributeMatcher()
    yield dep_matcher
    del dep_matcher




# classes not tested: AggreementMatcher, NegationSetter

# TODO: move to own test file
class TestDependencyAttributeMatcher:
    def test_match_simple_pattern_length_two(self, get_search, get_dependency_attribute_matcher):
        search = get_search
        text = "We sold 10000 shares of our common stock at a purchase price of 2$ per share."
        doc = search.nlp(text)
        pattern = [
            {"RIGHT_ID": "one", "TOKEN": doc[0]},
            {"LEFT_ID": "one", "REL_OP": "<", "RIGHT_ID": "two", "RIGHT_ATTRS": {"POS": "VERB"}}
        ]
        dep_getter = get_dependency_attribute_matcher
        matches = dep_getter.get_candidate_matches(pattern)
        assert len(matches) == 1
        expected_tokens = [doc[0], doc[1]]
        found_tokens = [i[0] for i in matches[0]]
        assert expected_tokens == found_tokens

    def test_match_simple_pattern_one_child_per_level(self, get_search):
        search = get_search
        text = "This is a test sentence for detecting an exercise price."
        doc = search.nlp(text)
        pattern = [
            {"RIGHT_ID": "one", "TOKEN": doc[0]},
            {"LEFT_ID": "one", "REL_OP": "<", "RIGHT_ID": "two", "RIGHT_ATTRS": {"LOWER": "is"}},
            {"LEFT_ID": "two", "REL_OP": ">", "RIGHT_ID": "three", "RIGHT_ATTRS": {"LOWER": "sentence"}},
            {"LEFT_ID": "three", "REL_OP": ">", "RIGHT_ID": "four", "RIGHT_ATTRS": {"LOWER": "test"}},
        ]
        matches = search.dep_getter.get_candidate_matches(pattern)
        assert len(matches) == 1
        expected_tokens = [doc[0], doc[1], doc[4], doc[3]]
        found_tokens = [i[0] for i in matches[0]]
        assert found_tokens == expected_tokens

    def test_match_simple_pattern_two_children(self, get_search):
        search = get_search
        text = "This is a test sentence for detecting an exercise price."
        doc = search.nlp(text)
        pattern = [
            {"RIGHT_ID": "one", "TOKEN": doc[0]},
            {"LEFT_ID": "one", "REL_OP": "<", "RIGHT_ID": "two", "RIGHT_ATTRS": {"LOWER": "is"}},
            {"LEFT_ID": "two", "REL_OP": ">", "RIGHT_ID": "three", "RIGHT_ATTRS": {"LOWER": "sentence"}},
            {"LEFT_ID": "three", "REL_OP": ">", "RIGHT_ID": "four", "RIGHT_ATTRS": {"LOWER": "test"}},
            {"LEFT_ID": "three", "REL_OP": ">", "RIGHT_ID": "five", "RIGHT_ATTRS": {"LOWER": "for"}},
        ]
        matches = search.dep_getter.get_candidate_matches(pattern)
        assert len(matches) == 1
        expected_tokens = [doc[0], doc[1], doc[4], doc[3], doc[5]]
        found_tokens = [i[0] for i in matches[0]]
        assert found_tokens == expected_tokens
    
    # TODO: test above two cases for multiple matches and using the IS_OPTIONAL attribute

    def test__get_tokens_with_tag_from_match_tuples(self, get_dependency_attribute_matcher):
        dep_getter = get_dependency_attribute_matcher
        expected = ["this", "is", None]
        match = [("this", "tag1"), ("is", "tag1")]
        for idx, found in enumerate(dep_getter._get_tokens_with_tag_from_match_tuples(match, "tag1")):
            assert found == expected[idx]
            if found is None:
                break
    @pytest.mark.parametrize(["input", "expected", "origin_idx"], [
        (
            "The warrants are exercisable at $0.50 per share.",
            "are",
            1
        ),
    ])
    def test_get_root_verb(self, input, expected, origin_idx, get_search):
        search = get_search
        doc = search.nlp(input)
        match = search.dep_getter.get_root_verb(doc[origin_idx])
        assert match.text == expected
    
    @pytest.mark.parametrize(["input", "expected", "origin_idx"], [
        (
            "We are to purchase the warrants at $0.50 per unit.",
            "purchase",
            5
        ),
    ])
    def test_get_parent_verb(self, input, expected, origin_idx, get_search):
        search = get_search
        doc = search.nlp(input)
        match = search.dep_getter.get_parent_verb(doc[origin_idx])
        assert match.text == expected

    def test_get_quantities(self):
        pass
    
    @pytest.mark.parametrize(["input", "expected", "origin_idx"], [
        (
            "",
            "",
            0
        ),
    ])
    def test_get_expiry(self, input, expected, origin_idx, get_search):
        search = get_search
        doc = search.nlp(input)
        match = search.dep_getter.get_expiry(doc[origin_idx])
        assert match == expected
    
    #FIXME: adjust for new context format when decided on which to use
    @pytest.mark.parametrize(["input", "expected", "origin_idx"], [
        (
            "The warrants we issued pursuant to our Private Placement are exercisable as of May 5, 2021.",
            {"datetime": datetime.date(2021, 5, 5), "timedelta": []},
            1
        ),
    ])
    def test_get_date_relation(self, input, expected, origin_idx, get_search):
        search = get_search
        doc = search.nlp(input)
        match = search.dep_getter.get_date_relation(doc[origin_idx])
        assert match == expected
    
    @pytest.mark.parametrize(["input", "expected", "secu_idx"], [
        (
            "The Warrants have an exercise price of 10.50 $ per share.",
            (10.50, "$"),
            1
        ),
        (
            "The Warrants have an exercise price of $ 10.50 per share.",
            (10.50, "$"),
            1
        ),
        (
            "The Warrants have an exercise price of $10.50 per share.",
            (10.50, "$"),
            1
        ),
        (
            "The Warrants have an exercise price of 10.50$ per share.",
            (10.50, "$"),
            1
        ),
        (
            "The common shares are issuable upon exercise of the Warrants at an exercise price of 10.50 $ per share.",
            (10.50, "$"),
            9
        ),
        (
            "The common shares are issuable upon exercise of the Warrants at an exercise price of 0.50 $ to 10.50 $ per share.",
            None,
            9
        )
    ])
    def test_get_exercise_price(self, input, expected, secu_idx, get_search):
        search = get_search
        doc = search.nlp(input)
        match = search.dep_getter.get_exercise_price(doc[secu_idx])
        print(match)
        assert match == expected
        
        

class TestSECUMatcher:
    '''
    Tests currently missing:
        * setting of possible alias spans
        * tokens to alias map
        * retokenization of SECU spans into tokens
        * setting SECUREF
        * setting SECUATTR
    '''

    def test_matcher_SECU(self, get_en_core_web_lg_nlp):
        nlp = get_en_core_web_lg_nlp
        nlp.add_pipe("secu_matcher")
        doc = nlp("The Common stock, Series A Preferred stock and the Series C Warrants of company xyz is fairly valued.")
        secus = [i.text for i in filter(lambda x: x.ent_type_ == "SECU", doc)]
        assert secus == ['Common', 'stock', 'Series', 'A', 'Preferred', 'stock', 'Series', 'C', 'Warrants']

class TestSECUQuantityMatcher:

    @pytest.mark.parametrize(["input", "expected"], [
        (
            "This prospectus covers the sale of an aggregate of 2,388,050 shares (the “shares”) of our common stock , $0.001 par value per share (the “ common stock ”), by the selling stockholders identified in this prospectus (collectively with any of the holder’s transferees, pledgees, donees or successors, the “selling stockholders”). The shares are issuable upon the exercise of warrants (the “ warrants ”) purchased by the selling stockholders in a private placement transaction exempt from registration under Section 4(a)(2) of the Securities Act of 1933, as amended (the “Securities Act”), pursuant to a Securities Purchase Agreement dated April 9, 2021 (the “Purchase Agreement”).",
            ["2,388,050"]
        ),
        (
            "This prospectus relates to an aggregate of up to 9,497,051 shares of common stock , par value $0.01 per share, of Basic Energy Services, Inc. (“Basic”) that may be resold from time to time by the selling stockholders named on page 5 of this prospectus for their own account.",
            ["up", "to", "9,497,051"]
        )
    ])
    def test_matcher_SECUQUANTITY(self, input, expected, get_en_core_web_lg_nlp):
        nlp = get_en_core_web_lg_nlp
        nlp.add_pipe("secu_matcher")
        nlp.add_pipe("secuquantity_matcher")
        doc = nlp(input)
        secuquantities = [i.text for i in filter(lambda x: x.ent_type_ == "SECUQUANTITY", doc)]
        assert expected == secuquantities

class TestSpacyFilingTextSearch:
    '''
    Tests currently missing:
        * getting correct SECU objects with get_SECU_objects
        *
    '''
    def test__create_span_dependency_matcher_dict_lower(self, get_search):
        search:  SpacyFilingTextSearch = get_search
        text = "The Series A Warrants have an exercise price of $11.50 per share."
        doc = search.nlp(text)
        secu = doc[1:2]
        dep_dict = search._create_span_dependency_matcher_dict_lower(secu)
        print(dep_dict)
        assert dep_dict == [
            {
                'RIGHT_ID': 'anchor',
                'RIGHT_ATTRS': {'ENT_TYPE': 'SECU', 'LOWER': 'series a warrants'}
            }
            ]

    def test_get_queryable_similar_spans_from_lower(self, get_search):
        search: SpacyFilingTextSearch = get_search
        text = "The Series A Warrants have an exercise price of $11.50 per share. The Series A Warrant has been issued in connection witha private placement."
        doc = search.nlp(text)
        secu = doc[1:2]
        queryable_spans = search.get_queryable_similar_spans_from_lower(doc, secu)
        assert queryable_spans == [doc[13:14]]



    @pytest.mark.parametrize(["text", "expected", "secu_idx"], [
        (
            "Warrants to purchase 33,334 shares of common stock at any time on or prior to September 26, 2022 at an initial exercise price of $3.00 per share.",
            3,
            (0, 1)
        ),
        (
            "The Series A Warrants have an exercise price of $11.50 per share",
            11.5,
            (1, 2)
        ),
        (
            "Warrants to purchase 96,668 shares of common stock and remain outstanding at any time on or prior to December 31, 2022 at an initial exercise price of $3.00 per share.",
            3,
            (0, 1)
        ),
        ])
    def test_match_exercise_price(self, text, expected, secu_idx, get_search):
        search: SpacyFilingTextSearch = get_search
        doc = search.nlp(text)
        secu = doc[secu_idx[0]:secu_idx[1]]
        matches = search.match_secu_exercise_price(doc, secu)
        print("matches: ", matches)
        print("expected: ", expected)
        assert expected == matches[0]



    @pytest.mark.parametrize(["text", "expected", "secu_idx"], [
            (
                "The Warrants expires on August 6, 2025.",
                datetime.datetime(2025, 8, 6),
                (1, 2),

            ),
            (
                "The Warrants are exercisable at an exercise price of $2.00 per share and expire on the fourth year anniversary of December 14, 2021, the initial issuance date of the Warrants",
                datetime.datetime(2021, 12, 14) + 4*datetime.timedelta(365.25),
                (1, 2),

            ),
            (
                "The Warrants have an exercise price of $11.50 per share will be exercisable beginning on the calendar day following the six month anniversary of the date of issuance, will expire on March 17, 2026.",
                datetime.datetime(2026, 3, 17),
                (1, 2),

            ),
        ])
    def test_match_expiry(self, text, expected, secu_idx, get_search):
        search: SpacyFilingTextSearch = get_search
        doc = search.nlp(text)
        secu = doc[secu_idx[0]:secu_idx[1]]
        matches = search.match_secu_expiry(doc, secu)
        print(matches)
        assert expected == matches[0]


    def test_secu_alias_map(self, get_search):
        search = get_search
        text = "On February 22, 2021, we entered into the Securities Purchase Agreement (the “Securities Purchase Agreement”), pursuant to which we agreed to issue the investor named therein (the “Investor”) 8,888,890 shares (the “Shares”) of our common stock, par value $0.000001 per share, at a purchase price of $2.25 per share, and a warrant to purchase up to 6,666,668 shares of our common stock (the “Investor Warrant”) in a private placement (the “Private Placement”). The closing of the Private Placement occurred on February 24, 2021."
        # text = "On February 22, 2021, we entered into the Securities Purchase Agreement (the “Securities Purchase Agreement”), pursuant to which we agreed to issue the investor named therein (the “Investor”) 8,888,890 shares (the “Shares”) of our common stock, par value $0.000001 per share, at a purchase price of $2.25 per share, and a warrant (the “Investor Warrant”) to purchase up to 6,666,668 shares of our common stock in a private placement (the “Private Placement”). The closing of the Private Placement occurred on February 24, 2021."
        doc = search.nlp(text)
        # print(doc._.single_secu_alias_tuples)
        # print(doc._.single_secu_alias)
        expected_bases = ["common stock", "common stock", "warrant"]
        expected_alias = ["Investor Warrant"]
        alias_map = doc._.single_secu_alias
        received_bases = sum([alias_map[k]["base"] for k, v in alias_map.items()], [])
        received_alias = sum([alias_map[k]["alias"] for k, v in alias_map.items()], [])
        print(f"doc._.single_secu_alias: {alias_map}")
        assert len(expected_bases) == len(received_bases)
        assert len(expected_alias) == len(received_alias)
        for expected, received in zip(expected_bases, received_bases):
            assert expected == received.text
        for expected, received in zip(expected_alias, received_alias):
            assert expected == received.text

    @pytest.mark.parametrize(["phrase", "expected"],
        [
            (
                "As of May 4, 2021, 46,522,759 shares of our common stock were issued and outstanding.",
                {"date": to_datetime("May 4, 2021"),"amount": 46522759}),
            (
                "The number of shares and percent of class stated above are calculated based upon 399,794,291 total shares outstanding as of May 16, 2022",
                {"date": to_datetime("May 16, 2022"),"amount": 399794291}),
            (
                "based on 34,190,415 total outstanding shares of common stock of the Company as of January 17, 2020. ",
                {"date": to_datetime("January 17, 2020"),"amount": 34190415}),
            (
                "are based on 30,823,573 shares outstanding on April 11, 2022. ",
                {"date": to_datetime("April 11, 2022"),"amount": 30823573}),
            (
                "based on 70,067,147 shares of our Common Stockoutstanding as of October 18, 2021.",
                {"date": to_datetime("October 18, 2021"),"amount": 70067147}),
            (
                "based on 41,959,545 shares of our Common Stock outstanding as of October 26, 2020. ",
                {"date": to_datetime("October 26, 2020"),"amount": 41959545}) 
        ]
    )
    def test_match_outstanding_shares(self, get_search, phrase, expected):
        search: SpacyFilingTextSearch = get_search
        res = search.match_outstanding_shares(phrase)
        print(f"sent: {phrase}")
        print(f"res: {res}")
        assert res[0] == expected

    def test_get_conflicting_ents(self, get_search):
        search = get_search
        text = "one or three or five are the magic numbers."
        doc = search.nlp(text)
        from main.parser.filing_nlp import get_conflicting_ents
        print("ents in test doc: ", [ent for ent in doc.ents])
        conflicting = get_conflicting_ents(doc, 0, 1)
        assert conflicting == [doc.ents[0]]
        conflicting = get_conflicting_ents(doc, 0, 5)
        assert conflicting == [doc.ents[0], doc.ents[1]]
        conflicting = get_conflicting_ents(doc, 1, 2)
        assert conflicting == []
