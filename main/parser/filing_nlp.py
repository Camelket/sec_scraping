from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set
import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Span, Doc, Token
from spacy import Language
import logging
import string
import re
import pandas as pd
from pandas import Timestamp

from main.parser.filing_nlp_utils import (
    MatchFormater,
    get_dep_distance_between_spans,
    get_span_distance,
    extend_token_ent_to_span,
)
from main.parser.filing_nlp_certainty_setter import create_certainty_setter
from main.parser.filing_nlp_negation_setter import create_negation_setter
from main.parser.filing_nlp_dependency_matcher import (
    SecurityDependencyAttributeMatcher,
)
from main.parser.filing_nlp_patterns import (
    add_anchor_pattern_to_patterns,
    SECU_EXERCISE_PRICE_PATTERNS,
    SECU_EXPIRY_PATTERNS,
    SECU_ENT_REGULAR_PATTERNS,
    SECU_ENT_DEPOSITARY_PATTERNS,
    SECU_ENT_SPECIAL_PATTERNS,
    SECUQUANTITY_ENT_PATTERNS,

)
from main.parser.filing_nlp_constants import (
    PLURAL_SINGULAR_SECU_TAIL_MAP,
    SINGULAR_PLURAL_SECU_TAIL_MAP,
    SECUQUANTITY_UNITS,
)
from main.parser.filing_nlp_SECU import SECU, SECUQuantity, UnitAmount, QuantityRelation, SourceQuantityRelation

logger = logging.getLogger(__name__)
formater = MatchFormater()


class UnclearInformationExtraction(Exception):
    pass


class CommonFinancialRetokenizer:
    '''
    matches and retokenizes common used words in financial texts which otherwise get wrong tags.
    for now only affects "par value"
    
    SideEffect:
        indices shift, so place component at the beginning of the pipeline.
    '''
    def __init__(self, vocab):
        pass

    def __call__(self, doc):
        expressions = [
            re.compile(r"par\svalue", re.I),
        ]
        match_spans = []
        for expression in expressions:
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if match is not None:
                    match_spans.append([span, start, end])
        with doc.retokenize() as retokenizer:
            for span in match_spans:
                span = span[0]
                if span is not None:
                    retokenizer.merge(span, attrs={"POS": "NOUN", "TAG": "NN"})
        return doc


class FilingsSecurityLawRetokenizer:
    def __init__(self, vocab):
        pass

    def __call__(self, doc):
        expressions = [
            # eg: 415(a)(4)
            re.compile(
                r"(\d\d?\d?\d?(?:\((?:(?:[a-zA-Z0-9])|(?:(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})))\)){1,})",
                re.I,
            ),
            re.compile(
                r"\s\((?:(?:[a-zA-Z0-9])|(?:(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})))\)\s",
                re.I,
            ),
            re.compile(r"(\s[a-z0-9]{1,2}\))|(^[a-z0-9]{1,2}\))", re.I | re.MULTILINE),
        ]
        match_spans = []
        for expression in expressions:
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if match is not None:
                    match_spans.append([span, start, end])
        # sorted_match_spans = sorted(match_spans, key=lambda x: x[1])
        longest_matches = filter_matches(match_spans)
        with doc.retokenize() as retokenizer:
            for span in longest_matches:
                span = span[0]
                if span is not None:
                    retokenizer.merge(span)
        return doc


class SecurityActMatcher:
    '''
    This component matches the usual pattern of security acts present in SEC filings
    and retokenizes the tokens into one, for easier matching in other components.

    Sideeffect:
        shift of indices, so place this component at the beginning of the pipeline.
    '''
    def __init__(self, vocab):
        if not Token.has_extension("sec_act"):
            Token.set_extension("sec_act", default=False)
        self.matcher = Matcher(vocab)
        self.add_sec_acts_to_matcher()

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
                for token in span:
                    token._.sec_act = True
        return doc

    def add_sec_acts_to_matcher(self):
        patterns = [
            [
                {"ORTH": {"IN": ["Rule", "Section", "section"]}},
                {
                    "LOWER": {
                        "REGEX": r"(\d\d?\d?\d?(?:\((?:(?:[a-z0-9])|(?:(?=[mdclxvi])m*(c[md]|d?C{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})))\)){0,})"
                    },
                    "OP": "*",
                },
            ],
            [
                {"ORTH": {"IN": ["Section" + x for x in list(string.ascii_uppercase)]}},
            ],
        ]
        self.matcher.add("sec_act", patterns, greedy="LONGEST")


def get_span_secuquantity_float(span: Span):
    if not isinstance(span, Span):
        raise TypeError(
            "span must be of type spacy.tokens.span.Span, got: {}".format(type(span))
        )
    if span.label_ == "SECUQUANTITY":
        return formater.quantity_string_to_float(span.text)
    else:
        raise AttributeError(
            "get_secuquantity can only be called on Spans with label: 'SECUQUANTITY'"
        )


def get_token_secuquantity_float(token: Token):
    if not isinstance(token, Token):
        raise TypeError(
            "token must be of type spacy.tokens.token.Token, got: {}".format(
                type(token)
            )
        )
    if token.ent_type_ == "SECUQUANTITY":
        return formater.quantity_string_to_float(token.text)
    else:
        raise AttributeError(
            "get_secuquantity can only be called on Spans with label: 'SECUQUANTITY'"
        )


def retokenize_SECU(doc: Doc):
    # this needs to decouple the source/premerge tokens so
    # we dont have problem with a shift of token positions
    # after the merge
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            if ent.label_ == "SECU":
                source_doc_slice = ent.as_doc(copy_user_data=True)
                # handle previously merged tokens to retain
                # original/first source_span_unmerged
                if not ent.has_extension("was_merged"):
                    source_tokens = tuple([t for t in source_doc_slice])
                else:
                    if ent._.was_merged:
                        source_tokens = ent._.premerge_tokens
                    else:
                        source_tokens = tuple([t for t in source_doc_slice])
                attrs = {
                    # "tag": ent.root.tag,
                    "tag": "NOUN",
                    "pos": "NOUN",
                    "dep": ent.root.dep,
                    "ent_type": ent.label,
                    # might fix some wrong dependency setting (SECU should be a NOUN in any case, correct?)
                    "_": {"source_span_unmerged": source_tokens, "was_merged": True},
                }
                ent._.was_merged = True
                retokenizer.merge(ent, attrs=attrs)
    return doc


def get_span_to_span_similarity_map(
    secu: list[Token] | Span, alias: list[Token] | Span, threshold: float = 0.65
):
    similarity_map = {}
    for secu_token in secu:
        for alias_token in alias:
            similarity = secu_token.similarity(alias_token)
            similarity_map[(secu_token, alias_token)] = similarity
    return similarity_map
    # if  similarity > 0.65:
    #     similarity_map_entry["very_similar"].append((secu_token, alias_token, secu_token.similarity(alias_token)))
    #     very_similar_count += 1

    # compare token to token2 and check if we have a token that is very similar >.65
    # count the very similar tokens
    # take the alias with the highest count ?
    # bootstrap a statistical model to predict the alias?

def calculate_similarity_score(
    alias: list[Token] | Span,
    similarity_map,
    dep_distance: int,
    span_distance: int,
    very_similar_threshold: float,
    dep_distance_weight: float,
    span_distance_weight: float,
) -> float:
    very_similar = sum([v > very_similar_threshold for v in similarity_map.values()])
    very_similar_score = very_similar / len(alias) if very_similar != 0 else 0
    dep_distance_score = dep_distance_weight * (1 / dep_distance)
    span_distance_score = span_distance_weight * (10 / span_distance)
    total_score = dep_distance_score + span_distance_score + very_similar_score
    return total_score


def get_span_similarity_score(
    span1: list[Token] | Span,
    span2: list[Token] | Span,
    dep_distance_weight: float = 0.7,
    span_distance_weight: float = 0.3,
    very_similar_threshold: float = 0.65,
) -> float:
    premerge_tokens = (
        span1._.premerge_tokens if span1.has_extension("premerge_tokens") else span1
    )
    similarity_map = get_span_to_span_similarity_map(premerge_tokens, span2)
    dep_distance = get_dep_distance_between_spans(span1, span2)
    span_distance = get_span_distance(span1, span2)
    if dep_distance and span_distance and similarity_map:
        score = calculate_similarity_score(
            span2,
            similarity_map,
            dep_distance,
            span_distance,
            very_similar_threshold=very_similar_threshold,
            dep_distance_weight=dep_distance_weight,
            span_distance_weight=span_distance_weight,
        )
        return score


def get_alias(doc: Doc, secu: Span):
    # logger.debug(f"getting alias for: {secu}")
    if doc._.is_alias(secu) is True:
        return None
    else:
        secu_first_token = secu[0]
        secu_last_token = secu[-1]
        similarity_score_store = []
        checked_combination = set()
        # TODO: optimize this, but how?
        for sent in doc.sents:
            if secu_first_token in sent:
                secu_counter = 0
                token_idx_offset = sent[0].i
                for token in sent[secu_last_token.i - token_idx_offset + 1 :]:
                    alias = doc._.tokens_to_alias_map.get(token.i)
                    if token.ent_type_ == "SECU":
                        secu_counter += 1
                        if alias is None and secu_counter > 2:
                            return None
                    if alias:
                        if (secu, alias) in checked_combination:
                            continue
                        score = get_span_similarity_score(secu, alias)
                        checked_combination.add((secu, alias))
                        if score:
                            similarity_score_store.append(
                                (secu, alias, sent[0].i, score)
                            )
        # logger.debug(f"similarity_score_map: {similarity_score_store}")
        if similarity_score_store != []:
            highest_similarity = sorted(similarity_score_store, key=lambda x: x[3])[-1]
            return highest_similarity[1]
        else:
            return None


def _get_SECU_in_doc(doc: Doc) -> list[Span]:
    secu_spans = []
    for ent in doc.ents:
        if ent.label_ == "SECU":
            secu_spans.append(ent)
    return secu_spans


def _set_single_secu_alias_map(doc: Doc) -> dict:
    if (doc.spans.get("SECU") is None) or (doc.spans.get("alias") is None):
        raise AttributeError(
            f"Didnt set spans correctly missing one or more keys of (SECU, alias). keys found: {doc.spans.keys()}"
        )
    single_secu_alias = dict()
    secu_ents = doc._.secus
    # logger.debug(f"working from ._.secus: {secu_ents}")
    for secu in secu_ents:
        if doc._.is_alias(secu):
            continue
        secu_key = get_secu_key(secu)
        # logger.debug(f"got secu_key: {secu_key} -> from secu -> {secu}")
        if secu_key not in single_secu_alias.keys():
            single_secu_alias[secu_key] = {"base": [secu], "alias": []}
        else:
            single_secu_alias[secu_key]["base"].append(secu)
        alias = doc._.get_alias(secu)
        if alias:
            single_secu_alias[secu_key]["alias"].append(alias)
    # logger.debug(f"got single_secu_alias map: {single_secu_alias}")
    doc._.single_secu_alias = single_secu_alias


def _set_single_secu_alias_map_as_tuples(doc: Doc) -> dict:
    if (doc.spans.get("SECU") is None) or (doc.spans.get("alias") is None):
        # print(doc.spans.get("SECU"), doc.spans.get("alias"))
        raise AttributeError(
            f"Didnt set spans correctly missing one or more keys of (SECU, alias). keys found: {doc.spans.keys()}"
        )
    single_secu_alias_tuples = dict()
    secu_ents = doc._.secus
    for secu in secu_ents:
        if doc._.is_alias(secu):
            continue
        secu_key = get_secu_key(secu)
        if secu_key not in single_secu_alias_tuples.keys():
            single_secu_alias_tuples[secu_key] = {"alias": [], "no_alias": []}
        alias = doc._.get_alias(secu)
        if alias:
            single_secu_alias_tuples[secu_key]["alias"].append((secu, alias))
        else:
            single_secu_alias_tuples[secu_key]["no_alias"].append(secu)
    logger.info(f"single_secu_alias_tuples: {single_secu_alias_tuples}")
    doc._.single_secu_alias_tuples = single_secu_alias_tuples


def is_alias(doc: Doc, secu: Span) -> bool:
    if secu.text in doc._.alias_list:
        return True
    return False


def get_secu_key(secu: Span | Token) -> str:
    premerge_tokens = (
        secu._.premerge_tokens if secu.has_extension("premerge_tokens") else None
    )
    # logger.debug(f"premerge_tokens while getting secu_key: {premerge_tokens}")
    if premerge_tokens:
        secu = premerge_tokens
    core_tokens = secu if secu[-1].text.lower() not in ["shares"] else secu[:-1]
    body = [token.text_with_ws.lower() for token in core_tokens[:-1]]
    try:
        current_tail = core_tokens[-1].lemma_.lower()
        if current_tail in PLURAL_SINGULAR_SECU_TAIL_MAP.keys():
            tail = PLURAL_SINGULAR_SECU_TAIL_MAP[current_tail]
        else:
            tail = current_tail
        body.append(tail)
    except IndexError:
        logger.debug(
            "IndexError when accessing tail information of secu_key -> no tail -> pass"
        )
        pass
    result = "".join(body)
    # logger.debug(f"get_secu_key() returning key: {result} from secu: {secu}")
    return result


def get_secu_key_extension(target: Span | Token) -> str:
    if isinstance(target, Span):
        return _get_secu_key_extension_for_span(target)
    elif isinstance(target, Token):
        return _get_secu_key_extension_for_token(target)
    else:
        raise TypeError(f"target must be of type Span or Token, got {type(target)}")


def _get_secu_key_extension_for_span(span: Span):
    if span.label_ != "SECU":
        raise AttributeError(
            f"Can only get secu_key for SECU spans, span is not a SECU span. received span.label_: {span.label_}"
        )
    return get_secu_key(span)


def _get_secu_key_extension_for_token(token: Token):
    if token.ent_type_ != "SECU":
        raise AttributeError(
            f"Can only get secu_key for SECU tokens, token is not a SECU token. received token.ent_type_: {token.ent_type_}"
        )
    return get_secu_key(token)


def get_premerge_tokens_for_span(span: Span) -> tuple | None:
    premerge_tokens = []
    source_spans_seen = set()
    # logger.debug(f"getting premerge_tokens for span: {span}")
    for token in span:
        # logger.debug(f"checking token: {token}")
        if token.has_extension("was_merged"):
            if token._.was_merged is True:
                # logger.debug(f"token._.source_span_unmerged: {token._.source_span_unmerged}")
                if token._.source_span_unmerged not in source_spans_seen:
                    premerge_tokens.append([i for i in token._.source_span_unmerged])
                    source_spans_seen.add(token._.source_span_unmerged)
    # flatten
    premerge_tokens = sum(premerge_tokens, [])
    if premerge_tokens != []:
        return tuple(premerge_tokens)
    else:
        return None


def get_premerge_tokens_for_token(token: Token) -> tuple | None:
    if not isinstance(token, Token):
        raise TypeError(
            f"get_premerge_tokens_for_token() expects a Token object, received: {type(token)}"
        )
    if token.has_extension("was_merged"):
        if token._.was_merged is True:
            return tuple([i for i in token._.source_span_unmerged])
    return None

def _set_extension(cls, name, kwargs):
    if not cls.has_extension(name):
        cls.set_extension(name, **kwargs)


class SECUQuantityMatcher:
    '''
    This component will mark quantities associated with SECU entities.
    Custom extension attributes added with this component:
        Token extensions:
            - ._.secuquantity (easy access to underlying value as a float)
            - ._.secuquantity_unit (type of unit associated with the quantity, either MONEY or COUNT)
        Span extensions:
            - ._.secuquantity (easy access to underlying value as a float)
            - ._.secuquantity_unit (type of unit associated with the quantity, either MONEY or COUNT)
    
    This component needs to be placed after the SECUMatcher component or
    a custom component which adds SECU entities to the doc and sets the needed
    Span and Token extensions.
    See the SECUMatcher for specifications for a SECU entity.
    '''
    def __init__(self, vocab):
        self.vocab = vocab
        self.matcher = Matcher(vocab)
        self._set_needed_extensions()
        self.add_SECUQUANTITY_ent_to_matcher(self.matcher)
    
    
    def _set_needed_extensions(self):
        token_extensions = [
            {"name": "secuquantity", "kwargs": {"getter": get_token_secuquantity_float}},
            {"name": "secuquantity_unit", "kwargs": {"default": None}},
        ]
        span_extensions = [
            {"name": "secuquantity", "kwargs": {"getter": get_span_secuquantity_float}},
            {"name": "secuquantity_unit", "kwargs": {"default": None}},
        ]
        for each in span_extensions:
            _set_extension(Span, each["name"], each["kwargs"])
        for each in token_extensions:
            _set_extension(Token, each["name"], each["kwargs"])
    
    def add_SECUQUANTITY_ent_to_matcher(self, matcher: Matcher):
        matcher.add(
            "SECUQUANTITY_ENT",
            [*SECUQUANTITY_ENT_PATTERNS],
            on_match=_add_SECUQUANTITY_ent_regular_case,
        )
        logger.debug("added SECUQUANTITY patterns to matcher")

    def __call__(self, doc: Doc):
        self.matcher(doc)
        return doc


class SECUObjectMapper:
    '''
    This component handles the creation of SECU,
    QunatityRelation, SourceQunatityRelation and TODO: [add as we go here] objects and
    creates the necessary custom extension attributes.

    Custom extension attributes added with this component:
    Doc extensions:
        - ._.secu_objects (stores all SECU objects in the doc grouped by secu_key)
        - ._.secu_objects_map (stores SECU by index in doc)
        - ._.quantity_relation_map (
                maps the root token idx of the secuquantity
                to the created QuantityRelation
            )
        - ._.source_quantity_relation_map (
                maps the root token idx of the secuquantity
                to the created SourceQuantityRelation
            )

    This component must be placed after the SecuQuantityMatcher
    and the SECUMatcher in the spacy pipeline.
    '''
    def __init__(self, vocab):
        self.vocab = vocab
        self._secu_attr_getter = SecurityDependencyAttributeMatcher()
        self._set_needed_extensions()
    
    def _set_needed_extensions(self):
        doc_extensions = [
            {"name": "secu_objects", "kwargs": {"default": defaultdict(list)}}, #Type: Dict[str, List[SECU]]
            {"name": "secu_objects_map", "kwargs": {"default": dict()}}, #Type: Dict[int, SECU]
            {"name": "quantity_relation_map", "kwargs": {"default": dict()}}, #Type: Dict[int, QuantityRelation]
            {"name": "source_quantity_relation_map", "kwargs": {"default": dict()}}, #Type: Dict[int, SourceQuantityRelation], int being the index of the secuquantity of the relation
        ]
        for each in doc_extensions:
            _set_extension(Doc, each["name"], each["kwargs"])
    
    def create_secu_objects(self, doc: Doc) -> None:
        for secu in doc._.secus:
            if len(secu) == 1:
                secu = secu[0]
            secu_obj = SECU(secu, self._secu_attr_getter)
            doc._.secu_objects[secu_obj.secu_key].append(secu_obj)
            doc._.secu_objects_map[secu_obj.original.i] = secu_obj

    def set_quantity_relation_map(self, doc: Doc):
        for secu_key, secus in doc._.secu_objects.items():
            for secu in secus:
                if secu.quantity_relations:
                    for quantity_relation in secu.quantity_relations:
                        if isinstance(quantity_relation, QuantityRelation):
                            doc._.quantity_relation_map[quantity_relation.quantity.original.i] = quantity_relation
    
    def handle_source_quantity_relations(self, doc: Doc):
        for secu_key, secus in doc._.secu_objects.items():
            for secu in secus:
                for source_quantity_rel in self._get_source_quantity_relations(secu, doc):
                    if source_quantity_rel:
                        self._add_to_source_quantity_relation_map(doc, source_quantity_rel)
                        secu.add_source_quantity_relation(source_quantity_rel)
                    else:
                        break

    def _add_to_source_quantity_relation_map(self, doc: Doc, source_quantity_relation: SourceQuantityRelation) -> None:
        if isinstance(source_quantity_relation, SourceQuantityRelation):
            doc._.source_quantity_relation_map[source_quantity_relation.quantity.original.i] = source_quantity_relation
        else:
            raise TypeError(f"expecting SourceQuantityRelation, got: {type(source_quantity_relation)}")
    
    def _get_source_quantity_relations(self, secu: SECU, doc: Doc):
        possible_source_quantities = self._secu_attr_getter.get_possible_source_quantities(secu.original)
        if possible_source_quantities:
            for incomplete_source_quantity in possible_source_quantities:
                print(incomplete_source_quantity)
                quantity_relation = doc._.quantity_relation_map.get(incomplete_source_quantity.i, None)
                if quantity_relation is not None:
                    if quantity_relation.main_secu != secu:
                        # logger.warning((incomplete_source_quantity, "<- incomplete source quantity"))
                        # TODO[epic=important]: how can i include varying contexts between source and target security?
                        # source_context: SourceContext = self._secu_attr_getter._get_source_secu_context_through_secuquantity(incomplete_source_quantity)
                        # if source_context is None:
                        #     logger.warning(f"couldnt establish a source_context for this source_quantity_relation: {incomplete_source_quantity, quantity_relation}")
                        source_quantity_relation = SourceQuantityRelation(quantity_relation.quantity, quantity_relation.main_secu, source_secu=secu)
                        yield source_quantity_relation
        yield None
    
    def __call__(self, doc: Doc):
        # TODO[epic=maybe]: maybe add a map of SECU objects to sent indices so we can create a map of context for sents and therefor a context to SECU map?
        self.create_secu_objects(doc)
        self.set_quantity_relation_map(doc)
        self.handle_source_quantity_relations(doc)
        return doc
    


class SECUMatcher:
    '''
    This component adds SECU, SECUREF and SECUATTR entities.

    Custom extension attributes added with this component:
        Doc extensions:
            - ._.alias_list (set of aliases in doc as string eg: alias.text)
            - ._.tokens_to_alias_map (maps children token.i to parent alias span)
            - ._.single_secu_alias (?)
            - ._.single_secu_alias_tuples (aliases and no alias SECU's grouped by secu_key)
            - ._.is_alias (method, check wether this token is an alias to another token or not)
            - ._.get_alias (method, get aliases to the specified token)
            - ._.secus (helper to get all SECU ents)
        Span extensions:
            - ._.premerge_tokens (tokens before retokenization)
            - ._.was_merged (were any tokens in the Span retokenized)
            - ._.secu_key (string key to identify securities across singular and plural, WIP)
            - ._.amods (amods close in the dependency tree, according to patterns in filing_nlp_patterns.py)
        Token extensions:
            - ._.was_merged (is this token the result of retokenization)
            - ._.premerge_tokens (tokens before retokenization)
            - ._.source_span_unmerged (helper for ._.premerge_tokens)
            - ._.secu_key (string key to identify securities across singular and plural, WIP)
            - ._.amods (amods close in the dependency tree, according to patterns in filing_nlp_patterns.py)
            - ._.nsubjpass (nsubjpass were this token is the origin)
            - ._.adj (adjectives close in the dependency tree, according to patterns in filing_nlp_patterns.py)

    SideEffect:
        Indices shift, so place component as early as possible and after all other components which shift indices.
        Overwrites previously set entities if they conflict with SECU, SECUREF or SECUATTR
    '''
    def __init__(self, vocab):
        self.vocab = vocab
        self.matcher_SECU = Matcher(vocab)
        self.matcher_SECUREF = Matcher(vocab)
        self.matcher_SECUATTR = Matcher(vocab)
        self._set_needed_extensions()
        self.add_SECU_ent_to_matcher(self.matcher_SECU)
        self.add_SECUREF_ent_to_matcher(self.matcher_SECUREF)
        self.add_SECUATTR_ent_to_matcher(self.matcher_SECUATTR)

    def __call__(self, doc: Doc):
        self._init_span_labels(doc)
        self.chars_to_token_map = self.get_chars_to_tokens_map(doc)
        self.set_possible_alias_spans(doc)
        self.set_tokens_to_alias_map(doc)
        self.matcher_SECU(doc)
        update_doc_secus_spans(doc)
        self.matcher_SECUREF(doc)
        doc = self.handle_retokenize_SECU(doc)
        self.handle_SECU_special_cases(doc)
        doc = self.handle_retokenize_SECU(doc)
        self.matcher_SECUATTR(doc)
        return doc
    
    def _set_needed_extensions(self):
        token_extensions = [
            {"name": "source_span_unmerged", "kwargs": {"default": None}},
            {"name": "was_merged", "kwargs": {"default": False}},
            {"name": "amods", "kwargs": {"getter": token_amods_getter}},
            {"name": "nsubjpass", "kwargs": {"getter": token_nsubjpass_getter}},
            {"name": "adj", "kwargs": {"getter": token_adj_getter}},
            {"name": "premerge_tokens", "kwargs": {"getter": get_premerge_tokens_for_token}},
            {"name": "secu_key", "kwargs": {"getter": get_secu_key_extension}},
        ]
        span_extensions = [
            {"name": "secu_key", "kwargs": {"getter": get_secu_key_extension}},
            {"name": "amods", "kwargs": {"getter": span_amods_getter}},
            {"name": "premerge_tokens", "kwargs": {"getter": get_premerge_tokens_for_span}},
            {"name": "was_merged", "kwargs": {"default": False}},
        ]
        doc_extensions = [
            {"name": "alias_list", "kwargs": {"default": list()}},
            {"name": "tokens_to_alias_map", "kwargs": {"default": dict()}},
            {"name": "single_secu_alias", "kwargs": {"default": dict()}},
            {"name": "single_secu_alias_tuples", "kwargs": {"default": dict()}},
            {"name": "is_alias", "kwargs": {"method": is_alias}},
            {"name": "get_alias", "kwargs": {"method": get_alias}},
            {"name": "secus", "kwargs": {"getter": _get_SECU_in_doc}},
        ]

        for each in span_extensions:
            _set_extension(Span, each["name"], each["kwargs"])
        for each in doc_extensions:
            _set_extension(Doc, each["name"], each["kwargs"])
        for each in token_extensions:
            _set_extension(Token, each["name"], each["kwargs"])

    def handle_retokenize_SECU(self, doc: Doc):
        doc = retokenize_SECU(doc)
        update_doc_secus_spans(doc)
        self.chars_to_token_map = self.get_chars_to_tokens_map(doc)
        self.set_possible_alias_spans(doc)
        self.set_tokens_to_alias_map(doc)
        _set_single_secu_alias_map(doc)
        _set_single_secu_alias_map_as_tuples(doc)
        return doc

    def _init_span_labels(self, doc: Doc):
        doc.spans["SECU"] = []
        doc.spans["alias"] = []

    def handle_SECU_special_cases(self, doc: Doc):
        special_case_matcher = Matcher(self.vocab)
        self.add_alias_SECU_cases_to_matcher(doc, special_case_matcher)
        special_case_matcher(doc)

    def add_alias_SECU_cases_to_matcher(self, doc, matcher):
        special_patterns = []
        for secu_key, values in doc._.single_secu_alias_tuples.items():
            # logger.info(f"current key: {secu_key}")
            alias_tuples = doc._.single_secu_alias_tuples[secu_key]["alias"]
            # logger.info(f"current alias_tuples: {alias_tuples}")
            if alias_tuples and alias_tuples != []:
                for entry in alias_tuples:
                    # base_span = entry[0]
                    alias_span = entry[1]
                    special_patterns.append([{"LOWER": x.lower_} for x in alias_span])
                    if len(alias_span) > 1:
                        special_patterns.append(
                            [
                                *[{"LOWER": x.lower_} for x in alias_span[:-1]],
                                {"LEMMA": alias_span[-1].lemma_},
                            ]
                        )
        logger.debug(
            f"adding special alias_SECU patterns to matcher: {special_patterns}"
        )
        matcher.add("alias_special_cases", special_patterns, on_match=_add_SECU_ent)

    def get_chars_to_tokens_map(self, doc: Doc):
        chars_to_tokens = {}
        for token in doc:
            for i in range(token.idx, token.idx + len(token.text)):
                chars_to_tokens[i] = token.i
        return chars_to_tokens

    def get_tokens_to_alias_map(self, doc: Doc):
        tokens_to_alias_map = {}
        alias_spans = doc.spans["alias"]
        for span in alias_spans:
            for token in span:
                tokens_to_alias_map[token.i] = span
        return tokens_to_alias_map

    def get_possible_alias_spans(self, doc: Doc, chars_to_tokens: Dict):
        secu_alias_exclusions = set(["we", "us", "our"])
        spans = []
        parenthesis_pattern = re.compile(r"\([^(]*\)")
        possible_alias_pattern = re.compile(r"(?:\"|“)[a-zA-Z\s-]*(?:\"|”)")
        for match in re.finditer(parenthesis_pattern, doc.text):
            if match:
                start_idx = match.start()
                for possible_alias in re.finditer(
                    possible_alias_pattern, match.group()
                ):
                    if possible_alias:
                        start_token = chars_to_tokens.get(
                            start_idx + possible_alias.start()
                        )
                        end_token = chars_to_tokens.get(
                            start_idx + possible_alias.end() - 1
                        )
                        if (start_token is not None) and (end_token is not None):
                            alias_span = doc[start_token + 1 : end_token]
                            if alias_span.text in secu_alias_exclusions:
                                logger.debug(
                                    f"Was in secu_alias_exclusions -> discarded alias span: {alias_span}"
                                )
                                continue
                            spans.append(alias_span)
                        else:
                            logger.debug(
                                f"couldnt find start/end token for alias: {possible_alias}; start/end token: {start_token}/{end_token}"
                            )
        return spans

    def _set_possible_alias_spans(self, doc: Doc, spans: list[Span]):
        doc.spans["alias"] = spans

    def set_possible_alias_spans(self, doc: Doc):
        self._set_possible_alias_spans(
            doc, self.get_possible_alias_spans(doc, self.chars_to_token_map)
        )
        logger.debug(f"set alias spans: {doc.spans['alias']}")
        for span in doc.spans["alias"]:
            doc._.alias_list.append(span.text)
        logger.debug(f"set alias_list extension: {doc._.alias_list}")

    def set_tokens_to_alias_map(self, doc: Doc):
        doc._.tokens_to_alias_map = self.get_tokens_to_alias_map(doc)

    def add_SECUATTR_ent_to_matcher(self, matcher):
        patterns = [[{"LOWER": "exercise"}, {"LOWER": "price"}]]
        matcher.add("SECUATTR_ENT", [*patterns], on_match=_add_SECUATTR_ent)

    def add_SECUREF_ent_to_matcher(self, matcher):
        general_pre_sec_modifiers = ["convertible"]
        general_pre_sec_compound_modifiers = [
            [{"LOWER": "non"}, {"LOWER": "-"}, {"LOWER": "convertible"}],
            [{"LOWER": "pre"}, {"LOWER": "-"}, {"LOWER": "funded"}],
        ]
        general_affixes = ["series", "tranche", "class"]
        # exclude particles, conjunctions from regex match
        patterns = [
            [
                {"LOWER": {"IN": general_affixes}},
                {"TEXT": {"REGEX": "[a-zA-Z0-9]{1,3}", "NOT_IN": ["of"]}, "OP": "?"},
                {"LOWER": {"IN": general_pre_sec_modifiers}, "OP": "?"},
                {
                    "LOWER": {
                        "IN": ["preferred", "common", "warrant", "warrants", "ordinary"]
                    }
                },
                {"LOWER": {"IN": ["shares"]}},
            ],
            [
                {"LOWER": {"IN": general_pre_sec_modifiers}, "OP": "?"},
                {
                    "LOWER": {
                        "IN": ["preferred", "common", "warrant", "warrants", "ordinary"]
                    }
                },
                {"LOWER": {"IN": ["shares"]}},
            ],
            *[
                [
                    *general_pre_sec_compound_modifier,
                    {
                        "LOWER": {
                            "IN": [
                                "preferred",
                                "common",
                                "warrant",
                                "warrants",
                                "ordinary",
                            ]
                        }
                    },
                    {"LOWER": {"IN": ["shares"]}},
                ]
                for general_pre_sec_compound_modifier in general_pre_sec_compound_modifiers
            ],
            [
                {"LOWER": {"IN": general_affixes}},
                {"TEXT": {"REGEX": "[a-zA-Z0-9]{1,3}", "NOT_IN": ["of"]}, "OP": "?"},
                {"LOWER": {"IN": general_pre_sec_modifiers}, "OP": "?"},
                {"LOWER": {"IN": ["warrant", "warrants"]}},
                {"LOWER": {"IN": ["shares"]}},
            ],
        ]

        matcher.add("SECUREF_ENT", [*patterns], on_match=_add_SECUREF_ent)

    def add_SECU_ent_to_matcher(self, matcher):
        matcher.add(
            "SECU_ENT",
            [
                *SECU_ENT_REGULAR_PATTERNS,
                *SECU_ENT_DEPOSITARY_PATTERNS,
                *SECU_ENT_SPECIAL_PATTERNS,
            ],
            on_match=_add_SECU_ent,
        )


def _is_match_followed_by(doc: Doc, start: int, end: int, exclude: list[str]):
    if end == len(doc):
        end -= 1
    if doc[end].lower_ not in exclude:
        return False
    return True


def _is_match_preceeded_by(doc: Doc, start: int, end: int, exclude: list[str]):
    if (start == 0) or (exclude == []):
        return False
    if doc[start - 1].lower_ not in exclude:
        return False
    return True


def update_doc_secus_spans(doc: Doc):
    doc._.secus = []
    # logger.debug(f"updating doc._.secus")
    for ent in doc.ents:
        if ent.label_ == "SECU":
            # logger.debug(f"found SECU ent in doc.ents: {ent}")
            if doc._.is_alias(ent):
                continue
            # logger.debug(f"ent was not an alias, adding it.")
            doc._.secus.append(ent)


def add_SECU_to_spans(doc: Doc, entity: Span):
    if doc._.is_alias(entity):
        return
    else:
        # logger.debug(f"adding to SECU spans: {entity}")
        add_entity_to_spans(doc, entity, "SECU")


def add_entity_to_spans(doc: Doc, entity: Span, span_label: str):
    if not doc.spans.get(span_label):
        doc.spans[span_label] = []
    doc.spans[span_label].append(entity)


def _add_SECUREF_ent(matcher, doc: Doc, i: int, matches):
    _add_ent(
        doc,
        i,
        matches,
        "SECUREF",
        exclude_after=["agreement", "agent", "indebenture", "rights"],
    )


def _add_SECU_ent(matcher, doc: Doc, i: int, matches):
    # logger.debug(f"adding SECU ent: {matches[i]}")
    _add_ent(
        doc,
        i,
        matches,
        "SECU",
        exclude_after=[
            "agreement",
            "agent",
            "indebenture",
            # "rights",
            "shares",
        ],
        ent_callback=None,
        always_overwrite=["ORG", "WORK_OF_ART", "LAW"],
    )


def _add_SECUATTR_ent(matcher, doc: Doc, i: int, matches):
    _add_ent(doc, i, matches, "SECUATTR")


def _add_CONTRACT_ent(matcher, doc: Doc, i: int, matches):
    _add_ent(
        doc,
        i,
        matches,
        "CONTRACT",
        adjust_ent_before_add_callback=adjust_CONTRACT_ent_before_add,
    )


def adjust_CONTRACT_ent_before_add(entity: Span):
    logger.debug(f"adjust_contract_ent_before_add entity before adjust: {entity}")
    doc = entity.doc
    root = entity.root
    start = entity.start
    if start != 0:
        for i in range(entity.start - 1, 0, -1):
            if doc[i].dep_ == "compound" and root.is_ancestor(doc[i]):
                start = i
            else:
                break
    end = entity.end
    for i in range(entity.end, len(doc), 1):
        if doc[i].dep_ == "compound" and root.is_ancestor(doc[i]):
            end = i
        else:
            break
    entity = Span(doc, start, end, label="CONTRACT")
    logger.debug(f"adjust_contract_ent_before_add entity after adjust: {entity}")
    return entity


def _add_SECUQUANTITY_ent_regular_case(matcher, doc: Doc, i, matches):
    _, match_start, match_end = matches[i]
    match_tokens = [t for t in doc[match_start:match_end]]
    # logger.debug(f"handling SECUQUANTITY for match: {match_tokens}")
    match_id, start, _ = matches[i]
    end = None
    wanted_tokens = []
    for token in match_tokens:
        # logger.debug(f"token: {token}, ent_type: {token.ent_type_}")
        if ((token.ent_type_ in ["MONEY", "CARDINAL", "SECUQUANTITY"]) and (token.dep_ != "advmod")) or (
            token.dep_ == "nummod" and token.pos_ == "NUM"
        ):
            # end = token.i-1
            wanted_tokens.append(token.i)
    end = sorted(wanted_tokens)[-1] + 1 if wanted_tokens != [] else None
    start = sorted(wanted_tokens)[0] if wanted_tokens != [] else start
    if end is None:
        raise AttributeError(
            f"_add_SECUQUANTITY_ent_regular_case couldnt determine the end token of the entity, match_tokens: {match_tokens}"
        )
    entity = Span(doc, start, end, label="SECUQUANTITY")
    # logger.debug(f"Adding ent_label: SECUQUANTITY. Entity: {entity} [{start}-{end}], original_match:{doc[match_start:match_end]} [{match_start}-{match_end}]")
    _set_secuquantity_unit_on_span(match_tokens, entity)
    try:
        doc.ents += (entity,)
    except ValueError as e:
        if "[E1010]" in str(e):
            # logger.debug("handling overlapping ents")
            handle_overlapping_ents(doc, start, end, entity)


def _set_secuquantity_unit_on_span(match_tokens: Span, span: Span):
    if span.label_ != "SECUQUANTITY":
        raise TypeError(
            f"can only set secuquantity_unit on spans of label: SECUQUANTITY. got span: {span}"
        )
    unit = "COUNT"
    if "MONEY" in [t.ent_type_ for t in match_tokens]:
        unit = "MONEY"
    span._.secuquantity_unit = unit
    for token in span:
        token._.secuquantity_unit = unit


def _add_ent(
    doc: Doc,
    i,
    matches,
    ent_label: str,
    exclude_after: list[str] = [],
    exclude_before: list[str] = [],
    adjust_ent_before_add_callback: Optional[Callable] = None,
    ent_callback: Optional[Callable] = None,
    ent_exclude_condition: Optional[Callable] = None,
    always_overwrite: Optional[list[str]] = None,
):
    """add a custom entity through an on_match callback.

    Args:
        adjust_ent_before_add_callback:
            a callback that can be used to adjust the entity before
            it is added to the doc.ents. Takes the entity (Span) as single argument
            and should return a Span (the adjusted entity to be added)
        ent_callback: function which will be called, if entity was added, with the entity and doc as args.
        ent_exclude_condition: callable which returns bool and takes entity and doc as args."""
    match_id, start, end = matches[i]
    if (not _is_match_followed_by(doc, start, end, exclude_after)) and (
        not _is_match_preceeded_by(doc, start, end, exclude_before)
    ):
        entity = Span(doc, start, end, label=ent_label)
        if adjust_ent_before_add_callback is not None:
            entity = adjust_ent_before_add_callback(entity)
            if not isinstance(entity, Span):
                raise TypeError(f"entity should be a Span, got: {entity}")
        if ent_exclude_condition is not None:
            if ent_exclude_condition(doc, entity) is True:
                logger.debug(
                    f"ent_exclude_condition: {ent_exclude_condition} was True; not adding: {entity}"
                )
                return
        # logger.debug(f"entity: {entity}")
        try:
            doc.ents += (entity,)
            # logger.debug(f"Added entity: {entity} with label: {ent_label}")
        except ValueError as e:
            if "[E1010]" in str(e):
                # logger.debug(f"handling overlapping entities for entity: {entity}")
                handle_overlapping_ents(
                    doc, start, end, entity, overwrite_labels=always_overwrite
                )
        if (ent_callback) and (entity in doc.ents):
            ent_callback(doc, entity)


def handle_overlapping_ents(
    doc: Doc,
    start: int,
    end: int,
    entity: Span,
    overwrite_labels: Optional[list[str]] = None,
):
    previous_ents = set(doc.ents)
    conflicting_ents = get_conflicting_ents(
        doc, start, end, overwrite_labels=overwrite_labels
    )
    # logger.debug(f"conflicting_ents: {conflicting_ents}")
    # if (False not in [end-start >= k[0] for k in conflicting_ents]) and (conflicting_ents != []):
    if conflicting_ents != []:
        [previous_ents.remove(k) for k in conflicting_ents]
        # logger.debug(f"removed conflicting ents: {[k for k in conflicting_ents]}")
        previous_ents.add(entity)
        try:
            doc.ents = previous_ents
        except ValueError as e:
            if "E1010" in str(e):
                token_idx = int(
                    sorted(re.findall(r"token \d*", str(e)), key=lambda x: len(x))[0][
                        6:
                    ]
                )
                token_conflicting = doc[token_idx]
                logger.debug(
                    f"token conflicting: {token_conflicting} with idx: {token_idx}"
                )
                logger.debug(f"sent: {token_conflicting.sent}")
                conflicting_entity = []
                for i in range(token_idx, 0, -1):
                    if doc[i].ent_type_ == token_conflicting.ent_type_:
                        conflicting_entity.insert(0, doc[i])
                    else:
                        break
                for i in range(token_idx + 1, 10000000, 1):
                    if doc[i].ent_type_ == token_conflicting.ent_type_:
                        conflicting_entity.append(doc[i])
                    else:
                        break

                logger.debug(f"conflicting with entity: {conflicting_entity}")
                raise e
        # logger.debug(f"Added entity: {entity} with label: {entity.label_}")


def get_conflicting_ents(
    doc: Doc, start: int, end: int, overwrite_labels: Optional[list[str]] = None
):
    conflicting_ents = []
    seen_conflicting_ents = []
    covered_tokens = range(start, end)
    # logger.debug(f"new ent covering tokens: {[i for i in covered_tokens]}")
    for ent in doc.ents:
        # if ent.end == ent.start-1:
        #     covered_tokens = [ent.start]
        # else:
        # covered_tokens = range(ent.start, ent.end)
        # logger.debug(f"potentital conflicting ent: {ent}; with tokens: {[i for i in range(ent.start, ent.end)]}; and label: {ent.label_}")
        possible_conflicting_tokens_covered = [i for i in range(ent.start, ent.end - 1)]
        # check if we have a new longer ent or a new shorter ent with a required overwrite label
        if ((ent.start in covered_tokens) or (ent.end - 1 in covered_tokens)) or (
            any([i in covered_tokens for i in possible_conflicting_tokens_covered])
        ):
            if conflicting_ents == []:
                seen_conflicting_ents.append(ent)
            if ((ent.end - ent.start) <= (end - start)) or (
                ent.label_ in overwrite_labels if overwrite_labels else False
            ) is True:
                if conflicting_ents == []:
                    conflicting_ents = seen_conflicting_ents
                else:
                    conflicting_ents.append(ent)
            # else:
            # logger.debug(f"{ent} shouldnt be conflicting")

    return conflicting_ents


def _get_singular_or_plural_of_SECU_token(token):
    singular = PLURAL_SINGULAR_SECU_TAIL_MAP.get(token.lower_)
    plural = SINGULAR_PLURAL_SECU_TAIL_MAP.get(token.lower_)
    if singular is None:
        return plural
    else:
        return singular


class AgreementMatcher:
    """
    What do i want to tag?
        1) contractual agreements between two parties CONTRACT
        2) placements (private placement), public offerings -> specifiers for a CONTRACT or standing alone in text
        how else could the context of origin be present in a filing?
    """

    # change the name to something more fitting, also adjust the Language.factory for name
    def __init__(self, vocab):
        self.vocab = vocab
        self.matcher = Matcher(vocab)
        self.add_CONTRACT_ent_to_matcher()

    def add_CONTRACT_ent_to_matcher(self):
        patterns = [
            [{"LOWER": "agreement"}],
        ]
        self.matcher.add("contract", patterns, on_match=_add_CONTRACT_ent)

    def add_PLACEMENT_ent_to_matcher(self):
        pass

    def __call__(self, doc: Doc):
        self.matcher(doc)
        return doc

    # def agreement_callback()




@Language.factory("secu_matcher")
def create_secu_matcher(nlp, name):
    return SECUMatcher(nlp.vocab)

@Language.factory("secuquantity_matcher")
def create_secuquantity_matcher(nlp, name):
    return SECUQuantityMatcher(nlp.vocab)

@Language.factory("secu_act_matcher")
def create_secu_act_matcher(nlp, name):
    return SecurityActMatcher(nlp.vocab)

@Language.factory("secu_object_mapper")
def create_secu_object_mapper(nlp, name):
    return SECUObjectMapper(nlp.vocab)


@Language.factory("security_law_retokenizer")
def create_regex_retokenizer(nlp, name):
    return FilingsSecurityLawRetokenizer(nlp.vocab)


@Language.factory("common_financial_retokenizer")
def create_common_financial_retokenizer(nlp, name):
    return CommonFinancialRetokenizer(nlp.vocab)


@Language.factory("agreement_matcher")
def create_agreement_matcher(nlp, name):
    return AgreementMatcher(nlp.vocab)







class SpacyFilingTextSearch:
    _instance = None
    # make this a singleton/get it from factory through cls._instance so we can avoid
    # the slow process of adding patterns (if we end up with a few 100)
    def __init__(self):
        self.secu_attr_getter = SecurityDependencyAttributeMatcher()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpacyFilingTextSearch, cls).__new__(cls)
            cls._instance.nlp = spacy.load("en_core_web_lg")
            cls._instance.nlp.add_pipe("secu_act_matcher", first=True)
            cls._instance.nlp.add_pipe(
                "security_law_retokenizer", after="secu_act_matcher"
            )
            cls._instance.nlp.add_pipe(
                "common_financial_retokenizer", after="security_law_retokenizer"
            )
            cls._instance.nlp.add_pipe("negation_setter")
            cls._instance.nlp.add_pipe("secu_matcher")
            cls._instance.nlp.add_pipe("secuquantity_matcher")
            cls._instance.nlp.add_pipe("certainty_setter")
            cls._instance.nlp.add_pipe("secu_object_mapper")
            cls._instance.nlp.add_pipe("agreement_matcher")
            # cls._instance.nlp.add_pipe("coreferee")
        return cls._instance
    

    def handle_match_formatting(
        self,
        match: tuple[str, list[Token]],
        formatting_dict: Dict[str, Callable],
        doc: Doc,
        *args,
        **kwargs,
    ) -> tuple[str, dict]:
        try:
            # match_id = doc.vocab.strings[match[0]]
            match_id = match[0]
            logger.debug(f"string_id of match: {match_id}")
        except KeyError:
            raise AttributeError(
                f"No string_id found for this match_id in the doc: {match_id}"
            )
        tokens = match[1]
        try:
            formatting_func = formatting_dict[match_id]
        except KeyError:
            raise AttributeError(
                f"No formatting function associated with this match_id: {match_id}"
            )
        else:
            return (match_id, formatting_func(tokens, doc, *args, **kwargs))

    def match_secu_with_dollar_CD(self, doc: Doc, secu: Span):
        dep_matcher = DependencyMatcher(self.nlp.vocab, validate=True)
        anchor_pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"ENT_TYPE": "SECU", "LOWER": secu.root.lower_},
            }
        ]
        incomplete_patterns = [
            [
                {
                    "LEFT_ID": "anchor",
                    "REL_OP": "<",
                    "RIGHT_ID": "verb1",
                    "RIGHT_ATTRS": {
                        "POS": "VERB",
                        "LEMMA": {"IN": ["purchase", "have"]},
                    },
                },
                {
                    "LEFT_ID": "verb1",
                    "REL_OP": ">>",
                    "RIGHT_ID": "CD_",
                    "RIGHT_ATTRS": {"TAG": "CD"},
                },
                {
                    "LEFT_ID": "CD_",
                    "REL_OP": ">",
                    "RIGHT_ID": "nmod",
                    "RIGHT_ATTRS": {"DEP": "nmod", "TAG": "$"},
                },
            ]
        ]
        patterns = add_anchor_pattern_to_patterns(anchor_pattern, incomplete_patterns)
        dep_matcher.add("secu_cd", patterns)
        matches = dep_matcher(doc)
        if matches:
            matches = _convert_dep_matches_to_spans(doc, matches)
            return [match[1] for match in matches]
        return []

    def get_queryable_similar_spans_from_lower(self, doc: Doc, span: Span):
        """
        look for similar spans by matching through regex on the
        combined .text_with_ws of the tokens of span with re.I flag
        (checking for last token singular or plural aslong as it is registered
        in PLURAL_SINGULAR_SECU_TAIL_MAP and SINGULAR_PLURAL_SECU_TAIL_MAP.
        """
        # adjusted for merged SECUs
        matcher = Matcher(self.nlp.vocab)
        tokens = []
        to_check = []
        if span._.was_merged if span.has_extension("was_merged") else False:
            tokens = list(span._.premerge_tokens)
        else:
            tokens = [i for i in span]
        # add base case to check for later
        to_check.append(tokens)
        # see if we find an additional plural or singular case based on the last token
        tail_lower = _get_singular_or_plural_of_SECU_token(tokens[-1])
        if tail_lower:
            additional_case = tokens.copy()
            additional_case.pop()
            additional_case.append(tail_lower)
            to_check.append(additional_case)
        re_patterns = []
        # convert to string
        for entry in to_check:
            re_patterns.append(
                re.compile(
                    "".join(
                        [
                            x.text_with_ws if isinstance(x, (Span, Token)) else x
                            for x in entry
                        ]
                    ),
                    re.I,
                )
            )
        found_spans = set()
        for re_pattern in re_patterns:
            logger.debug(f"checking with re_pattern: {re_pattern}")
            for result in self._find_full_text_spans(re_pattern, doc):
                logger.debug(f"found a match: {result}")
                if result not in found_spans and result != span:
                    found_spans.add(result)
        matcher_patterns = []
        for entry in to_check:
            matcher_patterns.append(
                [
                    {"LOWER": x.lower_ if isinstance(x, (Span, Token)) else x}
                    for x in entry
                ]
            )
        logger.debug(f"working with matcher_patterns: {matcher_patterns}")
        matcher.add("similar_spans", matcher_patterns)
        matches = matcher(doc)
        match_results = _convert_matches_to_spans(doc, filter_matches(matches))
        logger.debug(f"matcher converted matches: {match_results}")
        if match_results is not None:
            for match in match_results:
                if match not in found_spans and match != span:
                    found_spans.add(match)
        logger.info(f"Found {len(found_spans)} similar spans for {span}")
        logger.debug(f"similar spans: {found_spans}")
        return list(found_spans) if len(found_spans) != 0 else None

    def _find_full_text_spans(self, re_term: re.Pattern, doc: Doc):
        for match in re.finditer(re_term, doc.text):
            start, end = match.span()
            result = doc.char_span(start, end)
            if result:
                yield result

    def get_prep_phrases(self, doc: Doc):
        phrases = []
        seen = set()
        for token in doc:
            if token.dep_ == "prep":
                subtree = [i for i in token.subtree]
                new = True
                for t in subtree:
                    if t in seen:
                        new = False
                if new is True:
                    phrases.append(doc[subtree[0].i : subtree[-1].i])
                    for t in subtree:
                        if t not in seen:
                            seen.add(t)
        return phrases

    def get_verbal_phrases(self, doc: Doc):
        phrases = []
        for token in doc:
            if token.pos_ == "VERB":
                subtree = [i for i in token.subtree]
                phrases.append(doc[subtree[0].i : subtree[-1].i])
        return phrases

    def _create_span_dependency_matcher_dict_lower(self, secu: Span) -> dict:
        """
        create a list of dicts for dependency match patterns.
        the root token will have RIGHT_ID of 'anchor'
        """
        secu_root_token = secu.root
        if secu_root_token is None:
            return None
        root_pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {
                    "ENT_TYPE": secu_root_token.ent_type_,
                    "LOWER": secu_root_token.lower_,
                },
            }
        ]
        if secu_root_token.children:
            for idx, token in enumerate(secu_root_token.children):
                if token in secu:
                    root_pattern.append(
                        {
                            "LEFT_ID": "anchor",
                            "REL_OP": ">",
                            "RIGHT_ID": token.lower_ + "__" + str(idx),
                            "RIGHT_ATTRS": {"LOWER": token.lower_},
                        }
                    )
        return root_pattern

    def match_secu_expiry(self, doc: Doc, secu: Span):
        secu_root_pattern = self._create_span_dependency_matcher_dict_lower(secu)
        if secu_root_pattern is None:
            logger.warning(f"couldnt get secu_root_pattern for secu: {secu}")
            return
        dep_matcher = DependencyMatcher(self.nlp.vocab, validate=True)
        patterns = add_anchor_pattern_to_patterns(
            secu_root_pattern, SECU_EXPIRY_PATTERNS
        )
        dep_matcher.add("expiry", patterns)
        matches = dep_matcher(doc)
        logger.debug(f"raw expiry matches: {matches}")
        for i in doc:
            print(i, i.lemma_)
        if matches:
            matches = _convert_dep_matches_to_spans(doc, matches)
            logger.info(f"matches: {matches}")
            formatted_matches = []
            for match in matches:
                formatted_matches.append(self._format_expiry_match(match[1]))
            return formatted_matches
        logger.debug("no matches for epxiry found in this sentence")

    def _format_expiry_match(self, match):
        # print([(i.ent_type_, i.text) for i in match])
        logger.debug("_format_expiry_match:")
        if match[-1].ent_type_ == "ORDINAL":
            match = match[:-1]
        if match[-1].lower_ != "anniversary":
            try:
                date = "".join([i.text_with_ws for i in match[-1].subtree])
                logger.debug(f"     date tokens joined: {date}")
                date = pd.to_datetime(date)
            except Exception as e:
                logger.debug(f"     failed to format expiry match: {match}")
            else:
                return date
        else:
            date_tokens = [i for i in match[-1].subtree]
            # print(date_tokens, [i.dep_ for i in date_tokens])
            date_spans = []
            date = []
            for token in date_tokens:
                if token.dep_ != "prep":
                    date.append(token)
                else:
                    date_spans.append(date)
                    date = []
            if date != []:
                date_spans.append(date)
            logger.debug(f"     date_spans: {date_spans}")
            dates = []
            deltas = []
            if len(date_spans) > 0:
                # handle anniversary with issuance date
                for date in date_spans:
                    possible_date = formater.coerce_tokens_to_datetime(date)
                    # print(f"possible_date: {possible_date}")
                    if possible_date:
                        dates.append(possible_date)
                    else:
                        possible_delta = formater.coerce_tokens_to_timedelta(date)
                        # print(f"possible_delta: {possible_delta}")
                        if possible_delta:
                            for delta in possible_delta:
                                deltas.append(delta[0])
                if len(dates) == 1:
                    if len(deltas) == 1:
                        return dates[0] + deltas[0]
                    if len(delta) == 0:
                        return dates[0]
                if len(dates) > 1:
                    raise UnclearInformationExtraction(
                        f"unhandled case of extraction found more than one date for the expiry: {dates}"
                    )
                if len(deltas) == 1:
                    return deltas[0]
                elif len(deltas) > 1:
                    raise UnclearInformationExtraction(
                        f"unhandled case of extraction found more than one timedelta for the expiry: {deltas}"
                    )
            return None

    def match_secu_exercise_price(self, doc: Doc, secu: Span):
        dep_matcher = DependencyMatcher(self.nlp.vocab, validate=True)
        logger.debug("match_secu_exercise_price:")
        logger.debug(f"     secu: {secu}")

        """
        acl   (SECU) <- VERB (purchase) -> 					 prep (of | at) -> [pobj (price) -> compound (exercise)] -> prep (of) -> pobj CD
		nsubj (SECU) <- VERB (have)  	->				     			       [dobj (price) -> compound (exercise)] -> prep (of) -> pobj CD
		nsubj (SECU) <- VERB (purchase) -> 					 prep (at) ->      [pobj (price) -> compound (exercise)] -> prep (of) -> pobj CD
		nsubj (SECU) <- VERB (purchase) -> conj (remain)  -> prep (at) ->      [pobj (price) -> compound (exercise)] -> prep (of) -> pobj CD
		nsubj (SECU) <- VERB (purchase) -> 					 prep (at) ->      [pobj (price) -> compound (exercise)] -> prep (of) -> pobj CD
        
        relcl (SECU) <- VERB (purchase) >> prep(at) -> 
        """
        secu_root_dict = self._create_span_dependency_matcher_dict_lower(secu)
        if secu_root_dict is None:
            return None
        patterns = add_anchor_pattern_to_patterns(
            secu_root_dict, SECU_EXERCISE_PRICE_PATTERNS
        )
        dep_matcher.add("exercise_price", patterns)
        matches = dep_matcher(doc)
        logger.debug(f"raw exercise_price matches: {matches}")
        if matches:
            matches = _convert_dep_matches_to_spans(doc, matches)
            logger.info(f"matches: {matches}")
            secu_dollar_CD = self.match_secu_with_dollar_CD(doc, secu)
            if len(secu_dollar_CD) > 1:
                logger.info(
                    f"unhandled ambigious case of exercise_price match: matches: {matches}; secu_dollar_CD: {secu_dollar_CD}"
                )
                return None

            def _get_CD_object_from_match(match):
                for token in match:
                    if token.tag_ == "CD":
                        return formater.quantity_string_to_float(token.text)

            return [_get_CD_object_from_match(match[1]) for match in matches]

    def match_prospectus_relates_to(self, text):
        # INVESTIGATIVE
        pattern = [
            # This prospectus relates to
            {"LOWER": "prospectus"},
            {"LEMMA": "relate"},
            {"LOWER": "to"},
            {"OP": "*", "IS_SENT_START": False},
        ]
        matcher = Matcher(self.nlp.vocab)
        matcher.add("relates_to", [pattern])
        doc = self.nlp(text)
        matches = _convert_matches_to_spans(
            doc, filter_matches(matcher(doc, as_spans=False))
        )
        return matches if matches is not None else []

    def match_aggregate_offering_amount(self, doc: Doc):
        # INVESTIGATIVE
        pattern = [
            {"ENT_TYPE": "SECU", "OP": "*"},
            {"IS_SENT_START": False, "OP": "*"},
            {"LOWER": "aggregate"},
            {"LOWER": "offering"},
            {"OP": "?"},
            {"OP": "?"},
            {"LOWER": "up"},
            {"LOWER": "to"},
            {"ENT_TYPE": "MONEY", "OP": "+"},
        ]
        matcher = Matcher(self.nlp.vocab)
        matcher.add("offering_amount", [pattern])
        matches = _convert_matches_to_spans(
            doc, filter_matches(matcher(doc, as_spans=False))
        )
        return matches if matches is not None else []

    def match_outstanding_shares(self, text):
        # WILL BE REPLACED
        pattern1 = [
            {"LEMMA": "base"},
            {"LEMMA": {"IN": ["on", "upon"]}},
            {"ENT_TYPE": {"IN": ["CARDINAL", "SECUQUANTITY"]}},
            {"IS_PUNCT": False, "OP": "*"},
            {"LOWER": "shares"},
            {"IS_PUNCT": False, "OP": "*"},
            {"LOWER": {"IN": ["outstanding", "stockoutstanding"]}},
            {"IS_PUNCT": False, "OP": "*"},
            {"LOWER": {"IN": ["of", "on"]}},
            {"ENT_TYPE": "DATE", "OP": "+"},
            {"ENT_TYPE": "DATE", "OP": "?"},
            {"OP": "?"},
            {"ENT_TYPE": "DATE", "OP": "*"},
        ]
        pattern2 = [
            {"LEMMA": "base"},
            {"LEMMA": {"IN": ["on", "upon"]}},
            {"ENT_TYPE": {"IN": ["CARDINAL", "SECUQUANTITY"]}},
            {"IS_PUNCT": False, "OP": "*"},
            {"LOWER": "outstanding"},
            {"LOWER": "shares"},
            {"IS_PUNCT": False, "OP": "*"},
            {"LOWER": {"IN": ["of", "on"]}},
            {"ENT_TYPE": "DATE", "OP": "+"},
            {"ENT_TYPE": "DATE", "OP": "?"},
            {"OP": "?"},
            {"ENT_TYPE": "DATE", "OP": "*"},
        ]
        pattern3 = [
            {"LOWER": {"IN": ["of", "on"]}},
            {"ENT_TYPE": "DATE", "OP": "+"},
            {"ENT_TYPE": "DATE", "OP": "?"},
            {"OP": "?"},
            {"ENT_TYPE": "DATE", "OP": "*"},
            {"OP": "?"},
            {"ENT_TYPE": {"IN": ["CARDINAL", "SECUQUANTITY"]}},
            {"IS_PUNCT": False, "OP": "*"},
            {"LOWER": {"IN": ["issued", "outstanding"]}},
        ]
        matcher = Matcher(self.nlp.vocab)
        matcher.add("outstanding", [pattern1, pattern2, pattern3])
        doc = self.nlp(text)
        possible_matches = matcher(doc, as_spans=False)
        if possible_matches == []:
            logger.debug("no matches for outstanding shares found")
            return []
        matches = _convert_matches_to_spans(doc, filter_matches(possible_matches))
        values = []
        for match in matches:
            value = {"date": ""}
            for ent in match.ents:
                print(ent, ent.label_)
                if ent.label_ in ["CARDINAL", "SECUQUANTITY"]:
                    value["amount"] = int(str(ent).replace(",", ""))
                if ent.label_ == "DATE":
                    value["date"] = " ".join([value["date"], ent.text])
            value["date"] = pd.to_datetime(value["date"])
            try:
                validate_filing_values(value, ["date", "amount"])
            except AttributeError:
                pass
            else:
                values.append(value)
        return values

    # def match_issuabel_secu_primary(self, doc: Doc, primary_secu: Span)

    def match_issuable_secu_primary(self, doc: Doc):
        # WILL BE REPLACED
        secu_transformative_actions = ["exercise", "conversion", "redemption"]
        part1 = [
            [
                {"ENT_TYPE": "SECUQUANTITY", "OP": "+"},
                {"OP": "?"},
                {"LOWER": "of"},
                {"LOWER": "our", "OP": "?"},
                {"ENT_TYPE": "SECU", "OP": "+"},
                {"LOWER": "issuable"},
                {"LOWER": "upon"},
                {"LOWER": "the", "OP": "?"},
            ]
        ]
        part2 = [
            [
                {"LOWER": {"IN": ["exercise", "conversion"]}},
                {"LOWER": "price"},
                {"LOWER": "of"},
                {"OP": "?", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
                {"ENT_TYPE": "MONEY"},
                {"OP": "*", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
            ],
            [
                {"LOWER": {"IN": ["exercise", "conversion"]}},
                {"LOWER": {"IN": ["price", "prices"]}},
                {"LOWER": "ranging"},
                {"LOWER": "from"},
                {"OP": "?", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
                {"ENT_TYPE": "MONEY"},
                {"OP": "*", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
                {"LOWER": "to"},
                {"OP": "?", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
                {"ENT_TYPE": "MONEY"},
                {"OP": "*", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
            ],
        ]
        primary_secu_pattern = []
        for transformative_action in secu_transformative_actions:
            p1 = part1[0]
            for p2 in part2:
                pattern = [
                    *p1,
                    {"LOWER": transformative_action},
                    {
                        "OP": "*",
                        "IS_SENT_START": False,
                        "LOWER": {"NOT_IN": [";", "."]},
                    },
                    *p2,
                    {"LOWER": "of"},
                    {"ENT_TYPE": "SECU", "OP": "+"},
                ]
                primary_secu_pattern.append(pattern)
        pattern2 = [
            [
                {"ENT_TYPE": "SECUQUANTITY", "OP": "+"},
                {"OP": "?"},
                {"LOWER": "of"},
                {"LOWER": "our", "OP": "?"},
                {"ENT_TYPE": "SECU", "OP": "+"},
                {"LOWER": "issuable"},
                {"LOWER": "upon"},
                {"LOWER": "the", "OP": "?"},
                {"LOWER": transformative_action},
                {"LOWER": "of"},
                {"OP": "*", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
                {"ENT_TYPE": "SECU", "OP": "+"},
            ]
            for transformative_action in secu_transformative_actions
        ]
        [primary_secu_pattern.append(x) for x in pattern2]
        matcher = Matcher(self.nlp.vocab)
        matcher.add("secu_issuabel_relation_primary_secu", [*primary_secu_pattern])
        matches = _convert_matches_to_spans(
            doc, filter_matches(matcher(doc, as_spans=False))
        )

    def match_issuable_secu_no_primary(self, doc: Doc):
        # WILL BE REPLACED
        secu_transformative_actions = ["exercise", "conversion", "redemption"]
        part1 = [
            [
                {"ENT_TYPE": "SECUQUANTITY", "OP": "+"},
                {"OP": "?"},
                {"LOWER": "issuable"},
                {"LOWER": "upon"},
            ]
        ]
        part2 = [
            [
                {"LOWER": {"IN": ["exercise", "conversion"]}},
                {"LOWER": "price"},
                {"LOWER": "of"},
                {"OP": "?", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
                {"ENT_TYPE": "MONEY"},
                {"OP": "*", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
            ],
            [
                {"LOWER": {"IN": ["exercise", "conversion"]}},
                {"LOWER": {"IN": ["price", "prices"]}},
                {"LOWER": "ranging"},
                {"LOWER": "from"},
                {"OP": "?", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
                {"ENT_TYPE": "MONEY"},
                {"OP": "*", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
                {"LOWER": "to"},
                {"OP": "?", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
                {"ENT_TYPE": "MONEY"},
                {"OP": "*", "IS_SENT_START": False, "LOWER": {"NOT_IN": [";", "."]}},
            ],
        ]
        no_primary_secu_pattern = []
        for transformative_action in secu_transformative_actions:
            for p2 in part2:
                pattern = [
                    *part1[0],
                    {"LOWER": transformative_action},
                    {
                        "OP": "*",
                        "IS_SENT_START": False,
                        "LOWER": {"NOT_IN": [";", "."]},
                    },
                    *p2,
                    {"LOWER": "of"},
                    {"ENT_TYPE": "SECU", "OP": "+"},
                ]
                no_primary_secu_pattern.append(pattern)
        matcher = Matcher(self.nlp.vocab)
        matcher.add(
            "secu_issuable_relation_no_primary_secu", [*no_primary_secu_pattern]
        )
        matches = _convert_matches_to_spans(
            doc, filter_matches(matcher(doc, as_spans=False))
        )
        return matches

    def match_issuable_secu_no_exercise_price(self, doc: Doc):
        # WILL BE REPLACED
        secu_transformative_actions = ["exercise", "conversion", "redemption"]
        part1 = [
            [
                {"ENT_TYPE": "SECUQUANTITY", "OP": "+"},
                {"OP": "?"},
                {"LOWER": "issuable"},
                {"LOWER": "upon"},
            ],
            [
                {"ENT_TYPE": "SECUQUANTITY", "OP": "+"},
                {"OP": "?"},
                {"LOWER": "of"},
                {"LOWER": "our", "OP": "?"},
                {"ENT_TYPE": "SECU", "OP": "+"},
                {"LOWER": "issuable"},
                {"LOWER": "upon"},
                {"LOWER": "the", "OP": "?"},
            ],
        ]
        patterns = []
        for transformative_action in secu_transformative_actions:
            for p1 in part1:
                pattern = [
                    *p1,
                    {"LOWER": transformative_action},
                    {
                        "IS_SENT_START": False,
                        "LOWER": {"NOT_IN": [";", "."]},
                        "ENT_TYPE": {"NOT_IN": ["SECUATTR"]},
                        "OP": "*",
                    },
                    {"LOWER": "of"},
                    {"ENT_TYPE": "SECU", "OP": "+"},
                ]
                patterns.append(pattern)
        matcher = Matcher(self.nlp.vocab)
        matcher.add("secu_issuable_relation_no_exercise_price", [*patterns])
        matches = _convert_matches_to_spans(
            doc, filter_matches(matcher(doc, as_spans=False))
        )
        return matches


def token_adj_getter(target: Token):
    if not isinstance(target, Token):
        raise TypeError("target must be a Token, got {}".format(type(target)))
    if target.ent_type_ == "SECUQUANTITY":
        return _secuquantity_adj_getter(target)
    if target.ent_type_ == "SECU":
        return _secu_adj_getter(target)
    return _regular_adj_getter(target)


def _regular_adj_getter(target: Token):
    adjs = []
    if target.children:
        for child in target.children:
            if child.pos_ == "ADJ":
                adjs.append(child)
    return adjs


def _secu_adj_getter(target: Token):
    if not isinstance(target, Token):
        raise TypeError("target must be a Token, got {}".format(type(target)))
    if target.ent_type_ != "SECU":
        raise ValueError("target must be a SECU, got {}".format(target.ent_type_))
    # only check in direct children
    adjs = []
    if target.children:
        for child in target.children:
            if child.pos_ == "ADJ":
                adjs.append(child)
    return adjs


def _secuquantity_adj_getter(target: Token):
    if not isinstance(target, Token):
        raise TypeError("target must be a Token")
    if not target.ent_type_ == "SECUQUANTITY":
        raise ValueError("target must be of ent_type_ 'SECUQUANTITY'")
    adjs = []
    amods = token_amods_getter(target)
    if amods:
        adjs += amods
    if target.head:
        if target.head.dep_ == "nummod":
            if target.head.lower_ in SECUQUANTITY_UNITS:
                nsubjpass = token_nsubjpass_getter(target.head)
                if nsubjpass:
                    if nsubjpass.get("adj", None):
                        adjs += nsubjpass["adj"]
    return adjs if adjs != [] else None


def token_nsubjpass_getter(target: Token):
    if not isinstance(target, Token):
        raise TypeError("target must be a Token. got {}".format(type(target)))
    if target.dep_ == "nsubjpass":
        nsubjpass = {}
        head = target.head
        if head.pos_ == "VERB":
            nsubjpass["verb"] = head
            nsubjpass["adj"] = []
            for child in head.children:
                if child.pos_ == "ADJ":
                    nsubjpass["adj"].append(child)
            if nsubjpass["adj"] == []:
                nsubjpass["adj"] = None
            return nsubjpass
        else:
            nsubjpass = [target] + [i for i in target.children]
            logger.info(
                f"found following nsubjpass with a different head than a verb, head -> {head, head.pos_, head.dep_}, whole -> {nsubjpass}"
            )
    return None


def span_amods_getter(target: Span):
    if not isinstance(target, Span):
        raise TypeError("target must be a Span, got: type {}".format(type(target)))
    # logger.debug(f"getting amods for: {target, target.label_}")
    if target.label_ == "SECUQUANTITY":
        seen_tokens = set([i for i in target])
        heads_with_dep = []
        for token in target:
            logger.debug(f"token: {token, token.dep_}")
            if token.dep_ == "nummod":
                if token.head:
                    head = token.head
                    if head in seen_tokens:
                        continue
                    else:
                        # always nouns?
                        heads_with_dep.append(head)
                        seen_tokens.add(head)
        amods = []
        if len(heads_with_dep) > 0:
            for head in heads_with_dep:
                if head.lower_ in SECUQUANTITY_UNITS:
                    amods += _get_amods_of_target(head)
        amods += _get_amods_of_target(target)
        return amods if amods != [] else None
    else:
        amods = _get_amods_of_target(target)
        return amods if amods != [] else None


def token_amods_getter(target: Token):
    if not isinstance(target, Token):
        raise TypeError("target must be a Token, got: type {}".format(type(target)))
    # logger.debug(f"getting amods for: {target, target.ent_type_}")
    amods = []
    if target.ent_type_ == "SECUQUANTITY":
        if target.dep_ == "nummod":
            # also get the amods of certain heads
            if target.head:
                if target.head.lower_ in SECUQUANTITY_UNITS:
                    amods += _get_amods_of_target(target.head)
    amods += _get_amods_of_target(target)
    return amods if amods != [] else None


def _get_amods_of_target(target: Span | Token) -> list:
    if isinstance(target, Span):
        return _get_amods_of_target_span(target)
    elif isinstance(target, Token):
        return _get_amods_of_target_token(target)
    else:
        raise TypeError("target must be a Span or Token")


def _get_amods_of_target_span(target: Span):
    """get amods of first order of target. Needs to have dep_ set."""
    amods = []
    amods_to_ignore = set([token if token.dep_ == "amod" else None for token in target])
    for token in target:
        pool = [i for i in token.children] + [token.head]
        for possible_match in pool:
            if possible_match.dep_ == "amod" and possible_match not in amods_to_ignore:
                if possible_match not in amods:
                    amods.append(possible_match)
    return amods


def _get_amods_of_target_token(target: Token):
    amods = []
    pool = [i for i in target.children] + [target.head]
    for possible_match in pool:
        if possible_match.dep_ == "amod":
            if possible_match not in amods:
                amods.append(possible_match)
    return amods



def filter_matches(matches):
    """works as spacy.util.filter_spans but for matches"""
    if len(matches) <= 1:
        return matches
    # logger.debug(f"pre filter matches: {[m for m in matches]}")
    get_sort_key = lambda match: (match[2] - match[1], -match[1])
    sorted_matches = sorted(matches, key=get_sort_key, reverse=True)
    result = []
    seen_tokens: Set[int] = set()
    for match in sorted_matches:
        # Check for end - 1 here because boundaries are inclusive
        if match[1] not in seen_tokens and match[2] - 1 not in seen_tokens:
            result.append(match)
            seen_tokens.update(range(match[1], match[2]))
    result = sorted(result, key=lambda match: match[1])
    return result


def filter_dep_matches(matches):
    """take the longest of the dep matches with same start token, discard rest"""
    if len(matches) <= 1:
        return matches
    len_map = {}
    result_map = {}
    # logger.debug(f"filtering dep matches: {matches}")
    for match in matches:
        if match[1][0] not in len_map.keys():
            len_map[match[1][0]] = len(match[1])
            result_map[match[1][0]] = match
        else:
            current_len = len(match[1])
            if current_len <= len_map[match[1][0]]:
                pass
            else:
                len_map[match[1][0]] = len(match[1])
                result_map[match[1][0]] = match
    return [v for _, v in result_map.items()]


def _convert_matches_to_spans(doc, matches):
    m = []
    for match in matches:
        m.append(doc[match[1] : match[2]])
    return m


def _convert_dep_matches_to_spans(doc, matches) -> list[tuple[str, list[Token]]]:
    m = []
    for match in matches:
        print(f"match: {match}")
        m.append((match[0], [doc[f] for f in match[1]]))
    return m


def validate_filing_values(values, attributes):
    """validate a flat filing value"""
    for attr in attributes:
        if attr not in values.keys():
            raise AttributeError
