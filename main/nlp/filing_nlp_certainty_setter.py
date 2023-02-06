from spacy.tokens import Token, Span, Doc
from spacy.matcher import DependencyMatcher, Matcher
from spacy import Language
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

CERTAINTY_LEVEL_MAP_MD = {
    "must": 0.9,
    "will": 0.8,
    "would": 0.7,
    "ought": 0.45,
    "should": 0.4,
    "can": 0.3,
    "could": 0.25,
    "may": 0.2,
    "might": 0.15,
}

CERTAINTY_LEVEL_MAP_ADV = {
    "definitly": 1.0,
    "surely": 1.0,
    "certainly": 1.0,
    "likely": 0.8,
    "probably": 0.75,
    "perhaps": 0.5,
    "possibly": 0.3
}

class CertaintyInfo:
    def __init__(self, marker_idx: int, doc: Doc):
        self.marker_idx = marker_idx
        self.doc = doc
    
    def get_marker(self):
        return self.doc[self.marker_idx] 
    
    def get_marker_scope(self):
        return self.doc._.certainty_marker_map.get(self.marker_idx)

    def determine_level(self):
        marker = self.doc[self.marker_idx]
        if marker.tag_ == "MD":
            certainty_level_map_entry = CERTAINTY_LEVEL_MAP_MD.get(marker.lemma_, None)
        elif marker.pos_ == "ADV":
            certainty_level_map_entry = CERTAINTY_LEVEL_MAP_ADV.get(marker.lower_, None)
        else:
            certainty_level_map_entry = None
        if not certainty_level_map_entry:
            logger.warning(f"folowing marker isnt assigned a degree [0, 1[ of certainty, add {marker} to the CERTAINTY_LEVEL_MAP, CERTAINTY_LEVEL_MAP_MD or CERTAINTY_LEVEL_MAP_ADV to resolve and not see this message.")
            return 1.0
        else:
            return certainty_level_map_entry

    def __repr__(self):
        return str(self.marker_idx) + " " + str(self.doc[self.marker_idx])


ADVERBS_OF_CERTAINTY = [
    "definitly",
    "surely",
    "certainly",
    "probably",
    "perhaps",
    "likely",
    "possibly"
]

MODALITY_CERTAINTY_MARKER_DEPENDENCY_PATTERNS = [
    [
        {
            "RIGHT_ID": "certainty_marker",
            "RIGHT_ATTRS": {"TAG": "MD", "LEMMA": {"NOT_IN": ["will"]}}
        },
        {
            "LEFT_ID": "certainty_marker",
            "REL_OP": ".*",
            "RIGHT_ID": "affected_main_verb",
            "RIGHT_ATTRS": {"POS": "VERB"}
        }
    ],
    [
        {
            "RIGHT_ID": "certainty_marker",
            "RIGHT_ATTRS": {"POS": "ADV", "LOWER": {"IN": ADVERBS_OF_CERTAINTY}}
        },
        {
            "LEFT_ID": "certainty_marker",
            "REL_OP": ".*",
            "RIGHT_ID": "affected_main_verb",
            "RIGHT_ATTRS": {"POS": "VERB"}
        }
    ],
]

class CertaintySetter:
    '''
    sets a CertaintyInfo class on each Token that needs it on the extension ._.certainty_info
    sets extension if not set yet: 
        ._.certainty_info
        ._.certainty_marker_map
        ._.certainty_scope_map
    '''

    def __init__(self, vocab):
        self.vocab = vocab
        self.dep_matcher = DependencyMatcher(vocab, validate=True)
        self._set_needed_extensions()
        self._add_modality_setter_to_matcher()
    
    def _set_needed_extensions(self) -> None:
        if not Doc.has_extension("certainty_marker_map"):
            Doc.set_extension("certainty_marker_map", default=dict())
        if not Doc.has_extension("token_to_certainty_marker_map"):
            Doc.set_extension("token_to_certainty_marker_map", default=defaultdict(list))
        if not Token.has_extension("certainty_info"):
            Token.set_extension("certainty_info", default=None)
        if not Token.has_extension("certainty_level"):
            Token.set_extension("certainty_level", getter=_get_certainty_level)
        
    def _add_modality_setter_to_matcher(self) -> None:
        self.dep_matcher.add(
            "certainty_markers",
            MODALITY_CERTAINTY_MARKER_DEPENDENCY_PATTERNS,
            on_match=on_certainty_marker_match)


    def __call__(self, doc: Doc) -> None:
        self.dep_matcher(doc)
        return doc


def on_certainty_marker_match(matcher: DependencyMatcher, doc: Doc, i: int, matches) -> None:
    _, token_idxs = matches[i]
    certainty_marker_idx = token_idxs[0]
    affected_main_verb = doc[token_idxs[1]]
    scope = [t.i for t in affected_main_verb.subtree]
    for i in scope:
        certainty_info = CertaintyInfo(certainty_marker_idx, doc)
        doc[i]._.certainty_info = certainty_info
    for i in scope:
        doc._.token_to_certainty_marker_map[i].append(certainty_marker_idx)
    doc._.certainty_marker_map[certainty_marker_idx] = scope
    
def _get_certainty_level(token: Token) -> float:
    if not Token.has_extension("certainty_info"):
        logger.warning("Token needs the 'certainty_info' extension set to determine the context of the certainty_level for the given token")
        return None
    if token._.certainty_info is None:
        # we assume sentence doesnt have certainty_marker (modifier) and is taken as a fact
        return 1.0
    else:
        return token._.certainty_info.determine_level()
    

@Language.factory("certainty_setter")
def create_certainty_setter(nlp, name):
    return CertaintySetter(nlp.vocab)

class TenseTimeSetter:
    pass #TODO: how can i add this? what is the current state of linguistics regarding tenses and time?
