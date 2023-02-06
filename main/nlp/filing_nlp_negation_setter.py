from spacy.tokens import Token, Doc
from spacy.matcher import Matcher
from spacy import Language
from main.nlp.filing_nlp_patterns import VERB_NEGATION_PATTERNS, ADJ_NEGATION_PATTERNS


class NegationSetter:
    """
    sets negation for ADJ and VERBS with the token extension ._.negated.
    """

    def __init__(self, vocab):
        self.vocab = vocab
        self.matcher = Matcher(vocab)
        self._set_needed_extensions()
        self.add_negation_ent_to_matcher()
    
    def _set_needed_extensions(self):
        if not Token.has_extension("negated"):
            Token.set_extension("negated", default=False)

    def add_negation_ent_to_matcher(self):
        self.matcher.add(
            "negation",
            [*VERB_NEGATION_PATTERNS, *ADJ_NEGATION_PATTERNS],
            on_match=_set_negation_extension,
        )

    def __call__(self, doc: Doc):
        self.matcher(doc)
        return doc


def _set_negation_extension(matcher: Matcher, doc: Doc, i: int, matches: list):
    """
    sets the negation extension for the matched tokens
    """
    match_id, start, end = matches[i]
    for token in doc[start:end]:
        if token.dep_ not in ["neg", "aux", "auxpass"]:
            token._.negated = True

@Language.factory("negation_setter")
def create_negation_setter(nlp, name):
    return NegationSetter(nlp.vocab)
