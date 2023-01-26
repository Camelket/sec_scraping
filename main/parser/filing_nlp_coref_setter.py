import spacy
from spacy.language import Language
import coreferee
from spacy.tokens import Token, Span, Doc
from filing_nlp_errors import ExtensionRequiredError, ComponentDependencyError

@Language.factory("coref_setter")
def create_coref_setter(nlp, name):
    return CorefSetter(nlp)

class CorefSetter:
    def __init__(self, nlp: Language):
        self.nlp = nlp
        self._ensure_required_components_are_present()
    
    def __call__(self, doc: Doc):
        self._ensure_alias_cache_initialized(doc)
        # get all mentions in chains as int tuples
        # compare the tuples to already covered references
        # extend references which cover only part of an entity to the whole entity and make note of such extension by type
        # only add new ones
        # add them to the alias cache if i can find a connection to an already existing reference chain
    
    def _ensure_required_components_are_present(self):
        if not self.nlp.has_pipe("coreferee"):
            raise ComponentDependencyError(f"Coreferee not registered on this pipeline. Make sure Coreferee is added before this component in the pipeline. https://github.com/explosion/coreferee")
        if not Token.has_extension("coref_chains"):
            raise ExtensionRequiredError(f"Token doesnt have extension 'coref_chains' set. make sure Coreferee is run before this Component in the pipeline.")
        if not Doc.has_extension("alias_cache"):
            raise ExtensionRequiredError(f"Doc Extension alias_cache isnt registered. Make sure AliasMatcher and AliasSetter are initalized in a previous Component.")
        
    def _ensure_alias_cache_initialized(self, doc: Doc):
        if doc._.alias_cache is None:
            raise ExtensionRequiredError(f"AliasCache on the Doc object wasnt initalized properly. Make sure AliasMatcher and AliasSetter are called in a previous Component and initalized the AliasCache properly.")
    
        
