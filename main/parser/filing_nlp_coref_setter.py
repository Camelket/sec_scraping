import spacy
from spacy.language import Language
import coreferee
from spacy.tokens import Token, Span, Doc
import logging
from main.parser.filing_nlp_errors import ExtensionRequiredError, ComponentDependencyError
from main.parser.filing_nlp_utils import extend_token_ent_to_span, create_single_token_span
logger = logging.getLogger(__name__)

@Language.factory("coref_setter")
def create_coref_setter(nlp, name):
    return CorefSetter(nlp)

class CorefSetter:
    def __init__(self, nlp: Language):
        self.nlp = nlp
        self._ensure_required_components_are_present()
    
    def __call__(self, doc: Doc):
        self._ensure_alias_cache_initialized(doc)
        for chain in doc._.coref_chains:
            logger.debug(("current chain: ", chain.pretty_representation))
            if chain:
                root_tuple = None
                if len(chain.mentions[0].token_indexes) == 1:
                    root_tuple = (chain.mentions[0].token_indexes[0], chain.mentions[0].token_indexes[0] + 1)
                    refs = []
                else:
                    root_tuple = (chain.mentions[0].token_indexes[0], chain.mentions[0].token_indexes[-1])
                    refs = chain.mentions[1:]
                if root_tuple is None:
                    logger.warning(f"Failed to fetch root_tuple of a coreferee.data_model.Chain, chain.mentions: {chain.mentions}")
                else:
                    if (root_tuple[0] in doc._.alias_cache._idx_alias_map) or (root_tuple[1] in doc._.alias_cache._idx_alias_map):
                        # if above check fails i should check other mentions in chain 
                        logger.info(f"corefchain root tuple linkable to already present alias")
                        parents_set = set([doc._.alias_cache._idx_alias_map.get(x) for x in range(root_tuple[0], root_tuple[1] + 1)])
                        logger.debug((parents_set, " ->parents_set"))
                        if len(parents_set) == 1:
                            parent_tuple = parents_set.pop()
                            parent_tuple_type = doc._.alias_cache.get_tuple_type(parent_tuple)
                            base_alias_tuple = None
                            if not parent_tuple_type:
                                raise ValueError(f"no type assigned in AliasCache to parent_tuple: {parent_tuple}. Make sure this tuple was added correctly.")
                            else:
                                if parent_tuple_type == "base_alias":
                                    base_alias_tuple = parent_tuple
                                else:
                                    base_alias_tuple = doc._.alias_cache.get_base_alias_tuple_by_reference_tuple(parent_tuple)
                            for mention in refs:
                                logger.debug((mention, " ->mention"))
                                # ref_token = doc[ref]
                                if any([doc._.alias_cache._idx_alias_map.get(x) for x in mention.token_indexes]):
                                    logger.debug(("ref is already assigned, Span of ref: ", [doc._.alias_cache._idx_alias_map.get(x) for x in mention.token_indexes]))
                                    # discard mention since it is already part of an alias
                                    continue
                                else:
                                    # new possible alias, extended if has ent label
                                    extended_to_span = extend_token_ent_to_span(mention.root_index)
                                    ref_tuple = None
                                    if extended_to_span:
                                        ref_tuple = (extended_to_span.start, extended_to_span.end)
                                    else:
                                        ref_tuple = (mention.root_index, mention.root_index+1)
                                    logger.debug((ref_tuple, " ->ref_tuple"))
                                    if ref_tuple:
                                        doc._.alias_cache.add_tuple_reference_to_tuple_base_alias(
                                            alias_tuple=base_alias_tuple,
                                            reference_tuple=ref_tuple,
                                            reference_designation="corefree_alias")
                                    else:
                                        raise ValueError(f"encountered ref_tuple being None for mention: {mention}")
                        # print(f"token_indexes of mentions: {[m.token_indexes for m in chain.mentions]} with already present ref: {doc._.alias_cache._idx_alias_map[root_tuple[0]]}")
                        # doc._.alias_cache._pretty_print_tree_from_origin(doc._.alias_cache._idx_alias_map.get(root_tuple[0]), doc)
        return doc

        # get all mentions in chains as int tuples
        # compare the tuples to already covered references
        # extend references which cover only part of an entity to the whole entity and make note of such extension by type
        # only add new ones
        # add them to the alias cache if i can find a connection to an already existing reference chain
    
    def is_valid_as_reference(self, parent: Span, reference: Span):
        pass
    
    def _ensure_required_components_are_present(self):
        if not self.nlp.has_pipe("coreferee"):
            #TODO["epic"=important] can and should I add coreferee from this component to the pipeline?
            raise ComponentDependencyError(f"Coreferee not registered on this pipeline. Make sure Coreferee is added before this component in the pipeline. https://github.com/explosion/coreferee")
        if not Token.has_extension("coref_chains"):
            raise ExtensionRequiredError(f"Token doesnt have extension 'coref_chains' set. make sure Coreferee is run before this Component in the pipeline.")
        if not Doc.has_extension("alias_cache"):
            raise ExtensionRequiredError(f"Doc Extension alias_cache isnt registered. Make sure AliasMatcher and AliasSetter are initalized in a previous Component.")
        
    def _ensure_alias_cache_initialized(self, doc: Doc):
        if doc._.alias_cache is None:
            raise ExtensionRequiredError(f"AliasCache on the Doc object wasnt initalized properly. Make sure AliasMatcher and AliasSetter are called in a previous Component and initalized the AliasCache properly.")
    
        
