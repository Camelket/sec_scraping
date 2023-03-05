from dataclasses import dataclass
from typing import Any
from spacy.tokens import Span, Token
from main.nlp.filing_nlp_dependency_matcher import (
    ContractDependencyAttributeMatcher,    
)
from main.nlp.filing_nlp_dateful_relations import DatetimeRelation

import logging
logger = logging.getLogger(__name__)


'''
Group LegalAgreement by unique identifier during creation, like secu_objects with the secu_key.
'''

class ContractObject:
    def __init__(self, original: Span, attr_matcher: ContractDependencyAttributeMatcher):
        self.original: Span = original
        self.scope: tuple[int, int] = None
        self.subjects: list[tuple[int, int]] = list()
        self.actions: list[tuple[int, int]] = list() #verbal phrases?
    


        

    



