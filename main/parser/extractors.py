from abc import ABC
from functools import reduce
import logging
from types import NoneType, NotImplementedType
from typing import Dict, List, Optional

from pydantic import ValidationError
from datetime import datetime, timedelta
import pandas as pd
import re
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span, Token
import json
from collections import defaultdict

from main.parser.filings_base import Filing, FilingSection
from main.domain import model, commands
from main.domain.model import CommonShare, DebtSecurity, PreferredShare, Security, SecurityType, SecurityTypeFactory, Warrant, Option
from main.services.messagebus import Message, MessageBus
from .filing_nlp import SpacyFilingTextSearch, MatchFormater, get_secu_key, UnclearInformationExtraction

from .filing_nlp_SECU import SECU

logger = logging.getLogger(__name__)
security_type_factory = SecurityTypeFactory()

'''
the extractors are here to convert data from a Filing instance
into model objects and into command objects for the MessageBus
'''

class UnhandledClassificationError(Exception):
    pass

class ExtractedSecurities:
    def __init__(self, securities: list[model.Security]=None):
        self.securities = list() if securities is None else securities
        self.objects_map = defaultdict(list)
    
    def add_security(self, security: model.Security, associated_secu_objects: list[SECU]):
        if security not in self.securities:
            self.securities.append(security)
            self.objects_map[security] = associated_secu_objects
        else:
            logger.warning(f"model.Security {security} is already present on this ExtractedSecurities object.")
    
    def get_secu_objects(self, security: model.Security):
        return self.objects_map[security]


class AbstractFilingExtractor(ABC):
    def extract_form_values(self, filing: Filing, company: model.Company, bus: MessageBus):
        """extracts values and issues Command to MessageBus"""
        pass


class BaseHTMExtractor:
    def __init__(self):
        self.formater = MatchFormater()
        self.spacy_text_search = SpacyFilingTextSearch()
    
    def doc_from_section(self, section: FilingSection) -> Doc:
        return self.spacy_text_search.nlp(section.text_only)
    
    def _merge_attributes(self, d1: dict, d2: dict) -> Dict:
        # TODO: DEPRECATE shouldnt be needed after rework
        logger.debug(f"merging: {d1} and {d2}")
        if (d2 is None) or (d2 == {}):
            return d1
        for d2key in d2.keys():
            try:
                if (d1[d2key] is None) and (d2[d2key] is not None):
                    d1[d2key] = d2[d2key]
            except KeyError:
                d1[d2key] = d2[d2key]
        return d1 
    
    def get_secu_exercise_price(self, secu: SECU) -> float|None:
        if secu.exercise_price is not None:
            return secu.exercise_price.amount
        return None
    
    def get_secu_expiry(self, secu: SECU):
        if secu.expiry_date is not None:
            return secu.expiry_date
        return None
    
    def get_secu_multiplier(self, secu: SECU):
        # TODO: implement in SECU objects
        logger.debug("get_secu_multiplier returning dummy value: 1")
        return 1

    def get_secu_latest_issue_date(self, secu: SECU):
        # TODO: implement in SECU objects first
        logger.debug("get_secu_latest_issue_date returning dummy value: None")
        return None

    def get_secu_interest_rate(self, secu: SECU):
        # TODO: implement in SECU objects first
        logger.debug("get_secu_interest_rate returning dummy value: 0.05")
        return 0.05

    def get_secu_right(self, secu: SECU):
        # TODO: implement in SECU objects first
        logger.debug("get_secu_right returning dummy value: 'Call'")
        return "Call"

    def get_secu_conversion(self, doc: Doc, secu: SECU):
        # TODO: implement in SECU objects first
        logger.debug("get_secu_conversion not implemented")
        pass
    
    def get_security_attributes(self, secu: SECU, security_type: Security) -> Dict:
        # TODO: add convertible Feature when implemented  
        attributes = {}
        if security_type == CommonShare:
            attributes = {
                "name": secu.secu_key
                }
        elif security_type == PreferredShare:
            attributes = {
                "name": secu.secu_key
                }
        elif security_type == Warrant:
            attributes = {
                "name": secu.secu_key,
                "exercise_price": self.get_secu_exercise_price(secu),
                "expiry_date": self.get_secu_expiry(secu),
                "right": self.get_secu_right(secu),
                "multiplier": self.get_secu_multiplier(secu),
                "issue_date": self.get_secu_latest_issue_date(secu)
                }
        elif security_type == Option:
            attributes = {
                "name": secu.secu_key,
                "strike_price": self.get_secu_exercise_price(secu),
                "expiry": self.get_secu_expiry(secu),
                "right": self.get_secu_right(secu),
                "multiplier": self.get_secu_multiplier(secu),
                "issue_date": self.get_secu_latest_issue_date(secu)
            }
        elif security_type == DebtSecurity:
            attributes = {
                "name": secu.secu_key,
                "interest_rate": self.get_secu_interest_rate(secu),
                "maturity": self.get_secu_expiry(secu),
                "issue_date": self.get_secu_latest_issue_date(secu)
            }
        logger.info(f"attributes: {attributes} of security: {secu.secu_key, secu.original.i} with type: {security_type}")
        return attributes
    
    def get_securities_from_docs(self, docs: List[Doc]) -> ExtractedSecurities:
        '''
        create model.Security instances from SECU objects found on the Doc and
        associate them through a model.Security->SECU map (present as objects_map on ExtractedSecurities)
        '''
        for doc in docs:
            if not doc.has_extension("secu_objects"):
                raise AttributeError(f"doc object is missing extension attribute")
        extracted = ExtractedSecurities()
        secu_keys = set([key for key, _ in doc._.secu_objects.items() for doc in docs])
        for key in secu_keys:
            # i work under the assumption that a secu_key is a unique identifier in a doc for a given security
            associated_secu_objects = []
            security_attributes = {}
            security_type = security_type_factory.get_security_type(key)
            for doc in docs:
                secus = doc._.secu_objects.get(key, None)
                if secus:
                    for secu in secus:
                        security_attributes = self._merge_attributes(
                            security_attributes,
                            self.get_security_attributes(secu, security_type)
                        )
                        associated_secu_objects.append(secu)
            try: 
                security = model.Security(security_type(**security_attributes))
            except ValidationError:
                logger.info(f"Failed to construct model.Security from args: security_type={security_type}, security_attributes={security_attributes}")
            else:
                extracted.add_security(security, associated_secu_objects)
        return extracted
    
    def handle_extracted_securities_through_bus(self, bus: MessageBus, filing: Filing, company: model.Company, securities: list[model.Security]):
        bus.handle(commands.AddSecurities(company.cik, company.symbol, securities))
        for security in securities:
            search_attributes = self.get_default_search_attributes_for_security(security)
            if search_attributes:
                bus.handle(commands.AddSecurityAccnOccurence(
                    cik=company.cik,
                    symbol=company.symbol,
                    security_attributes=search_attributes,
                    accn=filing.accession_number
                ))
            else:
                logger.warning(f"couldnt add security occurence, because required search_attributes werent present. security_attributes: {security.security_attributes}")
        

    def handle_outstanding_security_facts_through_bus(self, bus: MessageBus, filing: Filing, company: model.Company, extracted: ExtractedSecurities):
        for security in extracted.securities:
            secu_objects = extracted.get_secu_objects(security)
            if secu_objects != []:
                outstanding_facts = []
                for secu in secu_objects:
                    outstanding_relations = secu.get_outstanding_quantity_relations()
                    if outstanding_relations:
                        for relation in outstanding_relations:
                            fact = model.SecurityOutstanding(
                                relation.quantity.amount.amount,
                                relation.quantity.datetime_relation.timestamp
                            )
                            outstanding_facts.append(fact)
            if outstanding_facts != []:
                security_search_attributes = self.get_default_search_attributes_for_security(security)
                cmd = commands.AddOutstandingSecurityFact(
                    cik=company.cik,
                    symbol=company.symbol,
                    attributes=security_search_attributes,
                    outstanding=outstanding_facts
                )
                bus.handle(cmd)
    
    def default_security_pipeline(self, bus: MessageBus, filing: Filing, company: model.Company, docs: list[Doc]|None):
        # add a default for usually wanted things (securities, outstanding, ....)
        extracted_securities = self.get_securities_from_docs(docs)
        self.handle_extracted_securities_through_bus(
            bus,filing, company, extracted_securities.securities)
        self.handle_outstanding_security_facts_through_bus(
            bus, filing, company, extracted_securities
        )
        # TODO: update this if we add new facts
    
    def get_default_search_attributes_for_security(self, security: model.Security) -> dict|None:
        if not isinstance(security, model.Security):
            raise TypeError(f"{self} Expecting model.Security got type: {type(security)}")
        if security.security_type == "CommonShare":
            return {"name": security.name}
        elif security.security_type == "PreferredShare":
            return {"name": security.name}
        else:
            attrs = json.loads(security.security_attributes)
            wanted_attributes = {}
            if security.security_type == "Warrant":
                uniqueness_attributes = ["expiry_date", "exercise_price"]
                for each in uniqueness_attributes:
                    attr = attrs.get(each, None)
                    if attr:
                        wanted_attributes[each] = attr
                logger.warning(f"wanted_attributes: {wanted_attributes}")
                if len(wanted_attributes.keys()) >= len(uniqueness_attributes)-1:
                    wanted_attributes["name"] = security.name
                    return wanted_attributes
            elif security.security_type == "Option":
                uniqueness_attributes = ["strike_price", "expiry", "issue_date"]
                for each in uniqueness_attributes:
                    attr = attrs.get(each, None)
                    if attr:
                        wanted_attributes[each] = attr
                if len(wanted_attributes.keys()) >= len(uniqueness_attributes)-1:
                    wanted_attributes["name"] = security.name
                    return wanted_attributes
            elif security.security_type == "DebtSecurity":
                uniqueness_attributes = ["interest_rate", "maturity", "issue_date"]
                for each in uniqueness_attributes:
                    attr = attrs.get(each, None)
                    if attr:
                        wanted_attributes[each] = attr
                if len(wanted_attributes.keys()) >= len(uniqueness_attributes)-1 and "interest_rate" in wanted_attributes.keys():
                    wanted_attributes["name"] = security.name
                    return wanted_attributes
        return None

# class BaseHTMExtractor2:
#     def __init__(self):
#         self.spacy_text_search = SpacyFilingTextSearch()
#         self.formater = MatchFormater()
    
#     def get_secu_key(self, security: Span|str) -> str:
#         if isinstance(security, str):
#             doc = self.spacy_text_search.nlp(security)
#             security = doc[0:]
#         if isinstance(security, Span):
#             return get_secu_key(security)
#         else:
#             raise TypeError(f"BaseHTMExtractor.get_secu_key is expecting type:Span got:{type(security)}")
    
#     def get_mentioned_secus(self, doc: Doc, secus: Optional[Dict]=None) -> Dict:
#         #TODO: check if this can be DEPRECATED
#         if secus is None:
#             secus = dict()
#         single_secu_alias_tuples = doc._.single_secu_alias_tuples
#         for key in single_secu_alias_tuples.keys():
#             if key not in secus.keys():
#                 secus[key] = []
#             secus[key] += single_secu_alias_tuples[key]["no_alias"]
#             for alias in single_secu_alias_tuples[key]["alias"]:
#                 alias_span = alias[1]
#                 alias_key = get_secu_key(alias_span)
#                 if alias_key:
#                     if alias_key not in secus.keys():
#                         secus[alias_key] = []
#                     similar_spans = self.spacy_text_search.get_queryable_similar_spans_from_lower(doc, alias_span)
#                     if similar_spans:
#                         secus[alias_key] += similar_spans
#         return secus
    
#     def get_security_type(self, security_name: str) -> SecurityType:
#         return security_type_factory.get_security_type(security_name)
    
#     def handle_securities_through_bus(self, bus: MessageBus, filing: Filing, company: model.Company, securities: list[model.Security]):
#         bus.handle(commands.AddSecurities(company.cik, company.symbol, securities))
#         for security in securities:
#             search_attributes = self.get_default_search_attributes_for_security(security)
#             if search_attributes:
#                 bus.handle(commands.AddSecurityAccnOccurence(
#                     cik=company.cik,
#                     symbol=company.symbol,
#                     security_attributes=search_attributes,
#                     accn=filing.accession_number
#                 ))
#             else:
#                 logger.warning(f"couldnt add security occurence, because required search_attributes werent present. security_attributes: {security.security_attributes}")



#     def merge_attributes(self, d1: dict, d2: dict) -> Dict:
#         logger.debug(f"merging: {d1} and {d2}")
#         if (d2 is None) or (d2 == {}):
#             return d1
#         for d2key in d2.keys():
#             try:
#                 if (d1[d2key] is None) and (d2[d2key] is not None):
#                     d1[d2key] = d2[d2key]
#             except KeyError:
#                 d1[d2key] = d2[d2key]
#         return d1 
    
#     def get_default_search_attributes_for_security(self, security: model.Security) -> dict|None:
#         if not isinstance(security, model.Security):
#             raise TypeError(f"{self} Expecting model.Security got type: {type(security)}")
#         if security.security_type == "CommonShare":
#             return {"name": security.name}
#         elif security.security_type == "PreferredShare":
#             return {"name": security.name}
#         else:
#             attrs = json.loads(security.security_attributes)
#             wanted_attributes = {}
#             if security.security_type == "Warrant":
#                 uniqueness_attributes = ["expiry_date", "exercise_price"]
#                 for each in uniqueness_attributes:
#                     attr = attrs.get(each, None)
#                     if attr:
#                         wanted_attributes[each] = attr
#                 if len(wanted_attributes.keys()-1) >= len(uniqueness_attributes):
#                     wanted_attributes["name"] = security.name
#                     return wanted_attributes
#             elif security.security_type == "Option":
#                 uniqueness_attributes = ["strike_price", "expiry", "issue_date"]
#                 for each in uniqueness_attributes:
#                     attr = attrs.get(each, None)
#                     if attr:
#                         wanted_attributes[each] = attr
#                 if len(wanted_attributes.keys()-1) >= len(uniqueness_attributes):
#                     wanted_attributes["name"] = security.name
#                     return wanted_attributes
#             elif security.security_type == "DebtSecurity":
#                 uniqueness_attributes = ["interest_rate", "maturity", "issue_date"]
#                 for each in uniqueness_attributes:
#                     attr = attrs.get(each, None)
#                     if attr:
#                         wanted_attributes[each] = attr
#                 if len(wanted_attributes.keys()-1) >= len(uniqueness_attributes) and "interest_rate" in wanted_attributes.keys():
#                     wanted_attributes["name"] = security.name
#                     return wanted_attributes
#         return None
    
#     def get_security_attributes(self, doc: Doc, security_name: str, security_type: Security, security_spans) -> Dict:  
#         #TODO: replace with function: check_security_has_inital_required_security_type_attributes (think of better name)
#         attributes = {}
#         _kwargs = {"doc": doc, "security_name": security_name, "security_spans": security_spans}
#         if security_type == CommonShare:
#             attributes = {
#                 "name": security_name
#                 }
#         elif security_type == PreferredShare:
#             attributes = {
#                 "name": security_name
#                 }
#         elif security_type == Warrant:
#             attributes = {
#                 "name": security_name,
#                 "exercise_price": self.get_secu_exercise_price(**_kwargs),
#                 "expiry_date": self.get_secu_expiry_by_type(wanted="datetime", **_kwargs),
#                 "expiry_timedelta": self.get_secu_expiry_by_type(wanted="timedelta", **_kwargs),
#                 "right": self.get_secu_right(**_kwargs),
#                 "multiplier": self.get_secu_multiplier(**_kwargs),
#                 "issue_date": self.get_secu_latest_issue_date(**_kwargs)
#                 }
#         elif security_type == Option:
#             attributes = {
#                 "name": security_name,
#                 "strike_price": self.get_secu_exercise_price(**_kwargs),
#                 "expiry": self.get_secu_expiry(**_kwargs),
#                 "right": self.get_secu_right(**_kwargs),
#                 "multiplier": self.get_secu_multiplier(**_kwargs),
#                 "issue_date": self.get_secu_latest_issue_date(**_kwargs)
#             }
#         elif security_type == DebtSecurity:
#             attributes = {
#                 "name": security_name,
#                 "interest_rate": self.get_secu_interest_rate(**_kwargs),
#                 "maturity": self.get_secu_expiry(**_kwargs),
#                 "issue_date": self.get_secu_latest_issue_date(**_kwargs)
#             }
#         logger.info(f"attributes: {attributes} of security: {security_name} with type: {security_type}")
#         return attributes
    
#     def get_securities_from_docs(self, docs: List[Doc]) -> List[model.Security]:
#         # TODO: rewrite this to use new SECU class 
#         # 1) get SECU objects
#         # 2) get security type for given secu key
#         # 3) check the securities with given key against requirements for given security type
#         # 4) create the model and append to securities list
#         securities = []
#         mentioned_secus = {}
#         for doc in docs:
#             mentioned_secus = self.get_mentioned_secus(doc, mentioned_secus)
#         logger.debug(f"finished collecting mentioned_secus: {mentioned_secus}")
#         for secu, secu_spans in mentioned_secus.items():
#             security_type = self.get_security_type(secu)
#             logger.debug(f"getting security_attributes for {secu} with type: {security_type}")
#             security_attributes = {}
#             for doc in docs:
#                 security_attributes = self.merge_attributes(
#                     security_attributes,
#                     self.get_security_attributes(doc, secu, security_type, secu_spans))
#             try:
#                 security = model.Security(security_type(**security_attributes))
#             except ValidationError:
#                 logger.info(f"Failed to construct model.Security from args: security_type={security_type}, security_attributes={security_attributes}")
#             else:
#                 securities.append(security)
#         return securities

#     def get_secu_exercise_price(self, doc: Doc, security_name: str, security_spans: List[Span]) -> str|NoneType:
#         # DEPRECATE shouldnt be needed since we have the exercise price within the SECU object
#         exercise_prices_seen = set()
#         exercise_prices = []
#         logger.debug("get_secu_exercise_price:")
#         logger.debug(f" getting exercise_price for:")
#         logger.debug(f"     security_name  - {security_name}")
#         logger.debug(f"     security_spans - {security_spans}")
#         for secu in security_spans:
#             for sent in doc.sents:
#                 if self._is_span_in_sent(sent, secu):
#                     temp_doc = doc[sent.start:sent.end]
#                     exercies_price = self.spacy_text_search.match_secu_exercise_price(temp_doc, secu)
#                     if exercies_price:
#                         for price in exercies_price:
#                             if price not in exercise_prices_seen:
#                                 exercise_prices_seen.add(price)
#                                 exercise_prices.append(price)
#                                 logger.debug(f"     exercise price found: {price}")
#                                 logger.debug(f"     sentence: {temp_doc}")
#         if len(exercise_prices) > 1:
#             logger.info(UnclearInformationExtraction(f"unhandled case of exercise_price extraction, got more than one exercise_price for a security: {exercise_prices}"), exc_info=True)
#             return None
#         return exercise_prices[0] if exercise_prices != [] else None
    
#     def _is_span_in_sent(self, sent: Span, span: Span|Token) -> bool:
#         token_idx_offset = sent[0].i
#         if isinstance(span, Span):
#             span_length = len(span)
#             first_token = span[0]
#             for token in sent:
#                 if token == first_token:
#                     first_idx = token.i - token_idx_offset
#                     try:
#                         if sent[first_idx:first_idx+span_length] == span:
#                             if self._span_neighbors_arent_SECU(sent, span):
#                                 return True
#                     except IndexError:
#                         logger.debug("excepted IndexError in _assert_any_secu_span_is_in_match; passing.")
#                         pass
#         if isinstance(span, Token):
#             if self._is_single_token_SECU_in_sent(sent, span):
#                 return True
#         return False
                
    
#     def _is_single_token_SECU_in_sent(self, sent: Span, token: Token) -> bool:
#         if token in sent:
#             if self._span_neighbors_arent_SECU(sent, token):
#                 return True
#         return False
    
#     def _span_neighbors_arent_SECU(self, sent: Span, span: Span|Token) -> bool:
#         token_idx_offset = sent[0].i
#         if isinstance(span, Token):
#             start = span.i
#             end = start
#         else:
#             start = span[0].i
#             end = span[-1].i
#         start = start - token_idx_offset
#         end = end - token_idx_offset
#         if start != 0 and end != sent[-1].i:
#             if sent[start-1].ent_type_ != "SECU" and sent[end+1].ent_type_ != "SECU":
#                 return True
#         else:
#             if start != 0:
#                 if sent[start-1].ent_type_ != "SECU":
#                     return True
#             else:
#                 if sent[end+1].ent_type_ != "SECU":
#                     return True
#         return False
    
#     def get_secu_expiry_by_type(self, wanted: str, doc: Doc, security_name: str, security_spans: List[Span]) -> str|NoneType:
#         # TODO: DEPRECATE shouldnt be needed after rework
#         '''
#         handle the case for the expiry as a timedelta from the issuance date or as a specific date in time.

#         Args:
#             wanted: 
#                 "timedelta" to get only timedelta returns
#                 "datetime" to get only date returns
        
#         Returns:
#             timedelta, datetime or None
#         '''
#         expiry = self.get_secu_expiry(doc=doc, security_name=security_name, security_spans=security_spans)
#         if expiry:
#             if isinstance(expiry, timedelta) and wanted == "timedelta":
#                 return expiry
#             if isinstance(expiry, datetime) and wanted == "datetime":
#                 return expiry
#         return None
    
#     def get_secu_expiry(self, doc: Doc, security_name: str, security_spans: List[Span]) -> str|NoneType:
#         # TODO: DEPREACTE since we have the expiry within the SECU objects
#         expiries_seen = set()
#         expiries = []
#         logger.debug("get_secu_expiry:")
#         logger.debug(f" getting expiry for:")
#         logger.debug(f"     security_name  - {security_name}")
#         logger.debug(f"     security_spans - {security_spans}")
#         for secu in security_spans:
#             for sent in doc.sents:
#                 if self._is_span_in_sent(sent, secu):
#                     temp_doc = doc[sent.start:sent.end]
#                     expiry = self.spacy_text_search.match_secu_expiry(temp_doc, secu)
#                     if expiry:
#                         for ex in expiry:
#                             logger.info(ex)
#                             if ex not in expiries_seen and ex is not None:
#                                 expiries_seen.add(ex)
#                                 expiries.append(ex)
#         if len(expiries) > 1:
#             raise UnclearInformationExtraction(f"Couldnt get a definitive match for the expiry, got multiple: {expiries}")
#         else:
#             return expiries[0] if expiries != [] else None

#     def get_secu_multiplier(self, doc: Doc, security_name: str, security_spans: List[Span]):
#         # TODO: implement
#         logger.debug("get_secu_multiplier returning dummy value: 1")
#         return 1

#     def get_secu_latest_issue_date(self, doc: Doc, security_name: str, security_spans: List[Span]):
#         # TODO: implement to use SECU objects
#         logger.debug("get_secu_latest_issue_date returning dummy value: None")
#         return None

#     def get_secu_interest_rate(self, doc: Doc, security_name: str, security_spans: List[Span]):
#         # TODO: implement to use SECU objects
#         logger.debug("get_secu_interest_rate returning dummy value: 0.05")
#         return 0.05

#     def get_secu_right(self, doc: Doc, security_name: str, security_spans: List[Span]):
#         # TODO: implement to use SECU objects
#         logger.debug("get_secu_right returning dummy value: 'Call'")
#         return "Call"

#     def get_secu_conversion(self, doc: Doc, security_name: str, security_spans: List[Span]):
#         # TODO: implement to use SECU objects
#         logger.debug("get_secu_conversion not implemented")
#         pass
    
#     def doc_from_section(self, section: FilingSection) -> Doc:
#         # TODO: implement to use SECU objects
#         return self.spacy_text_search.nlp(section.text_only)

#     def extract_outstanding_shares(self, filing: Filing) -> list[model.SecurityOutstanding]|NoneType:
#         # TODO: adjust to use SECU objects
#         text = filing.get_text_only()
#         if text is None:
#             logger.debug(filing)
#             return None
#         values = self.spacy_text_search.match_outstanding_shares(text)
#         return [model.SecurityOutstanding(v["amount"], v["date"]) for v in values]

class HTMS1Extractor(BaseHTMExtractor, AbstractFilingExtractor):
    # TODO: rework with SECU objects in mind
    def extract_form_values(self, filing: Filing):
        return [self.extract_outstanding_shares(filing)]

class HTMS3Extractor(BaseHTMExtractor, AbstractFilingExtractor):
    def extract_form_values(self, filing: Filing, company: model.Company, bus: MessageBus):
        complete_doc = self.spacy_text_search.nlp(filing.get_text_only())
        self.default_security_pipeline(bus, filing, company, [complete_doc])
        try:
            form_case = self.classify_s3(filing)
        except AttributeError as e:
            logger.error(f"excepted AttributeError in classify_s3 for filing({filing.path}) e: {e}", exc_info=True)
            return company
        cover_page_doc = self.doc_from_section(filing.get_section(re.compile("cover page")))
        # self.extract_securities(filing, company, bus, cover_page_doc)
        form_classifications = form_case["classifications"]
        if "shelf" in form_classifications:
            logger.info(f"Handling classification case: 'shelf' in form_case: {form_case}")
            # add shelf_registration
            self.handle_shelf(filing, company, bus, is_preliminary=form_case["is_preliminary"])
        if "resale" in form_classifications:
            logger.info(f"Handling classification case: 'resale' in form_case: {form_case}")
            # add resale_registration
            self.handle_resale(filing, company, bus, is_preliminary=form_case["is_preliminary"])
        if "ATM" in form_classifications:
            logger.info(f"Handling classification case: 'ATM' in form_case: {form_case}")
            # add ShelfOffering to BaseProspectus
            # check if we have ShelfRegistration added
            self.handle_ATM(filing, company, bus, cover_page_doc, is_preliminary=form_case["is_preliminary"])
        return company

    def handle_ATM(self, filing: Filing, company: model.Company, bus: MessageBus, cover_page_doc: Doc, is_preliminary: bool=False):
        shelf: model.ShelfRegistration = company.get_shelf(file_number=filing.file_number)
        print(f"found base registration 'shelf': {shelf}")
        if shelf:
            commencment_date = filing.filing_date #write function for this, for now assume filing_date
            kwargs = {
                    "offering_type": "ATM",
                    "accn": filing.accession_number,
                    "anticipated_offering_amount": self.extract_aggregate_offering_amount(cover_page_doc).get("amount"),
                    "commencment_date": commencment_date,
                    "end_date": commencment_date + timedelta(days=1095)
                }
            offering = model.ShelfOffering(**kwargs)
            bus.handle(commands.AddShelfOffering(company.cik, company.symbol, offering))
            # shelf.add_offering(offering)
            self.handle_ATM_security_registrations(filing, company, bus, cover_page_doc)

    def handle_ATM_security_registrations(self, filing: Filing, company: model.Company, bus: MessageBus, cover_page_doc: Doc):
        security_registrations = self.get_ATM_security_registrations(filing, company, cover_page_doc)
        for security_registration in security_registrations:
            self.handle_shelf_security_registration(
                    filing=filing,
                    company=company,
                    bus=bus,
                    security_registration=security_registration)
                
            

    def handle_resale(self, filing: Filing, company: model.Company, bus: MessageBus, is_preliminary: bool=False):
        kwargs =  {
                "accn": filing.accession_number,
                "form_type": filing.form_type,
                "file_number": filing.file_number,
                "filing_date": filing.filing_date
            }
        resale = model.ResaleRegistration(**kwargs)
        bus.handle(commands.AddResaleRegistration(filing.cik, company.symbol, resale))
        self.handle_resale_security_registrations(filing, company, bus)
        # if no registrations can be found add whole dollar amount as common
        # add ShelfOffering
        # get underwriters, registrations and completed then modify previously added ShelfOffering

    def handle_resale_security_registrations(self, filing: Filing, company: model.Company, bus: MessageBus):
        security_registrations = self.get_resale_security_registrations(filing, company)
    
    def get_resale_security_registrations(self, filing: Filing, company: model.Company):
        # get secus + secuquantity from cover_page and find a smart way to share secus with other docs
        pass

    def handle_shelf(self, filing: Filing, company: model.Company, bus: MessageBus, is_preliminary: bool=False):
        capacity = self.format_total_shelf_capacity(self.extract_shelf_capacity(filing))
        if not capacity:
            logger.error(f"couldnt find shelf capacity for filing: {filing.path}")
            return
        kwargs = {
                "accn": filing.accession_number,
                "file_number": filing.file_number,
                "form_type": filing.form_type,
                "capacity": capacity,
                "filing_date": filing.filing_date
            }
        shelf = model.ShelfRegistration(**kwargs)
        bus.handle(commands.AddShelfRegistration(filing.cik, company.symbol, shelf))                
    
    def handle_shelf_security_registration(self, filing: Filing, company: model.Company, bus: MessageBus, security_registration: model.ShelfSecurityRegistration):
        offering = company.get_shelf_offering(filing.accession_number)
        cmd = commands.AddShelfSecurityRegistration(filing.cik, company.symbol, filing.accession_number, security_registration=security_registration)
        bus.handle(cmd)
        logger.warning(f"{self} Not fully implemented yet")
    
    def get_ATM_security_registrations(self, filing: Filing, company: model.Company, cover_page: Doc):
        registrations = []
        aggregate_offering_amount = self.extract_aggregate_offering_amount(cover_page)
        if aggregate_offering_amount:
            if len(aggregate_offering_amount["SECU"]) == 1:
                secu = company.get_security_by_name(aggregate_offering_amount["SECU"][0])
                amount = aggregate_offering_amount["amount"]
                registrations.append(model.ShelfSecurityRegistration(secu, amount, None, None))
        return registrations if registrations != [] else None


    def extract_securities(self, filing: Filing, company: model.Company, bus: MessageBus, security_doc: Doc) -> List[model.Security]:
        # TODO: OBSOLETE with addition of default_securities_pipeline in BaseHTMExtractor
        '''extract securities from security_doc.'''
        extracted_securities = self.get_securities_from_docs([security_doc])
        self.handle_extracted_securities_through_bus(bus, filing, company, extracted_securities.securities)
        return extracted_securities.securities
    
    def extract_aggregate_offering_amount(self, cover_page: Doc) -> dict:
        matches = self.spacy_text_search.match_aggregate_offering_amount(cover_page)
        # check how well this works
        if len(matches) > 1:
            logger.debug(f"Unhandled case for extract_offering_amount: more than one match. matches: {matches}")
            raise NotImplementedError
        else:
            offering_data = {"SECU": [], "amount": None}
            for ent in matches[0].ents:
                if ent.label_ == "SECU":
                    offering_data["SECU"].append(get_secu_key(ent))
                if ent.label_ == "MONEY":
                    offering_data["amount"] = self.formater.quantity_string_to_float(ent.text)
            return offering_data if (offering_data["amount"] is not None) else None
 
    def extract_securities_conversion_attributes(self, filing: Filing, company: model.Company, bus: MessageBus) -> List[model.SecurityConversion]:
        logger.warning(f"{self} NOT IMPLEMENTED")
        description_sections = filing.get_sections(re.compile(r"description\s*of", re.I))
        description_docs = [self.doc_from_section(x) for x in description_sections]
        conversions = []
        for security in company.securities:
            for doc in description_docs:
                conversion_attr = self.get_secu_conversion(doc, security.name)
                if conversion_attr:
                    # add conversion attribute to Securities
                    pass # need to write get_secu_conversion first
        return conversions


    def classify_s3(self, filing: Filing):
        front_page = filing.get_section(re.compile("front page"))
        cover_page = filing.get_section(re.compile("cover page"))
        distribution = filing.get_section(re.compile("distribution", re.I))
        summary = filing.get_section(re.compile("summary", re.I))
        about = filing.get_section(re.compile(r"about\s*this", re.I))
        if cover_page == []:
            raise AttributeError(f"couldnt get the cover page section; sections present: {[s.title for s in filing.sections]}")
        cover_page_doc = self.doc_from_section(cover_page)
        distribution_doc = self.doc_from_section(distribution) if distribution != [] else []
        summary_doc = self.doc_from_section(summary) if summary != [] else []
        about_doc = self.doc_from_section(about) if about != [] else []
        form_case = {"is_preliminary": False, "is_combination_form_case": False, "classifications": set()}
        if front_page:
            front_page_doc = self.doc_from_section(front_page)
            if self._is_preliminary_prospectus(front_page_doc):
                form_case["is_preliminary"] = True
        if self._is_resale_prospectus(cover_page_doc):
            form_case["classifications"].add("resale")
        if self._is_at_the_market_prospectus(cover_page_doc) or self._is_at_the_market_prospectus(distribution_doc):
            form_case["classifications"].add("ATM")
        if True in [self._is_shelf_prospectus(x) if x != [] else False for x in [cover_page_doc, about_doc, summary_doc]]:
            form_case["classifications"].add("shelf")
        if len(form_case["classifications"]) > 1:
            form_case["is_combination_form_case"] = True
        if form_case["classifications"] == []:
            raise UnhandledClassificationError
        return form_case
    
    def _is_preliminary_prospectus(self, doc: Doc):
        phrase_matcher = PhraseMatcher(self.spacy_text_search.nlp.vocab, attr="LOWER")
        pm_patterns = [self.spacy_text_search.nlp.make_doc(term) for term in [
            "SUBJECT TO COMPLETION,",
            "The information in this prospectus is not complete and may be changed.",
            "The information set forth in this preliminary prospectus is not complete and may be changed.",
            "The information set forth in this prospectus is not complete and may be changed."
        ]]
        phrase_matcher.add("is_preliminary_prospectus", pm_patterns)
        if len(phrase_matcher(doc)) > 0:
            return True
        else:
            return False

    def _is_resale_prospectus(self, doc: Doc) -> bool:
        '''
        determine if this is a resale of securities by anyone other than the registrar,
        by checking for key phrases in the "cover page".'''
        matcher = Matcher(self.spacy_text_search.nlp.vocab)
        phrase_matcher = PhraseMatcher(self.spacy_text_search.nlp.vocab, attr="LOWER")
        # action_verbs = ["sell", "offer", "resell", "disposition"]
        m_pattern1 = [
            {"LOWER": "prospectus"},
            {"LEMMA": "relate"},
            {"LOWER": "to"},
            {"OP": "*", "IS_SENT_START": False},
            {"LOWER": "by"},
            {"LOWER": "the"},
            {"LOWER": "selling"},
            {"LOWER": {"IN": ["stockholder", "stockholders"]}}
        ]
        pm_patterns = [self.spacy_text_search.nlp.make_doc(term) for term in [
            "“Selling Stockholder”",
            "“Selling Stockholders”",
            "Selling Stockholder will receive all of the net proceeds"
            "Selling Stockholders will receive all of the net proceeds"
        ]]
        phrase_matcher.add("selling_stockholder", pm_patterns)
        matcher.add("ATM", [m_pattern1])
        possible_phrase_matches = phrase_matcher(doc, as_spans=True)
        possible_matches = matcher(doc, as_spans=True)
        if len(possible_matches) > 0:
            logger.debug(f"This is a 'resale' Prospectus, determined by matches: {possible_matches}")
            return True
        if len(possible_phrase_matches) > 0:
            logger.debug(f"This is a 'resale' Prospectus, determined by phrase_matches: {possible_phrase_matches}")
            return True
        return False


    def _is_at_the_market_prospectus(self, doc: Doc) -> bool:
        if not isinstance(doc, Doc):
            logger.debug(f"_is_at_the_market_prospectus failed; Expecting type: Doc got type: {type(doc), doc}")
            return False
        '''
        Determine if this is an "at-the-market" offering, rule 415(a)(4) Act 1933,
        by checking for key phrase in the "cover page" and "plan of distribution".
        
        '''
        matcher = Matcher(self.spacy_text_search.nlp.vocab)
        at_the_market_cases = [
            ["at", "-", "the", "-", "market"],
            ["at", "the", "market", "offering"],
            ["at", "the", "market", "offerings"],
            ["at", "the", "market"]
        ]
        pattern1 = [
            [
            {"LEMMA": {"IN": ["sale", "Sale", "sell"]}},
            {"OP": "*", "IS_SENT_START": False},
            {"LOWER": "under"},
            {"LOWER": "this"},
            {"LOWER": "prospectus"},
            {"OP": "*", "IS_SENT_START": False},
            {"LEMMA": "be"},
            {"LEMMA": "make"},
            {"OP": "*", "IS_SENT_START": False},
            {"LOWER": "deemed"},
            {"LOWER": "to"},
            {"LOWER": "be"},
            {"LOWER": {"IN": ["an", "a"]}, "OP": "?"},
            {"IS_PUNCT": True, "IS_SENT_START": False},
            *[{"LOWER": x} for x in at_the_market_case],
            {"IS_PUNCT": True, "IS_SENT_START": False},
            {"OP": "*", "IS_SENT_START": False},
            {"LOWER": "as"},
            {"LEMMA": "define"},
            {"LOWER": "in"},
            {"_": {"sec_act": True}, "TEXT": {"REGEX": "415"}}]
            for at_the_market_case in at_the_market_cases
            # need to add label LAW or SECACT for rules from 1933/1934
            # add rule 415 cases then create dicts from at_the-market_case list

        ]
        matcher.add("ATM", [*pattern1])
        possible_matches = matcher(doc, as_spans=True)
        if len(possible_matches) > 0:
            logger.debug(f"This is an 'ATM' (at-the-market) Prospectus, determined by matches: {possible_matches}")
            return True
        return False


    def _is_shelf_prospectus(self, doc: Doc) -> bool:
        '''
        Determine if this is a base prospectus
        by checking for key phrases in the "cover page", "about this" or "summary".
        '''
        matcher = Matcher(self.spacy_text_search.nlp.vocab)
        pattern1 = [
            {"IS_SENT_START": True},
            {"OP": "*", "IS_SENT_START": False},
            {"LOWER": "not"},
            {"LEMMA": "be"},
            {"LEMMA": "use"},
            {"ORTH": "to"},
            {"LEMMA": {"IN": ["offer", "consummate", "sell"]}},
            {"OP": "*", "IS_SENT_START": False},
            {"LOWER": {"IN": ["any", "these"]}},
            {"LOWER": {"IN": ["securities", "security"]}},
            {"LOWER": "unless"},
            {"OP": "*", "IS_SENT_START": False},
            {"LEMMA": "accompany"},
            {"OP": "*", "IS_SENT_START": False},
            {"ORTH": "by"},
            {"ORTH": "a"},
            {"OP": "?", "IS_SENT_START": False},
            {"ORTH": {"IN": ["supplement", "supplements"]}}
            ]
        pattern2 = [
            {"LOWER": "each"},
            {"LOWER": "time"},
            {"OP": "?", "IS_SENT_START": False},
            {"LEMMA": {"IN": ["sell", "offer"]}},
            {"LOWER": {"IN": ["securities", "security"]}},
            {"OP": "*", "IS_SENT_START": False},
            {"LEMMA": "provide"},
            {"OP": "*", "IS_SENT_START": False},
            {"LOWER": {"IN": ["supplement", "supplements"]}}
        ]
        pattern3 = [
            {"LOWER": "each"},
            {"LOWER": "time"},
            {"OP": "?", "IS_SENT_START": False},
            {"LOWER": {"IN": ["securities", "security"]}},
            {"LEMMA": "be"},
            {"LEMMA": {"IN": ["sell", "offer"]}},
            {"OP": "*", "IS_SENT_START": False},
            {"LEMMA": "provide"},
            {"OP": "*", "IS_SENT_START": False},
            {"LOWER": {"IN": ["supplement", "supplements"]}}
        ]
        matcher.add("base_prospectus", [pattern1, pattern2, pattern3])
        possible_matches = matcher(doc, as_spans=True)
        if len(possible_matches) > 0:
            logger.debug(f"This is a 'shelf' Prospectus, determined by matches: {possible_matches}")
            return True
        return False

    def extract_shelf_capacity(self, filing: Filing) -> dict:
        '''
        Extract the maximum offering amount (preferring $ over shares) from the registration table on the front page.
        
        Returns:
            Dict with keys: 'amount' and 'unit' or None.
        
        Raises:
            ValueError: if unhandled case is encountered.
        '''
        fp = filing.get_section("front page")
        if isinstance(fp, list):
            raise AttributeError(f"couldnt get the front page section; sections present: {[s.title for s in filing.sections]}")
        registration_table  = fp.get_tables(classification="registration_table", table_type="extracted")
        if registration_table is not None:
            if len(registration_table) == 1:
                registration_table = registration_table[0]["parsed_table"]
                logger.debug(f"registration_table: {registration_table}")
                try:
                    if re.search(re.compile("^total.*", re.I | re.MULTILINE), registration_table[-1][0]):
                        registration_table[-1][0] = "total"
                except IndexError:
                    pass

                column_names = self._get_registration_table_column_names(registration_table)
                if len(column_names) != len(registration_table[0]):
                    raise ValueError(f"Couldnt determine the correct column names for the registration_table to registration_df conversion. got column_names({column_names}) from first row({registration_table[0]})")
                registration_df = pd.DataFrame(registration_table[1:], columns=column_names)
                # values = []
                logger.debug(f"registration_df: {registration_df}")
                row = None
                if "total" in registration_df["Title"].values:
                    row = registration_df[registration_df["Title"] == "total"]
                elif len(registration_df["Title"].values) == 1:
                    row = registration_df.iloc[0]
                if row is None:
                    raise ValueError(f"Couldnt determine correct row while extracting from registration table: {registration_df}")
                if "Price Aggregate" in column_names:
                    if row["Price Aggregate"].item() != "":
                        return {"total_shelf_capacity": {"amount": row['Price Aggregate'].item(), "unit": "$"}}
                if "Amount" in column_names:
                    if row["Amount"].item() != "":
                        return {"total_shelf_capacity": {"amount": row["Amount"].item(), "unit": "shares"}}
                raise ValueError(f"Couldnt extract the offering amount with current cases handled, registration_df: {registration_df.columns, registration_df}")
        else:
            logger.info("Couldnt find registration table.. returning None")
        return None
    
    def _get_registration_table_column_names(self, registration_table: list[list]):
        middle_rows_re_to_column_title = [
                    (re.compile(r"Aggregate\s*offering.*Price", re.I), "Price Aggregate"),
                    (re.compile(r"Amount.*Registered", re.I), "Amount"),
                    (re.compile(r"Price.*per.*Unit", re.I), "Price Per Unit")
                ]
        column_names  = [x for x in range(len(registration_table[0]))]
        for t in middle_rows_re_to_column_title:
            re_term = t[0]
            title = t[1]
            for idx, col in enumerate(registration_table[0]):
                if re.search(re_term, col):
                    column_names[idx] = title
                    break
        column_names[0] = "Title"
        column_names[-1] = "Fee"
        return column_names
    
    def format_total_shelf_capacity(self, capacity_dict: Dict):
        if not capacity_dict:
            return None
        if "total_shelf_capacity" not in capacity_dict.keys():
            return None
        capacity = capacity_dict["total_shelf_capacity"]
        if capacity["unit"] == "$":
            return self.formater.quantity_string_to_float(capacity["amount"])
        if capacity["unit"] == "shares":
            return str(self.formater.quantity_string_to_float(capacity["amount"])) + " shares"
    

class HTMDEF14AExtractor(BaseHTMExtractor, AbstractFilingExtractor):
    def extract_form_values(self, filing: Filing):
        logger.warning(f"{self} NOT IMPLEMENTED DEF14A")


class HTMSC13GExtractor(BaseHTMExtractor, AbstractFilingExtractor):
    pass

class HTMSC13DExtractor(BaseHTMExtractor, AbstractFilingExtractor):
    pass

# add model for ownership tracking
# add commands for said model

class XMLEFFECTExtractor(AbstractFilingExtractor):
    def extract_form_values(self, filing: Filing, company: model.Company, bus: MessageBus):
        if filing.sections == []:
            return
        main_section = filing.sections[0]
        # if not isinstance(main_section, XMLFilingSection):
        #     raise TypeError(f"expecting XMLFilingSection got: {type(main_section)}")
        content = main_section.content_dict
        effect_registration = model.EffectRegistration(
            accn=filing.accession_number,
            file_number=content["file_number"],
            form_type=content["for_form"],
            effective_date=content["effective_date"]
        )
        bus.handle(commands.AddEffectRegistration(
                cik=company.cik,
                symbol=company.symbol,
                effect_registration=effect_registration
            )
        )
        # company.add_effect(effect_registration)
        return company


class ExtractorFactory:
    def __init__(self, defaults: list[tuple]=[]):
        self.extractors = {}
        if len(defaults) > 0:
            for case in defaults:
                self.register_extractor(*case)

    def register_extractor(
        self, form_type: str, extension: str, extractor: AbstractFilingExtractor
    ):
        self.extractors[(form_type, extension)] = extractor

    def get_extractor(self, form_type: str, extension: str, **kwargs):
        extractor = self.extractors.get((form_type, extension))
        if extractor:
            return extractor(**kwargs)
        else:
            raise ValueError(
                f"no extractor for that form_type and extension combination({form_type}, {extension}) registered"
            )

extractor_factory_default = [
    ("S-1", ".htm", HTMS1Extractor),
    ("DEF 14A", ".htm", HTMDEF14AExtractor),
    ("S-3", ".htm", HTMS3Extractor),
    ("EFFECT", ".xml", XMLEFFECTExtractor)
    # (None, ".htm", BaseHTMExtractor)
    ]

extractor_factory = ExtractorFactory(defaults=extractor_factory_default)


