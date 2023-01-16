from dataclasses import dataclass
from typing import Any
from spacy.tokens import Span, Token
from main.parser.filing_nlp_dependency_matcher import (
    SourceContext,
    SecurityDependencyAttributeMatcher,    
)
from main.parser.filing_nlp_dateful_relations import DatetimeRelation
from main.parser.filing_nlp_patterns import (
    SECU_GET_EXERCISE_DATE_LEMMA_COMBINATIONS,
    SECU_GET_EXPIRY_DATE_LEMMA_COMBINATIONS, 
)
import logging
logger = logging.getLogger(__name__)

class UnitAmount:
    def __init__(self, amount, unit):
        self.amount: float = amount
        self.unit: str = unit

    def __eq__(self, other):
        if isinstance(other, UnitAmount):
            if (self.amount == other.amount) and (self.unit == other.unit):
                return True
        return False

    def __repr__(self):
        return f"{self.amount} {self.unit}"


class SECUQuantity:
    def __init__(self, original, attr_matcher: SecurityDependencyAttributeMatcher):
        self.original: Token | Span = original
        self._quantity = original._.secuquantity
        self._unit: str = original._.secuquantity_unit
        self.amount: UnitAmount = UnitAmount(self._quantity, self._unit)
        self.amods: list = original._.amods
        self.parent_verb = attr_matcher.get_parent_verb(self.original)
        self._date_relations = attr_matcher.get_date_relation(self.original)
        self.datetime_relation: DatetimeRelation = self._get_datetime_relation()

    def _get_datetime_relation(self):
        if self._date_relations:
            if len(self._date_relations["datetime"]) == 0:
                return None
            elif len(self._date_relations["datetime"]) == 1:
                return self._date_relations["datetime"][0]
            elif len(self._date_relations) > 1:
                raise ValueError(
                    f"More than one datetime relation present for {self.original}, failed to determine a certain context -> failed to create the SECUQuantity object."
                )
        else:
            return None

    def __repr__(self):
        return f"SECUQuantity({self.amount} -\
                \t {self.amods} - \
                \t {self.datetime_relation})"

    def __eq__(self, other):
        if isinstance(other, SECUQuantity):
            has_same_timestamp = False
            if isinstance(self.datetime_relation, DatetimeRelation) and isinstance(
                other.datetime_relation, DatetimeRelation
            ):
                if (
                    self.datetime_relation.timestamp
                    == other.datetime_relation.timestamp
                ):
                    has_same_timestamp = True
            if (self.amount == other.amount) and (
                self.datetime_relation == other.datetime_relation or has_same_timestamp
            ):
                return True
        return False




@dataclass
class QuantityRelation:
    quantity: SECUQuantity
    main_secu: Any
    rel_type: str = "quantity"

    def __repr__(self):
        return f"QuantityRelation(\
                {self.quantity}\
                \t {self.main_secu.original})"

    def __eq__(self, other):
        if (self.quantity == other.quantity) and (self.main_secu == other.main_secu):
            return True
        return False
    
    def fulfills_conditions(self, is_negated=False, has_datetime_relation=True, min_certainty=0.85, certainty_fallback=1.0, has_parent_verb=True):
        if self.quantity.original._.negated is is_negated:
            if (True if self.quantity.datetime_relation else False) is has_datetime_relation:
                certainty = certainty_fallback
                if (True if self.quantity.parent_verb else False) is has_parent_verb:
                    if self.quantity.parent_verb:
                        if self.quantity.parent_verb._.certainty_info:
                            certainty = self.quantity.parent_verb._.certainty_info.determine_level()
                    if certainty >= min_certainty:
                        return True
        return False

class SourceQuantityRelation():
    def __init__(self, quantity: SECUQuantity, target_secu: Any, source_secu: Any):
        self.quantity: SECUQuantity = quantity
        self.target_secu: SECU = target_secu
        self.source_secu: SECU = source_secu
        self.rel_type = "source_quantity"

    def __repr__(self):
        return f"SourceQuantityRelation({self.quantity.amount} {self.target_secu.secu_key} from {self.source_secu.secu_key})"


# TODO: handle and give better error message for failing to extend datetime root token to full span (eg out of bounds nanosecond timestamp)
    

class SECU:
    def __init__(self, original, attr_matcher: SecurityDependencyAttributeMatcher):
        self.original: Token = self._set_original(original)
        self.attr_matcher: SecurityDependencyAttributeMatcher = attr_matcher
        self.secu_key: str = original._.secu_key
        self.amods = original._.amods
        self.adj = original._.adj
        self.other_relations: list = list()
        self.quantity_relations: list[QuantityRelation] = list()
        self._set_quantity_relations()
        self.source_quantity_relations: list[SourceQuantityRelation] = list()
        self.root_verb: Token | None = attr_matcher.get_root_verb(self.original)
        self.parent_verb: Token | None = attr_matcher.get_parent_verb(self.original)
        self.aux_verbs: list = attr_matcher.get_aux_verbs(self.original)
        self.date_relations: list[dict] = attr_matcher.get_date_relation(self.original)
        self.exercise_price: UnitAmount | None = self._format_exercise_price(self.original)
        self.expiry_date = self._get_expiry_datetime_relation()
        self.exercise_date = self._get_exercise_datetime_relation()
        # TODO: maybe add securitytype through the factory with secu_key?
    
    def _format_exercise_price(self, original: Token) -> UnitAmount|None:
        ep = self.attr_matcher.get_exercise_price(original)
        if ep:
            return UnitAmount(ep[0], ep[1])
        return None
    
    def get_outstanding_quantity_relations(self) -> list[QuantityRelation]|None:
        '''
        check the quantity_relations instance variable for a relation meeting the criteria for
        an outstanding security quantity.
        criteria are:
            - "outstanding" in amods
            - has a datetime relation
            - certainty above 0.85
            - not negated
        '''
        if self.amods:
            str_amods = [i.lower_ for i in self.amods]
        else:
            str_amods = []
        result = []
        for qr in self.quantity_relations:
            if qr.fulfills_conditions(
                is_negated=False,
                has_datetime_relation=True,
                min_certainty=0.85,
                certainty_fallback=1.0,
                has_parent_verb=True
            ):
                quant = qr.quantity
                if quant.amods is not None:
                    amods = [i.lower_ for i in quant.amods] + str_amods
                else:
                    amods = str_amods
                if ("outstanding" in amods):
                    if quant.amount.amount is not None:
                        result.append(qr)
        return result if result != [] else None
                                    
    
    def _date_relation_attr_with_subset_of_pattern(
        self, valid_patterns: list[dict[str, set]], attr: str = "lemmas"
    ):
    # FIXME: adjust for addition of DatetimeRelation instead of before used dict -> just access the attribute 
        """
        valid_patterns:
        [
            {
                "prep": set(["as", "of", "on"]),
                "adj": set(["exercisable"]),
            },
            ...
        ]
        """
        matches = list()
        for date_relation in self.date_relations["datetime"]:
            if not getattr(date_relation, attr):
                continue
            for pattern in valid_patterns:
                keys_to_check = pattern.keys()
                attrs = getattr(date_relation, attr, None)
                if not attrs:
                    continue
                if not all([k in attrs.keys() for k in keys_to_check]):
                    logger.debug(
                        f"SECU. _date_relation_attr_has_subset_of_pattern(): not all required keys found. Missing:{[k if k not in attrs.keys() else None for k in keys_to_check]} "
                    )
                    continue
                if all([attrs[key].issubset(pattern[key]) for key in keys_to_check]):
                    if date_relation not in matches:
                        matches.append(date_relation)
        return matches

    def _get_exercise_datetime_relation(self):
        if (self.date_relations is None) or (len(self.date_relations["datetime"]) == 0):
            return None
        valid_patterns = SECU_GET_EXERCISE_DATE_LEMMA_COMBINATIONS
        matches = self._date_relation_attr_with_subset_of_pattern(
            valid_patterns, attr="lemmas"
        )
        return list(matches)[0] if len(matches) == 1 else None

    def _get_expiry_datetime_relation(self):
        # TODO: implement this for the timedelta case (when i have a grip on issuance dates)
        if (self.date_relations is None) or (len(self.date_relations["datetime"]) == 0):
            return None
        valid_patterns = SECU_GET_EXPIRY_DATE_LEMMA_COMBINATIONS
        matches = self._date_relation_attr_with_subset_of_pattern(
            valid_patterns, "lemmas"
        )
        if len(matches) > 1:
            raise ValueError(
                f"SECU._get_expiry_datetime_relation() found more than one expiry for a specific security, case not handled. dates found: {matches}, secu: {self}"
            )
        return list(matches)[0] if len(matches) == 1 else None
    
    def _set_quantity_relations(self):
        quants = self.attr_matcher.get_quantities(self.original)
        if not quants:
            # logger.debug(f"no quantities found for secu: {self.original}")
            return
        for quant in quants:
            try:
                quant_obj = SECUQuantity(quant, self.attr_matcher)
                rel = QuantityRelation(quant_obj, self)
                self._add_quantity_relation(rel)
            except ValueError as e:
                logger.debug(e)
                continue
            
    def _set_original(self, original: Token | Span) -> None:
        if isinstance(original, Token):
            return original
        elif isinstance(original, Span):
            if len(original) == 1:
                return original[0]
            else:
                raise NotImplementedError(
                    "Span with more than one token for use as original argument is not supported yet."
                )

    def _add_quantity_relation(self, relation: QuantityRelation):
        if not isinstance(relation, QuantityRelation):
            raise TypeError(f"Tried adding object with wrong type: {type(relation)}; expecting a QuantityRelation object")
        if relation not in self.quantity_relations:
            self.quantity_relations.append(relation)
        else:
            logger.debug(f"quantity_relation {relation} already present in {self}")
    
    def add_source_quantity_relation(self, relation: SourceQuantityRelation):
        if not isinstance(relation, SourceQuantityRelation):
            raise TypeError(f"Tried adding object with wrong type: {type(relation)}; expecting a SourceQuantityRelation object")
        if relation not in self.source_quantity_relations:
            self.source_quantity_relations.append(relation)
        else:
            logger.debug(f"relation {relation} already present in {self}")

    def _add_relation(self, relation):
        #TODO[epic=maybe]: do I even need self.other_relations attribute?
        # consider the case were we want to assign a context to a secu instance,
        # wouldnt I just assign that through extensions on doc, span, token objects?
        # -> at most this should be an accessebility wrapper
        if relation not in self.other_relations:
            self.other_relations.append(relation)
        else:
            logger.debug(f"relation {relation} already in relations of {self}")

    def __repr__(self):
        rels = ""
        for x in range(len(self.other_relations)):
            rels += f"\n\t {x}) {self.other_relations[x]}"
        return f"SECU({self.original}\
                \t secu_key: {self.secu_key}\
                \t amods: {self.amods}\
                \n\t relations: {rels})"

    def __eq__(self, other):
        if not isinstance(other, SECU):
            logger.debug(
                f"cant compare {self} to {other}. other not of type SECU, but of type: {type(other)}"
            )
            return False
        return self.secu_key == other.secu_key and self.original == other.original

