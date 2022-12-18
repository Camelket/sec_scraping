from enum import Enum
from types import NoneType
from typing import Any, Dict, List, Optional, Set, TypeAlias, TypeVar, Union
from typing_extensions import Self
from pydantic import AnyUrl, Json, ValidationError, validator, BaseModel, root_validator
from datetime import date, datetime, timedelta
import logging
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

class SecurityNotFound(Exception):
    pass

class AllowedSecurityTypes(Enum):
    CommonShare = 'CommonShare'
    PreferredShare = 'PreferredShare'
    DebtSecurity = 'DebtSecurity'
    Option = 'Option'
    Warrant = 'Warrant'


class ResetFeature(BaseModel):
    '''resets feature to a certain value if certain conditions are met. eg: conversion price is adjusted if stock price is below x for y days'''
    duration: timedelta 
    min_occurence: float # how many x of min_occurence_frequenzy should evaluate to true of the reset_clause/reset_feature and security_feature during duration 
    min_occurence_frequency: str 
    security: str #  security on which the reset_clause is checked over duration
    security_feature: str # feature of the security to be checked for reset_clause eg: VWAP, close, high, low
    reset_feature: str #  must be a field of the parent eg: conversion_price
    reset_clause: float # percent as float eg: 100% == 1.0
    reset_to: str # value to which the reset_feature is set if the reset_clause is valid over duration in the security_feature
   

class CallFeature(BaseModel):
    '''issuer has option of early redemption if conditions are met'''
    duration: timedelta
    min_occurence: float
    min_occurence_frequency: str
    security: str
    security_feature: str
    call_feature: str
    call_clause: float
   
    
class PutFeature(BaseModel):
    '''holder has option of early repayment of all or part'''
    start_date: datetime
    duration: timedelta
    frequency: timedelta
    other_conditions: str


class ContigentConversionFeature(BaseModel):
    duration: timedelta
    min_occurence: float
    min_occurence_frequency: str
    security: str
    security_feature: str
    conversion_feature: str
    conversion_clause: float
    

class ConvertibleFeature(BaseModel):
    conversion_ratio: float
    contigent_conversion_features: list[ContigentConversionFeature] = []
    call_features: list[CallFeature] = []
    put_features: list[PutFeature] = []
    reset_features: list[ResetFeature] = []
    
class CommonShare(BaseModel):
    name: str = "common stock"
    par_value: float = 0.001

    def __repr__(self):
        return "CommonShare"

class PreferredShare(BaseModel):
    name: str
    par_value: float = 0.001
    convertible_feature: Optional[ConvertibleFeature] = None

    def __repr__(self):
        return "PreferredShare"

class Option(BaseModel):
    name: str
    strike_price: float
    expiry: datetime
    right: str
    multiplier: float = 100
    issue_date: Optional[datetime] = None

    def __repr__(self):
        return "Option"

class Warrant(BaseModel):
    name: str
    exercise_price: float
    right: str = "Call" 
    multiplier: float = 1.0
    expiry_date: Optional[datetime] = None
    expiry_timedelta: Optional[timedelta] = None
    issue_date: Optional[datetime] = None

    def __repr__(self):
        return "Warrant"

class DebtSecurity(BaseModel):
    name: str
    interest_rate: float
    maturity: datetime
    issue_date: Optional[datetime] = None

    def __repr__(self):
        return "DebtSecurity"

    

SecurityType: TypeAlias =  Union[CommonShare, PreferredShare, Option, Warrant, DebtSecurity, NoneType]

class SecurityTypeFactory:
    '''get the correct Security Type from a string'''
    def __init__(self):
        self.builders = {
            "common": CommonShare,
            "preferred": PreferredShare,
            "option": Option,
            "warrant": Warrant,
            "note": DebtSecurity
        }
        self.keywords = list(self.builders.keys())
        self.fallback = DebtSecurity

    def register_builders(self, keywords: list, security_type: SecurityType):
        for kw in keywords:
            if kw in self.keywords:
                raise AttributeError(f"keyword is already associated with a different Security Type: {kw} associated with {self.builders[kw]}")
            else:
                self.builders[kw] = security_type
                self.keywords.append(kw)
                self.keywords.sort(key=lambda x: len(x))
    
    def get_security_type(self, name: str) -> SecurityType:
        for kw in self.keywords:
            if kw in name:
                return self.builders[kw]
        return self.fallback

@dataclass
class Underwriter:
    name: str

@dataclass
class FormType:
    form_type: str
    category: str

@dataclass
class OfferingStatus:
    status_name: str

@dataclass
class SecurityCusip:
    cusip_number: str
    security_name: str

@dataclass
class SecurityOutstanding:
    amount: int
    instant: date

    def __eq__(self, other):
        if isinstance(other, SecurityOutstanding):
            if (
                (self.amount == other.amount) and
                (self.instant == other.instant)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.amount, self.instant))

@dataclass
class SecurityAuthorized:
    security_type: str
    amount: int
    instant: date

    def __post_init__(self):
        if self.security_type not in [member.value for member in AllowedSecurityTypes]:
            raise TypeError(f"unallowed security_type given: {self.security_type}. Allowed are: {[member.value for member in AllowedSecurityTypes]}")

    def __eq__(self, other):
        if isinstance(other, SecurityAuthorized):
            if (
                (self.security_type == other.security_type) and
                (self.amount == other.amount) and
                (self.instant == other.instant)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.amount, self.instant, self.security_type))


class Security:
    def __init__(self, secu_attributes: SecurityType, name: str=None, underlying: Optional[Self|str]=None):
        self.name = secu_attributes.name if name is None else name
        self.security_type = repr(secu_attributes)
        self.security_attributes = secu_attributes.json()
        self.underlying = underlying
        self.outstanding: Set = set()
        # self.convertible_to = Dict[Security]
    
    def add_outstanding(self, new_outstanding: SecurityOutstanding):
        if new_outstanding not in self.outstanding:
            self.outstanding.add(new_outstanding)
    
    def __repr__(self):
        return str(self.security_attributes)
    
    def __eq__(self, other):
        if isinstance(other, Security):
            if (
                (self.name == other.name) and
                (self.security_attributes == other.security_attributes) and
                (self.underlying == other.underlying)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.name, self.underlying))


@dataclass
class SecurityConversion:
    conversion_attributes: Json
    from_security: Security
    to_security: Security

    def __eq__(self, other):
        if isinstance(other, SecurityConversion):
            if (
                (self.conversion_attributes == other.conversion_attributes) and
                (self.from_security == other.from_security) and
                (self.to_security == other.to_security)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.from_security, self.to_security))

@dataclass
class SecurityAccnOccurence:
    security_id: int
    accn: str

    def __eq__(self, other):
        if isinstance(other, SecurityAccnOccurence):
            if (
                (self.security_id == other.security_id) and
                (self.accn == other.accn)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.security_id, self.accn))
    

@dataclass
class ShelfSecurityRegistration:
    security: Security
    amount_dollar: int
    amount_security: int
    source_security: Optional[Security] = field(default=None)

    def __eq__(self, other):
        if isinstance(other, ShelfSecurityRegistration):
            if (
                (self.security == other.security) and
                (self.amount_dollar == other.amount_dollar) and
                (self.amount_security == other.amount_security) and
                (self.source_security == other.source_security)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.amount_dollar, self.amount_security, self.source_security, self.security))

@dataclass
class ShelfSecurityComplete:
    security: Security
    amount_dollar: int
    amount_security: int
    source_security: Optional[Security] = field(default=None)

    def __eq__(self, other):
        if isinstance(other, ShelfSecurityComplete):
            if (
                (self.security == other.security) and
                (self.amount_dollar == other.amount_dollar) and
                (self.amount_security == other.amount_security) and
                (self.source_security == other.source_security)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.amount_dollar, self.amount_security, self.source_security, self.security))


@dataclass
class ShelfOffering:
    offering_type: str
    accn: str
    anticipated_offering_amount: float
    commencment_date: datetime
    end_date: datetime
    final_offering_amount_usd: float = 0.0
    underwriters: Set[Underwriter] = field(default_factory=set)
    registrations: Set[ShelfSecurityRegistration] = field(default_factory=set)
    completed: Set[ShelfSecurityComplete] = field(default_factory=set)

    def add_registration(self, registered: ShelfSecurityRegistration):
        self.registrations.add(registered)
    
    def add_complete(self, completed: ShelfSecurityComplete):
        self.completed.add(completed)

    def __repr__(self):
        return f"{self.__class__}: offering_type:{self.offering_type}; accn: {self.accn}; anticipated_amount:{self.anticipated_offering_amount}; commencment_date: {self.commencment_date}; end_date: {self.end_date}"
    
    def __eq__(self, other):
        if isinstance(other, ShelfOffering):
            if (
                (self.accn == other.accn) and
                (self.offering_type == other.offering_type) and
                (self.anticipated_offering_amount == other.anticipated_offering_amount) and
                (self.commencment_date == other.commencment_date) and
                (self.end_date == self.end_date) and
                (self.underwriters == self.underwriters) and
                (self.registrations == self.registrations) and
                (self.completed == self.completed)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((
            self.accn,
            self.offering_type,
            self.anticipated_offering_amount,
            self.commencment_date,
            self.end_date
            ))

@dataclass
class ShelfRegistration:
    accn: str
    file_number: str
    form_type: str
    capacity: int
    filing_date: date
    effect_date: Optional[date] = field(default=None)
    last_update: Optional[date] = field(default=None)
    expiry: Optional[date] = field(default=None)
    total_amount_raised: Optional[int] = field(default=None)
    offerings: Set[ShelfOffering] = field(default_factory=set)

    def get_offering_by_accn(self, accn: str):
        for o in self.offerings:
            if o.accn == accn:
                return o
    
    def add_offering(self, offering: ShelfOffering):
        if offering in self.offerings:
            logger.debug(f"Tried to add duplicate ShelfOffering into ShelfRegistration.offerings")
        else:    
            self.offerings.add(offering)
    

    def __eq__(self, other):
        if isinstance(other, ShelfRegistration):
            if (
                (self.accn == other.accn)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.accn,))

@dataclass
class ResaleSecurityRegistration:
    security: Security
    amount_security: int
    source_security: Optional[Security] = field(default=None)

    def __eq__(self, other):
        if isinstance(other, ResaleSecurityRegistration):
            if (
                (self.security == other.security) and
                (self.amount_security == other.amount_security) and
                (self.source_security == other.source_security)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.amount_security, self.source_security, self.security))

@dataclass
class ResaleSecurityComplete:
    security: Security
    amount_security: int
    source_security: Optional[Security] = field(default=None)

    def __eq__(self, other):
        if isinstance(other, ResaleSecurityComplete):
            if (
                (self.security == other.security) and
                (self.amount_security == other.amount_security) and
                (self.source_security == other.source_security)
            ):
                return True
        return False

    def __hash__(self):
        return hash((self.amount_security, self.source_security, self.security))

@dataclass
class ResaleRegistration:
    accn: str
    file_number: str
    form_type: str
    filing_date: date
    effect_date: Optional[date] = field(default=None)
    last_update: Optional[date] = field(default=None)
    expiry: Optional[date] = field(default=None)
    registrations: Set[ResaleSecurityRegistration] = field(default_factory=set)
    completed: Set[ResaleSecurityComplete] = field(default_factory=set)

    def add_registration(self, registered: ResaleSecurityRegistration):
        self.registrations.add(registered)
    
    def add_complete(self, completed: ResaleSecurityComplete):
        self.completed.add(completed)

    def __eq__(self, other):
        if isinstance(other, ResaleRegistration):
            if (
                (self.accn == other.accn)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.accn, ))

@dataclass
class EffectRegistration:
    accn: str
    file_number: str
    form_type: str
    effective_date: date

    def __eq__(self, other):
        if isinstance(other, EffectRegistration):
            if (
                (self.accn == other.accn) and
                (self.form_type == other.form_type) and
                (self.effective_date == other.effective_date) and
                (self.file_number == other.file_number)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.accn, self.form_type, self.file_number, self.effective_date))


@dataclass
class Sic:
    sic: int
    sector: str
    industry: str
    division: str

    def __eq__(self, other):
        if isinstance(other, Sic):
            if (
                (self.sic == other.sic)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.sic,))


@dataclass
class FilingParseHistoryEntry:
    accession_number: str
    date_parsed: date

    def __eq__(self, other):
        if isinstance(other, FilingParseHistoryEntry):
            if (
                (self.accession_number == other.accession_number) and
                (self.date_parsed == other.date_parsed)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.accession_number, self.date_parsed))

@dataclass
class FilingLink:
    filing_html: AnyUrl
    accn: str
    form_type: str
    filing_date: date
    description_: str
    file_number: str

    def __eq__(self, other):
        if isinstance(other, FilingLink):
            if (
                (self.filing_html == other.filing_html)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.filing_html,))

@dataclass
class CashOperating:
    from_date: date
    to_date: date
    amount: int

    def __eq__(self, other):
        if isinstance(other, CashOperating):
            if (
                (self.from_date == other.from_date) and
                (self.to_date == other.to_date) and
                (self.amount == other.amount)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.from_date, self.to_date, self.amount))

@dataclass
class CashFinancing:
    from_date: date
    to_date: date
    amount: int

    def __eq__(self, other):
        if isinstance(other, CashFinancing):
            if (
                (self.from_date == other.from_date) and
                (self.to_date == other.to_date) and
                (self.amount == other.amount)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.from_date, self.to_date, self.amount))

@dataclass
class CashInvesting:
    from_date: date
    to_date: date
    amount: int

    def __eq__(self, other):
        if isinstance(other, CashInvesting):
            if (
                (self.from_date == other.from_date) and
                (self.to_date == other.to_date) and
                (self.amount == other.amount)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.from_date, self.to_date, self.amount))

@dataclass
class NetCashAndEquivalents:
    instant: date
    amount: int

    def __eq__(self, other):
        if isinstance(other, NetCashAndEquivalents):
            if (
                (self.instant == other.instant) and
                (self.amount == other.amount)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.instant, self.amount))

# @dataclass
# class CompanyLastUpdate:
#     pass


class Company:
    def __init__(self, name: str, cik: str, sic: int, symbol: str, description_: Optional[str]=None):
        self.name = name
        self.cik = cik
        self.sic = sic
        self.symbol = symbol
        self.description_ = description_
        self.securities = set()
        self.shelfs = set()
        self.resales = set()
        self.effects = set()
        self.filing_parse_history = set()
        self.filing_links = set()
        self.security_conversion = list()
        self.cash_operating = set()
        self.cash_finacing = set()
        self.cash_investing = set()
        self.net_cash_and_equivalents = set()
        self.securities_authorized = set()
    
    def __repr__(self):
        return str(self.cik)
    
    def __eq__(self, other):
        if isinstance(other, Company):
            if (
                (self.cik == other.cik)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.cik, ))


    def change_name(self, new_name):
        self.name = new_name
    
    def _get_security(self, secu: Security):
        for s in self.securities:
            if s == secu:
                return s
        return None
    
    def get_shelf_by_accn(self, accn: str):
        for shelf in self.shelfs:
            if shelf.accn == accn:
                return shelf
        return None

    def get_shelf(self, file_number: str):
        for shelf in self.shelfs:
            if shelf.file_number == file_number:
                return shelf
        return None
    
    def get_shelf_offering(self, offering_accn: str):
        for shelf in self.shelfs:
            offering = shelf.get_offering_by_accn(offering_accn)
            if offering:
                return offering
        return None
    
    def get_resale(self, accn: str):
        for resale in self.resales:
            if resale.accn == accn:
                return resale
        return None
    
    def add_filing_link(self, filing_link: FilingLink):
        if filing_link in self.filing_links:
            logger.debug(f"Tried to add duplicate ShelfRegistration into Company.shelfs")
        else:
            self.filing_links.add(filing_link)
    
    def add_shelf(self, shelf: ShelfRegistration):
        if shelf in self.shelfs:
            logger.debug(f"Tried to add duplicate ShelfRegistration into Company.shelfs")
        else:
            self.shelfs.add(shelf)
    
    def add_resale(self, resale: ResaleRegistration):
        if resale in self.resales:
            logger.debug(f"Tried to add duplicate ResaleRegistration into Company.resales")
        else:
            self.resales.add(resale)
    
    def add_effect(self, effect: EffectRegistration):
        if effect in self.effects:
            logger.debug(f"Tried to add duplicate EffectRegistration into Company.effects")
        else:
            self.effects.add(effect)
            self.set_effect_date(effect, "shelfs")
            self.set_effect_date(effect, "resales")
   
    def set_effect_date(self, effect: EffectRegistration, attr_name="shelfs"):
        registrations = getattr(self, attr_name, None)
        if registrations:
            registration = next((x for x in registrations if (x.file_number == effect.file_number and x.form_type == effect.form_type)), None)
            if registration:
                registration.effect_date = effect.effective_date
                logger.debug(f"set effect_date: {effect.effective_date} for registration: {registration}")
    
    def add_security(self, secu: Security):
        if (secu.underlying is not None) and (isinstance(secu.underlying, str)):
            underlying = self.get_security_by_name(secu.underlying)
            if underlying is None:
                raise SecurityNotFound(f"Couldnt find underlying Security with name: {secu['underlying']}. Make sure the underlying was added!")
            else:
                secu.underlying = underlying
        if secu in self.securities:
            logger.debug(f"Tried to add duplicate Security into Company.securities")
        else:
            self.securities.add(secu)
    
    def add_security_conversion(self, secu_conversion: SecurityConversion):
        if secu_conversion.from_security not in self.securities:
            raise ValueError
        if secu_conversion.to_security not in self.securities:
            raise ValueError
        from_security = self._get_security(secu_conversion.from_security)
        if from_security is not None:
            if secu_conversion not in from_security.conversion_from_self:
                from_security.conversion_from_self.append(secu_conversion)
        to_security = self._get_security(secu_conversion.to_security)
        if to_security is not None:
            if secu_conversion not in to_security.conversion_to_self:
                to_security.conversion_to_self.append(secu_conversion)
    
    def add_security_authorized(self, authorized: SecurityAuthorized):
        if authorized not in self.securities_authorized:
            self.securities_authorized.add(authorized)
    
    def get_security_by_attributes(self, attributes: dict[str, Any]):
        '''returns first security matching attributes or None if no security has matching attributes'''
        for secu in self.securities:
            secu_attr_dict = json.loads(secu.security_attributes)
            for attribute, value in attributes.items():
                secu_attr_value = getattr(secu_attr_dict, attribute, None)
                if secu_attr_value:
                    if secu_attr_value == value:
                        pass
                    else:
                        break
                else:
                    break
            return secu
        return None
            
    def get_security_by_name(self, name):
        for secu in self.securities:
            if name == secu.name:
                return secu
        return None
    
    def get_securities_by_type(self, secu_type: str):
        securities = []
        for secu in self.securities:
            if secu_type == secu.security_type:
                securities.append(secu)
        return securities


# make company an aggregate and do all the work from there ?