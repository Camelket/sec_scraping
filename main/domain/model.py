from types import NoneType
from typing import Dict, Optional, Set, TypeAlias, TypeVar, Union
from typing_extensions import Self
from pydantic import AnyUrl, Json, ValidationError, validator, BaseModel, root_validator
from datetime import date, datetime, timedelta
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class SecurityNotFound(Exception):
    pass


class CommonShare(BaseModel):
    name: str = "common stock"
    par_value: float = 0.001

    def __repr__(self):
        return "CommonShare"

class PreferredShare(BaseModel):
    name: str
    par_value: float = 0.001

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
    expiry: datetime
    right: str = "Call" 
    multiplier: float = 1.0
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
    '''issuer has option of early redemption if condition are met'''
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
    
    def get_security_type(self, name: str):
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

class Security:
    def __init__(self, secu_type: str, secu_attributes: SecurityType, name: str=None, underlying: Optional[Self|str]=None):
        self.name = secu_attributes.name if name is None else name
        self.security_type = repr(secu_attributes)
        self.security_attributes = secu_attributes.json()
        self.underlying = underlying
        # self.convertible_to = Dict[Security]
    
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

@dataclass
class ShelfSecurityRegistration:
    security: Security
    amount: int
    source_security: Optional[Security] = field(default=None)

    def __eq__(self, other):
        if isinstance(other, ShelfSecurityRegistration):
            if (
                (self.security == other.security) and
                (self.amount == other.amount) and
                (self.source_security == other.source_security)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.amount, self.source_security, self.security))

@dataclass
class ShelfSecurityComplete:
    security: Security
    amount: int
    source_security: Optional[Security] = field(default=None)

    def __eq__(self, other):
        if isinstance(other, ShelfSecurityComplete):
            if (
                (self.security == other.security) and
                (self.amount == other.amount) and
                (self.source_security == other.source_security)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.amount, self.source_security, self.security))


@dataclass
class ShelfOffering:
    offering_type: str
    anticipated_offering_amount: float
    commencment_date: datetime
    end_date: datetime
    final_offering_amount_usd: float = 0.0
    underwriters: Set[Underwriter] = field(default_factory=set)
    registrations: Set[ShelfSecurityRegistration] = field(default_factory=set)
    completed: Set[ShelfSecurityComplete] = field(default_factory=set)

    def __repr__(self):
        return f"{self}: offering_type:{self.offering_type};anticipated_amount:{self.anticipated_offering_amount}"
    
    def __eq__(self, other):
        if isinstance(other, ShelfOffering):
            if (
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
            self.offering_type,
            self.anticipated_offering_amount,
            self.commencment_date,
            self.end_date,
            self.underwriters,
            self.registrations,
            self.completed
            ))

@dataclass
class ShelfRegistration:
    accn: str
    form_type: str
    capacity: int
    filing_date: date
    effect_date: Optional[date] = field(default=None)
    last_update: Optional[date] = field(default=None)
    expiry: Optional[date] = field(default=None)
    total_amount_raised: Optional[int] = field(default=None)
    total_amount_raised_unit: Optional[str] = field(default="USD")
    offerings: Set[ShelfOffering] = field(default_factory=set)

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
    amount: int
    source_security: Optional[Security] = field(default=None)

    def __eq__(self, other):
        if isinstance(other, ResaleSecurityRegistration):
            if (
                (self.security == other.security) and
                (self.amount == other.amount) and
                (self.source_security == other.source_security)
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.amount, self.source_security, self.security))

@dataclass
class ResaleSecurityComplete:
    security: Security
    amount: int
    source_security: Optional[Security] = field(default=None)

    def __eq__(self, other):
        if isinstance(other, ResaleSecurityComplete):
            if (
                (self.security == other.security) and
                (self.amount == other.amount) and
                (self.source_security == other.source_security)
            ):
                return True
        return False

    def __hash__(self):
        return hash((self.amount, self.source_security, self.security))

@dataclass
class ResaleRegistration:
    accn: str
    form_type: str
    filing_date: date
    effect_date: Optional[date] = field(default=None)
    last_update: Optional[date] = field(default=None)
    expiry: Optional[date] = field(default=None)
    registrations: Set[ResaleSecurityRegistration] = field(default_factory=set)
    completed: Set[ResaleSecurityComplete] = field(default_factory=set)

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


class Company:
    def __init__(self, name: str, cik: str, sic: str, symbol: str, description_: Optional[str]=None, securities: set[Security]=None, shelfs: set[ShelfRegistration]=None, resales: set[ResaleRegistration]=None, filing_parse_history: set[FilingParseHistoryEntry]=None):
        self.name = name
        self.cik = cik,
        self.sic = sic,
        self.symbol = symbol,
        self.description_ = description_
        self.securities = set() if securities is None else securities
        self.shelfs = set() if shelfs is None else shelfs
        self.resales = set() if resales is None else resales
        self.filing_parse_history = set() if filing_parse_history is None else filing_parse_history
        self.filing_links = set()
        self.security_conversion = set()
        self.cash_operating = set()
        self.cash_finacing = set()
        self.cash_investing = set()
        self.net_cash_and_equivalents = set()
    
    def __eq__(self, other):
        if isinstance(other, Company):
            if (
                (self.cik == other.cik)
                
            ):
                return True
        return False
    
    def __hash__(self):
        return hash((self.cik))


    def change_name(self, new_name):
        self.name = new_name
    
    def add_security(self, secu: Security):
        if (secu.underlying is not None) and (isinstance(secu.underlying, str)):
            underlying = self.get_security_by_name(secu["underlying"])
            if underlying is None:
                raise SecurityNotFound(f"Couldnt find underlying Security with name: {secu['underlying']}. Make sure the underlying was added!")
            else:
                secu.underlying = underlying
        self.securities.add(secu)
    
    def add_security_conversion(self, secu_conversion: SecurityConversion):
        self.security_conversion.add(secu_conversion)
    
    # def add_security(self, name: str, secu_type: str, secu_attributes: SecurityType, underlying: Optional[Security]=None):
    #     self.securities.add(
    #         Security(
    #             name=name,
    #             secu_type=secu_type,
    #             secu_attributes=secu_attributes,
    #             underlying=underlying

    #         ))
    
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