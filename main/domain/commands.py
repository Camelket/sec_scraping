from datetime import date
from typing import Optional
from dataclasses import dataclass
import datetime

from main.domain import model



class Command:
    pass


@dataclass
class NeedCompanyCommand(Command):
    cik: str
    symbol: str

@dataclass
class SecuritiesCommand(NeedCompanyCommand):
    name: str

@dataclass
class AddSic(Command):
    sic: model.Sic

@dataclass
class AddFormType(Command):
    form_type: model.FormType

@dataclass
class AddFilingLinks(NeedCompanyCommand):
    filing_links: list[model.FilingLink]


@dataclass
class AddCompany(Command):
    company: model.Company


@dataclass
class AddSecurities(NeedCompanyCommand):
    securities: list[model.Security]

@dataclass
class AddSecurityAccnOccurence(NeedCompanyCommand):
    security_attributes: dict
    filing_accn: str

@dataclass
class AddShelfRegistration(NeedCompanyCommand):
    shelf_registration: model.ShelfRegistration

@dataclass
class AddResaleRegistration(NeedCompanyCommand):
    resale_registration: model.ResaleRegistration

@dataclass
class AddEffectRegistration(NeedCompanyCommand):
    effect_registration: model.EffectRegistration
    
@dataclass
class AddShelfOffering(NeedCompanyCommand):
    shelf_offering:  model.ShelfOffering

@dataclass
class AddShelfSecurityRegistration(NeedCompanyCommand):
    offering_accn: str
    security_registration: model.ShelfSecurityRegistration

@dataclass
class AddOutstandingSecurityFact(SecuritiesCommand):
    outstanding: list[model.SecurityOutstanding]




