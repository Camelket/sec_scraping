import re
import logging
from pathlib import Path
from tkinter import E
from turtle import shape
from types import NoneType
from typing import Callable
from numpy import require
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, element
import re
from abc import ABC, abstractmethod
import copy
import uuid
from xml.etree import ElementTree

from main.parser.filings_base import FilingSection, Filing, FilingSection
from bs4 import BeautifulSoup
import logging
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

'''
Notes:
    - need to check if we ever have a case where sourcepos of html5lib (closing tag)
      causes a problem with splitting into sections.
'''

DATE_OF_REPORT_PATTERN = r"(?:(?:(?:Date(?:.?|\n?)of(?:.?|\n?)report(?:[^\d]){0,40})((?:(?:(?:(?:January)|(?:February)|(?:March)|(?:April)|(?:May)|(?:June)|(?:July)|(?:August)|(?:September)|(?:October)|(?:November)|(?:December))|(?:(?:Jan(?:(?:\D){0,4}))|(?:Feb(?:(?:\D){0,5}))|(?:Mar(?:(?:\D){0,2}))|(?:Apr(?:(?:\D){0,2}))|(?:May)|(?:Jun(?:(?:\D){0,1}))|(?:Jul(?:(?:\D){0,1}))|(?:Aug(?:(?:\D){0,4}))|(?:Sep(?:(?:\D){0,6}))|(?:Oct(?:(?:\D){0,4}))|(?:Nov(?:(?:\D){0,5}))|(?:Dec(?:(?:\D){0,6}))))(?:(?:[^\d]){0,5}\d.(?:[^\d]){0,6}\d\d\d\d))|(?:(?:(?:\d\d)|(?:[^\d]\d))(?:.){0,2}(?:(?:(?:January)|(?:February)|(?:March)|(?:April)|(?:May)|(?:June)|(?:July)|(?:August)|(?:September)|(?:October)|(?:November)|(?:December))|(?:(?:Jan(?:(?:\D){0,4}))|(?:Feb(?:(?:\D){0,5}))|(?:Mar(?:(?:\D){0,2}))|(?:Apr(?:(?:\D){0,2}))|(?:May)|(?:Jun(?:(?:\D){0,1}))|(?:Jul(?:(?:\D){0,1}))|(?:Aug(?:(?:\D){0,4}))|(?:Sep(?:(?:\D){0,6}))|(?:Oct(?:(?:\D){0,4}))|(?:Nov(?:(?:\D){0,5}))|(?:Dec(?:(?:\D){0,6}))))(?:(?:[^\d]){0,5}\d.(?:[^\d]){0,6}\d\d\d\d))))|(?:((?:(?:(?:January)|(?:February)|(?:March)|(?:April)|(?:May)|(?:June)|(?:July)|(?:August)|(?:September)|(?:October)|(?:November)|(?:December))|(?:(?:Jan(?:(?:\D){0,4}))|(?:Feb(?:(?:\D){0,5}))|(?:Mar(?:(?:\D){0,2}))|(?:Apr(?:(?:\D){0,2}))|(?:May)|(?:Jun(?:(?:\D){0,1}))|(?:Jul(?:(?:\D){0,1}))|(?:Aug(?:(?:\D){0,4}))|(?:Sep(?:(?:\D){0,6}))|(?:Oct(?:(?:\D){0,4}))|(?:Nov(?:(?:\D){0,5}))|(?:Dec(?:(?:\D){0,6}))))(?:[^\d]){0,5}\d.(?:[^\d]){0,6}\d\d\d\d(?:\n{0,3}))(?:.){0,10}(?:date(?:.){0,3}of(?:.){0,3}report)))|(?:(?:(?:Date(?:[^\d]){0,5}of(?:[^\d]){0,20}report(?:[^\d]){0,40})(?:((?:(?:(?:January)|(?:February)|(?:March)|(?:April)|(?:May)|(?:June)|(?:July)|(?:August)|(?:September)|(?:October)|(?:November)|(?:December))|(?:(?:Jan(?:(?:\D){0,4}))|(?:Feb(?:(?:\D){0,5}))|(?:Mar(?:(?:\D){0,2}))|(?:Apr(?:(?:\D){0,2}))|(?:May)|(?:Jun(?:(?:\D){0,1}))|(?:Jul(?:(?:\D){0,1}))|(?:Aug(?:(?:\D){0,4}))|(?:Sep(?:(?:\D){0,6}))|(?:Oct(?:(?:\D){0,4}))|(?:Nov(?:(?:\D){0,5}))|(?:Dec(?:(?:\D){0,6}))))(?:(?:[^\d]){0,5}\d.(?:[^\d]){0,6}\d\d\d\d))|((?:(?:\d\d)|(?:[^\d]\d))(?:.){0,2}(?:(?:(?:January)|(?:February)|(?:March)|(?:April)|(?:May)|(?:June)|(?:July)|(?:August)|(?:September)|(?:October)|(?:November)|(?:December))|(?:(?:Jan(?:(?:\D){0,4}))|(?:Feb(?:(?:\D){0,5}))|(?:Mar(?:(?:\D){0,2}))|(?:Apr(?:(?:\D){0,2}))|(?:May)|(?:Jun(?:(?:\D){0,1}))|(?:Jul(?:(?:\D){0,1}))|(?:Aug(?:(?:\D){0,4}))|(?:Sep(?:(?:\D){0,6}))|(?:Oct(?:(?:\D){0,4}))|(?:Nov(?:(?:\D){0,5}))|(?:Dec(?:(?:\D){0,6}))))(?:(?:[^\d]){0,5}\d.(?:[^\d]){0,6}\d\d\d\d)))|(?:(?:(?:(?:(?:January)|(?:February)|(?:March)|(?:April)|(?:May)|(?:June)|(?:July)|(?:August)|(?:September)|(?:October)|(?:November)|(?:December))|(?:(?:Jan(?:(?:\D){0,4}))|(?:Feb(?:(?:\D){0,5}))|(?:Mar(?:(?:\D){0,2}))|(?:Apr(?:(?:\D){0,2}))|(?:May)|(?:Jun(?:(?:\D){0,1}))|(?:Jul(?:(?:\D){0,1}))|(?:Aug(?:(?:\D){0,4}))|(?:Sep(?:(?:\D){0,6}))|(?:Oct(?:(?:\D){0,4}))|(?:Nov(?:(?:\D){0,5}))|(?:Dec(?:(?:\D){0,6}))))(?:[^\d]){0,5}\d.(?:[^\d]){0,6}\d\d\d\d(?:\n{0,3}))(?:.){0,5}(?:Date(?:[^\d]){0,5}of(?:[^\d]){0,20}report))))"
COMPILED_DATE_OF_REPORT_PATTERN = re.compile(
    DATE_OF_REPORT_PATTERN, re.I | re.MULTILINE | re.X | re.DOTALL
)
ITEMS_8K = {
    "Item1.01": r"Item(?:.){0,2}1\.01(?:.){0,2}Entry(?:.){0,2}into(?:.){0,2}a(?:.){0,2}Material(?:.){0,2}Definitive(?:.){0,2}Agreement",
    "Item1.02": r"Item(?:.){0,2}1\.02(?:.){0,2}Termination(?:.){0,2}of(?:.){0,2}a(?:.){0,2}Material(?:.){0,2}Definitive(?:.){0,2}Agreement",
    "Item1.03": r"Item(?:.){0,2}1\.03(?:.){0,2}Bankruptcy(?:.){0,2}or(?:.){0,2}Receivership",
    "Item1.04": r"Item(?:.){0,2}1\.04(?:.){0,2}Mine(?:.){0,2}Safety",
    "Item2.01": r"Item(?:.){0,2}2\.01(?:.){0,2}Completion(?:.){0,2}of(?:.){0,2}Acquisition(?:.){0,2}or(?:.){0,2}Disposition(?:.){0,2}of(?:.){0,2}Assets",
    "Item2.02": r"Item(?:.){0,2}2\.02(?:.){0,2}Results(?:.){0,2}of(?:.){0,2}Operations(?:.){0,2}and(?:.){0,2}Financial(?:.){0,2}Condition",
    "Item2.03": r"Item(?:.){0,2}2\.03(?:.){0,2}Creation(?:.){0,2}of(?:.){0,2}a(?:.){0,2}Direct(?:.){0,2}Financial(?:.){0,2}Obligation(?:.){0,2}or(?:.){0,2}an(?:.){0,2}Obligation(?:.){0,2}under(?:.){0,2}an(?:.){0,2}Off-Balance(?:.){0,2}Sheet(?:.){0,2}Arrangement(?:.){0,2}of(?:.){0,2}a(?:.){0,2}Registrant",
    "Item2.04": r"Item(?:.){0,2}2\.04(?:.){0,2}Triggering(?:.){0,2}Events(?:.){0,2}That(?:.){0,2}Accelerate(?:.){0,2}or(?:.){0,2}Increase(?:.){0,2}a(?:.){0,2}Direct(?:.){0,2}Financial(?:.){0,2}Obligation(?:.){0,2}or(?:.){0,2}an(?:.){0,2}Obligation(?:.){0,2}under(?:.){0,2}an(?:.){0,2}Off-Balance(?:.){0,2}Sheet(?:.){0,2}Arrangement",
    "Item2.05": r"Item(?:.){0,2}2\.05(?:.){0,2}Costs(?:.){0,2}Associated(?:.){0,2}with(?:.){0,2}Exit(?:.){0,2}or(?:.){0,2}Disposal(?:.){0,2}Activities",
    "Item2.06": r"Item(?:.){0,2}2\.06(?:.){0,2}Material(?:.){0,2}Impairments",
    "Item3.01": r"Item(?:.){0,2}3\.01(?:.){0,2}Notice(?:.){0,2}of(?:.){0,2}Delisting(?:.){0,2}or(?:.){0,2}Failure(?:.){0,2}to(?:.){0,2}Satisfy(?:.){0,2}a(?:.){0,2}Continued(?:.){0,2}Listing(?:.){0,2}Rule(?:.){0,2}or(?:.){0,2}Standard;(?:.){0,2}Transfer(?:.){0,2}of(?:.){0,2}Listing",
    "Item3.02": r"Item(?:.){0,2}3\.02(?:.){0,2}Unregistered(?:.){0,2}Sales(?:.){0,2}of(?:.){0,2}Equity(?:.){0,2}Securities",
    "Item3.03": r"Item(?:.){0,2}3\.03(?:.){0,2}Material(?:.){0,2}Modification(?:.){0,2}to(?:.){0,2}Rights(?:.){0,2}of(?:.){0,2}Security(?:.){0,2}Holders",
    "Item4.01": r"Item(?:.){0,2}4\.01(?:.){0,2}Changes(?:.){0,2}in(?:.){0,2}Registrant's(?:.){0,2}Certifying(?:.){0,2}Accountant",
    "Item4.02": r"Item(?:.){0,2}4\.02(?:.){0,2}Non-Reliance(?:.){0,2}on(?:.){0,2}Previously(?:.){0,2}Issued(?:(?:.){0,2} | \.)((Financial(?:.){0,2}Statements)|(Related(?:.){0,2}Audit(?:.){0,2}Report)|(Completed(?:.){0,2}Interim(?:.){0,2}Review))",
    "Item5.01": r"Item(?:.){0,2}5\.01(?:.){0,2}Changes(?:.){0,2}in(?:.){0,2}Control(?:.){0,2}of(?:.){0,2}Registrant",
    "Item5.02": r"Item(?:.){0,2}5\.02(?:.){0,2}(?:(Departure(?:.){0,2}of(?:.){0,2}Directors(?:.){0,2}or(?:.){0,2}Certain(?:.){0,2}Officers)|(.Election(?:.){0,2}of(?:.){0,2}Directors)|(.Appointment(?:.){0,2}of(?:.){0,2}Certain(?:.){0,2}Officers)|(.Compensatory(?:.){0,2}Arrangements(?:.){0,2}of(?:.){0,2}Certain(?:.){0,2}Officers))",
    "Item5.03": r"Item(?:.){0,2}5\.03(?:.){0,2}Amendments(?:.){0,2}to(?:.){0,2}Articles(?:.){0,2}of(?:.){0,2}Incorporation(?:.){0,2}or(?:.){0,2}Bylaws;(?:.){0,2}Change(?:.){0,2}in(?:.){0,2}Fiscal(?:.){0,2}Year",
    "Item5.04": r"Item(?:.){0,2}5\.04(?:.){0,2}Temporary(?:.){0,2}Suspension(?:.){0,2}of(?:.){0,2}Trading(?:.){0,2}Under(?:.){0,2}Registrant's(?:.){0,2}Employee(?:.){0,2}Benefit(?:.){0,2}Plans",
    "Item5.05": r"Item(?:.){0,2}5\.05(?:.){0,2}Amendment(?:.){0,2}to(?:.){0,2}Registrant's(?:.){0,2}Code(?:.){0,2}of(?:.){0,2}Ethics,(?:.){0,2}or(?:.){0,2}Waiver(?:.){0,2}of(?:.){0,2}a(?:.){0,2}Provision(?:.){0,2}of(?:.){0,2}the(?:.){0,2}Code(?:.){0,2}of(?:.){0,2}Ethics",
    "Item5.06": r"Item(?:.){0,2}5\.06(?:.){0,2}Change(?:.){0,2}in(?:.){0,2}Shell(?:.){0,2}Company(?:.){0,2}Status",
    "Item5.07": r"Item(?:.){0,2}5\.07(?:.){0,2}Submission(?:.){0,2}of(?:.){0,2}Matters(?:.){0,2}to(?:.){0,2}a(?:.){0,2}Vote(?:.){0,2}of(?:.){0,2}Security(?:.){0,2}Holders",
    "Item5.08": r"Item(?:.){0,2}5\.08(?:.){0,2}Shareholder(?:.){0,2}Director(?:.){0,2}Nominations",
    "Item6.01": r"Item(?:.){0,2}6\.01(?:.){0,2}ABS(?:.){0,2}Informational(?:.){0,2}and(?:.){0,2}Computational(?:.){0,2}Material",
    "Item6.02": r"Item(?:.){0,2}6\.02(?:.){0,2}Change(?:.){0,2}of(?:.){0,2}Servicer(?:.){0,2}or(?:.){0,2}Trustee",
    "Item6.03": r"Item(?:.){0,2}6\.03(?:.){0,2}Change(?:.){0,2}in(?:.){0,2}Credit(?:.){0,2}Enhancement(?:.){0,2}or(?:.){0,2}Other(?:.){0,2}External(?:.){0,2}Support",
    "Item6.04": r"Item(?:.){0,2}6\.04(?:.){0,2}Failure(?:.){0,2}to(?:.){0,2}Make(?:.){0,2}a(?:.){0,2}Required(?:.){0,2}Distribution",
    "Item6.05": r"Item(?:.){0,2}6\.05(?:.){0,2}Securities(?:.){0,2}Act(?:.){0,2}Updating(?:.){0,2}Disclosure",
    "Item7.01": r"Item(?:.){0,2}7\.01(?:.){0,2}Regulation(?:.){0,2}FD(?:.){0,2}Disclosure",
    "Item8.01": r"Item(?:.){0,2}8\.01(?:.){0,2}Other(?:.){0,2}Events",
    "Item9.01": r"Item(?:.){0,2}9\.01(?:.){0,2}Financial(?:.){0,2}Statements(?:.){0,2}and(?:.){0,2}Exhibits",
}

ITEMS_SC13D = {
    "Security and Issuer": r"(Item(?:.|\n){0,2}1\.)",
    "Identity and Background": r"(Item(?:.|\n){0,2}2\.)",
    "Source and Amount of Funds or Other Consideration": r"(Item(?:.|\n){0,2}3\.)",
    "Purpose of Transaction": r"(Item(?:.|\n){0,2}4\.)",
    "Interest in Securities of the Issuer": r"(Item(?:.|\n){0,2}5\.)",
    "Contracts, Arrangements, Understandings or Relationships with Respect to Securities of the Issuer": r"(Item(?:.|\n){0,2}6\.)",
    "Material to Be Filed as Exhibits": r"(Item(?:.|\n){0,2}7\.)",
}

ITEMS_SC13G = {
    "Issuer": r"(Item(?:.){0,2}2)\.(.*$\n?(?:^Issuer.*$)?)",
    "Filing Person": r"(Item(?:.){0,2}1)\.(.*$\n?(?:^Filing(?:\s){0,2}Person.*$)?)",
    "Checkboxes": r"(Item(?:.){0,2}3)\.(.*$\n?(?:^If(?:\s){0,2}this(?:\s){0,2}statement(?:\s){0,2}is(?:\s){0,2}filed(?:\s){0,2}pursuant(?:\s){0,2}to(?:\s){0,2}.*$)?)",
    "Ownership": r"(Item(?:.){0,2}4)\.(.*$\n?(?:^Ownership.*$)?)",
    "Five or less": r"(Item(?:.){0,2}5)\.(.*$\n?(?:Ownership(?:\s){0,2}of(?:\s){0,2}Five(?:\s){0,2}Percent(?:\s){0,2}or(?:\s){0,2}Less.*$)?)",
    "Five or more": r"(Item(?:.){0,2}6)\.(.*$\n?(?:^Ownership(?:\s){0,2}of(?:\s){0,2}More(?:\s){0,2}than(?:\s){0,2}Five(?:\s){0,2}Percent.*$)?)",
    "Subsidiary": r"(Item(?:.){0,2}7)\.(.*$\n?(?:^Identification(?:\s){0,2}and(?:\s){0,2}Classification(?:\s){0,2}of.*$)?)",
    "Members of group": r"(Item(?:.){0,2}8)\.(.*$\n?(?:^	Identification(?:\s){0,2}and(?:\s){0,2}Classification(?:\s){0,2}of.*$)?)",
    "Dissolution": r"(Item(?:.){0,2}9)\.(.*$\n?(?:^Notice(?:\s){0,2}of(?:\s){0,2}Dissolution.*$)?)",
    "Certifications": r"(Item(?:.){0,2}10)\.(.*$\n?(?:^Certification.*$)?)",
}

MAIN_TABLE_ITEMS_SC13G = {
    "1": r"(Name(?:s)?(?:\s){0,2}of(?:\s){0,2}Reporting(?:\s){0,2}Person(?:s)?)(?::)?",
    "2": r"(Check(?:\s){0,2}the(?:\s){0,2}appropriate(?:\s){0,2}box(?:\s){0,2}if(?:\s){0,2}a(?:\s){0,2}member(?:\s){0,2}of(?:\s){0,2}a(?:\s){0,2}Group(?:\s){0,2}(?:\(see(?:\s){0,2}instructions\))?)(?::)?",
    "3": r"(Sec(?:\s){0,2}Use(?:\s){0,2}Only)(?::)?",
    "4": r"(Citizenship(?:\s){0,2}or(?:\s){0,2}Place(?:\s){0,2}of(?:\s){0,2}Organization)(?::)?",
    "5": r"(Sole(?:\s){0,2}Voting(?:\s){0,2}Power)(?::)?",
    "6": r"(Shared(?:\s){0,2}Voting(?:\s){0,2}Power)(?::)?",
    "7": r"(Sole(?:\s){0,2}Dispositive(?:\s){0,2}Power)(?::)?",
    "8": r"(Shared(?:\s){0,2}Dispositive(?:\s){0,2}Power)(?::)?",
    "9": r"(Aggregate(?:\s){0,2}Amount(?:\s){0,2}Beneficially(?:\s){0,2}Owned(?:\s){0,2}by(?:\s){0,2}Each(?:\s){0,2}Reporting Person)(?::)?",
    "10": r"(Check(?:\s){0,2}(?:Box)?(?:\s){0,2}if(?:\s){0,2}the(?:\s){0,2}aggregate(?:\s){0,2}amount(?:\s){0,2}in(?:\s){0,2}row(?:\s){0,2}\(?9\)?(?:\s){0,2}excludes(?:\s){0,2}certain(?:\s){0,2}shares(?:\s){0,2}(?:\(see(?:\s){0,2}instructions\))?)(?::)?",
    "11": r"(Percent(?:\s){0,2}of(?:\s){0,2}class(?:\s){0,2}represented(?:\s){0,2}by(?:\s){0,2}amount(?:\s){0,2}in(?:\s){0,2}row(?:\s){0,2}\(?9\)?)(?::)?",
    "12": r"(Type(?:\s){0,2}of(?:\s){0,2}Reporting(?:\s){0,2}Person(?:\s){0,2}(?:\(see(?:\s){0,2}instructions\))?)(?::)?",
}

MAIN_TABLE_ITEMS_SC13D = {
    "1": r"(Name(?:s)?(?:\s){0,2}of(?:\s){0,2}Reporting(?:\s){0,2}Person(?:s)?)(?::)?",
    "2": r"(Check(?:\s){0,2}the(?:\s){0,2}Appropriate(?:\s){0,2}Box(?:\s){0,2}if(?:\s){0,2}a(?:\s){0,2}Member(?:\s){0,2}of(?:\s){0,2}a(?:\s){0,2}Group(?:\s){0,2}(?:\(see(?:\s){0,2}instructions\))?)(?::)?",
    "3": r"(SEC(?:\s){0,2}Use(?:\s){0,2}Only)(?::)?",
    "4": r"(Source(?:\s){0,2}of(?:\s){0,2}Funds(?:\s){0,2}(?:\(see(?:\s){0,2}instructions\))?)(?::)?",
    "5": r"(Check(?:\s){0,2}(?:Box)?(?:\s){0,2}if(?:\s){0,2}Disclosure(?:\s){0,2}of(?:\s){0,2}Legal(?:\s){0,2}Proceedings(?:\s){0,2}Is(?:\s){0,2}Required(?:\s){0,2}Pursuant(?:\s){0,2}to(?:\s){0,2}(?:.){,30}2\(e\))(?::)?",
    "6": r"(Citizenship(?:\s){0,2}or(?:\s){0,2}Place(?:\s){0,2}of(?:\s){0,2}Organization)(?::)?",
    "7": r"(Sole(?:\s){0,2}Voting(?:\s){0,2}Power)(?::)?",
    "8": r"(Shared(?:\s){0,2}Voting(?:\s){0,2}Power)(?::)?",
    "9": r"(Sole(?:\s){0,2}Dispositive(?:\s){0,2}Power)(?::)?",
    "10": r"(Shared(?:\s){0,2}Dispositive(?:\s){0,2}Power)(?::)?",
    "11": r"(Aggregate(?:\s){0,2}Amount(?:\s){0,2}Beneficially(?:\s){0,2}Owned(?:\s){0,2}by(?:\s){0,2}Each(?:\s){0,2}Reporting Person)(?::)?",
    "12": r"(Check(?:\s){0,2}(?:Box)?(?:\s){0,2}if(?:\s){0,2}the(?:\s){0,2}Aggregate(?:\s){0,2}Amount(?:\s){0,2}in(?:\s){0,2}Row(?:\s){0,2}\(11\)(?:\s){0,2}Excludes(?:\s){0,2}Certain(?:\s){0,2}Shares(?:\s){0,2}(?:\(see(?:\s){0,2}instructions\))?)(?::)?",
    "13": r"(Percent(?:\s){0,2}of(?:\s){0,2}Class(?:\s){0,2}Represented(?:\s){0,2}by(?:\s){0,2}Amount(?:\s){0,2}in(?:\s){0,2}Row(?:\s){0,2}\(11\))(?::)?",
    "14": r"(Type(?:\s){0,2}of(?:\s){0,2}Reporting(?:\s){0,2}Person(?:\s){0,2}(?:\(see(?:\s){0,2}instructions\))?)(?::)?",
}

REGISTRATION_TABLE_HEADERS_S3 = [
    re.compile(r"Amount(\s*)of(\s*)Registration(\s*)Fee", re.I),
    re.compile(r"Title(\s*)of(\s*)Each(\s*)Class(\s*)(.*)Regi(.*)", re.I),
    re.compile(r"(Amount(\s*)(Being|to(\s*)be)(\s*)Registered)|(Proposed\s(?:Maximum)?\s*(?:Aggregate)?\s*Offering\s*Price)", re.I),
]

REQUIRED_TOC_ITEMS_S3 = [
    # re.compile("(ABOUT(\s)*THIS(\s)*PROSPECTUS) | (PROSPECTUS(\s)*SUMMARY)", re.I),
    # re.compile("RISK(\s*)FACTORS(\s*)", re.I),
    re.compile(r"(PLAN(\s*)OF(\s*)DISTRIBUTION)|(Rescission(\s*)Offer)|(legal\s*matters)", re.I),
    re.compile(r"USE(\s*)OF(\s*)PROCEEDS", re.I),
]

TOC_ALTERNATIVES = {
    "principal stockholders": [
        "SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT"
    ],
    "index to financial statements": ["INDEX TO CONSOLIDATED FINANCIAL STATEMENTS"],
}

IGNORE_HEADERS_BASED_ON_STYLE = set("TABLE OF CONTENTS")
HEADERS_TO_DISCARD = ["(unaudited)"]

RE_COMPILED = {
    "two_newlines_or_more": re.compile(r"(\n){2,}", re.MULTILINE),
    "one_newline": re.compile(r"(\n)", re.MULTILINE),
    "two_spaces_or_more": re.compile(r"(\s){2,}", re.MULTILINE),
}


def _add_unique_id_to_dict(target: dict, key: str = "UUID"):
    """Add a UUID to the target.

    Returns:
        a modified copy of target with a key that contains a string of a uuid4
    """
    target_copy = target.copy()
    target_copy[key] = str(uuid.uuid4())
    return target_copy


class AbstractFilingParser(ABC):
    @property
    @abstractmethod
    def form_type(cls):
        raise NotImplementedError

    @property
    @abstractmethod
    def extension(cls):
        raise NotImplementedError
    
    def parse(self, path: str):
        '''
        Convenince Method.
        
        Returns:
            list[FilingSection]
        '''
        return self.split_into_sections(self.get_doc(path))


    def split_into_sections(self, doc):
        """
        Must be implemented by the subclass.
        Should split the filing into logical sections.

        Returns:
            list[FilingSection]"""
        pass

    def get_doc(self, path: str):
        """
        opens the file the correct way and returns the type required
        by split_into_sections.
        """
        pass


class ParserFactory:
    """helper factory to get the correct parser for form_type and extension, when creating Filings"""

    def __init__(self, defaults: list[tuple] = [], default_fallbacks: bool = True):
        self.parsers = {}
        if default_fallbacks is True:
            fallbacks = [(".htm", None, HTMFilingParser)]
            for each in fallbacks:
                self.register_parser(*each)
        if defaults != []:
            for each in defaults:
                self.register_parser(*each)

    def register_parser(
        self, extension: str, form_type: str, parser: AbstractFilingParser
    ):
        self.parsers[(extension, form_type)] = parser

    def get_parser(self, extension: str, form_type: str = None, **kwargs):
        parser = self.parsers.get((extension, form_type))
        if parser:
            return parser(**kwargs)
        else:
            parser = self.parsers.get((extension, None))
            if parser:
                return parser()
            else:
                raise ValueError(
                    f"no parser for combination of extension, form_type ({extension, form_type}) registered."
                )


class FilingFactory:
    def __init__(self, default_fallbacks=False, defaults: list[tuple] = []):
        self.builders = {}
        if default_fallbacks is True:
            self.init_fallbacks()
        if len(defaults) > 0:
            for case in defaults:
                self.register_builder(*case)

    def init_fallbacks(self):
        """
        register fallbacks for form_types that werent added
        but mostlikely work with the default implementation of
        the extension provided
        """
        self.register_builder(None, ".htm", BaseHTMFiling)

    def register_builder(
        self, form_type: str, extension: str, builder: Filing | Callable
    ):
        """register a new builder for a combination of (form_type, extension)"""
        self.builders[(form_type, extension)] = builder

    def create_filing(self, form_type: str, extension: str, **kwargs):
        """try and get a builder for given args and create the Filing"""
        logger.debug(
            f"args passed to create_filing: {form_type, extension}, kwargs: {kwargs}"
        )
        builder = self.builders.get((form_type, extension))
        if builder:
            logger.debug(f"using builder: {builder} with kwargs: {kwargs}")
            return builder(form_type=form_type, extension=extension, **kwargs)
        else:
            builder = self.builders.get((None, extension))
            if builder:
                logger.info(
                    f"Using a fallback value for the builder as no builder was registered for the given combination ({form_type},{extension}). fallback builder used: {builder}"
                )
                return builder(form_type=form_type, extension=extension, **kwargs)
            else:
                raise ValueError(
                    f"no parser for that form_type and extension combination({form_type}, {extension}) registered"
                )


class HTMFilingParser(AbstractFilingParser):
    form_type = None
    extension = ".htm"
    """
    Baseclass for parsing HtmlFilings.

    Usage::

            from bs4 import Beautifulsoup
            parser = HtmlFilingParser()
            doc = parser.get_doc(path_to/filing.htm)
            sections = parser.split_into_sections(doc)
        
    """

    def __init__(self):
        pass

    def get_doc(self, path: str):
        """opens the file the correct way and returns the filing as string."""
        with open(Path(path), "r", encoding="utf-8") as f:
            doc = f.read()
            return self.preprocess_doc(doc)

    def extract_tables(
        self, soup: BeautifulSoup, reintegrate=["ul_bullet_points", "one_row_table"]
    ):
        """
        extract the tables, parse them and return them as a dict of nested lists.

        side effect: modifies the soup attribute if reintegrate isnt an empty list

        Args:
            soup: BeautifulSoup of content
            reintegrate: which tables to reintegrate as text. if this is an empty list
                         all tables will be returned in the "extracted" section of the dict
                         and the "reintegrated" section will be an empty list
        Returns:
            a dict of form: {
                "reintegrated": [
                        {
                        "classification": classification,
                        "reintegrated_as": new elements that were added to the original doc
                        "table_meta": meta data dict for the extracted tables. contains at least
                                      a key of 'table_elements' which contains the original <table>
                                      elements as a list.
                        "parsed_table": parsed representation of the table before reintegration
                        }
                    ],
                "extracted": [
                        {
                        "classification": classification,
                        "table_meta": meta data dict for the extracted tables. contains at least
                                      a key of 'table_elements' which contains the original <table>
                                      elements as a list.
                        "parsed_table": parsed representation of table,
                        }
                    ]
                }
        """
        unparsed_tables = self.get_unparsed_tables(soup)
        tables = {"reintegrated": [], "extracted": []}
        for t in unparsed_tables:
            parsed_table = None
            if self.table_has_header(t):
                parsed_table = self.parse_htmltable_with_header(t)
            else:
                parsed_table = self.primitive_htmltable_parse(t)
            cleaned_table = self._clean_parsed_table_columnwise(
                self._preprocess_table(parsed_table)
            )
            classification = self.classify_table(cleaned_table)
            if classification in reintegrate:
                reintegrate_html = self._make_reintegrate_html_of_table(
                    classification, cleaned_table
                )
                t.replace_with(reintegrate_html)
                tables["reintegrated"].append(
                    {
                        "classification": classification,
                        "reintegrated_as": reintegrate_html,
                        "table_meta": {"table_elements": [t]},
                        "parsed_table": cleaned_table,
                    }
                )
            else:
                # further reformat the table
                tables["extracted"].append(
                    {
                        "classification": classification,
                        "table_meta": {"table_elements": [t]},
                        "parsed_table": cleaned_table,
                    }
                )
        return tables

    def get_text_content(self, doc: BeautifulSoup = None, exclude=["table", "script"], strip=True):
        """extract the unstructured language"""
        doc_copy = copy.copy(doc)
        if exclude != [] or exclude is not None:
            [s.extract() for s in doc_copy(exclude)]
        return doc_copy.get_text(separator=" ", strip=strip)

    def get_span_of_element(self, doc: str, ele: element.Tag, pos: int = None):
        """gets the span (start, end) of the element ele in doc

        Args:
            doc: html string"""
        exp = re.compile(re.escape(str(ele)))
        span = exp.search(doc, pos=0 if pos is None else pos).span()
        if not span:
            raise ValueError("span of element couldnt be found")
        else:
            return span

    def find_next_by_position(self, doc: str, start_ele: element.Tag, filter):
        if not isinstance(filter, (bool, NoneType)) and not callable(filter):
            raise ValueError(
                "please pass a function, bool, or None for the filter arg to find_next_by_position"
            )
        after_pos = self.get_span_of_element(doc, start_ele)[1]
        for ele in start_ele.next_elements:
            ele_pos = self.get_span_of_element(doc, ele)[0]
            if after_pos < ele_pos:
                if filter is True:
                    return ele
                if filter is None:
                    return ele
                if isinstance(filter, function):
                    if filter(ele) is True:
                        return ele
        return None

    def split_into_sections(self, doc: BeautifulSoup | str):
        if isinstance(doc, str):
            doc_ = BeautifulSoup(doc, features="html5lib")
        else:
            doc_ = doc
        try:
            sections = self.split_by_table_of_contents(doc_)
            if sections is not None:
                return sections
        except ValueError as e:
            # logging.debug(e, exc_info=True)
            pass
        except AttributeError as e:
            # logging.debug(e, exc_info=True)
            pass
        logger.debug(f"sections1: {len(sections) if sections is not None else []}")
        # print()
        possible_headers = self._get_possible_headers_based_on_style(
            doc=doc_, ignore_toc=False
        )
        headers_style = self._format_matches_based_on_style(possible_headers)
        logger.debug(
            f"headers by style: {[s['full_norm_text'] for s in headers_style]}"
        )
        section_start_elements = [
            {"section_title": k["full_norm_text"], "ele": k["elements"][0]}
            for k in headers_style
        ]
        logger.debug(
            f"split_into_sections(): section_start_elements for splitting by style: {section_start_elements}"
        )
        sections = self._split_into_sections_by_tags(
            doc_, section_start_elements=section_start_elements
        )
        logger.debug(f"sections2: {len(sections) if sections is not None else []}")
        if (sections is None) or (sections == []):
            raise NotImplementedError(
                "no way to split into sections is implemented for this case"
            )
        else:
            return sections

    def split_by_table_of_contents(self, doc: BeautifulSoup):
        """split a filing with a TOC into sections based on the TOC.

        Args:
            doc: html parsed with bs4

        Returns:
            list[HTMFilingSection]
        """
        # try and split by document by hrefs of the toc
        try:
            sections = self._split_by_table_of_contents_based_on_hrefs(doc)
        except AttributeError as e:
            # couldnt find toc
            logger.info(
                (
                    "Split_by_table_of_content: Filing doesnt have a TOC or it couldnt be determined",
                    e,
                ),
                exc_info=True,
            )
            return None
        except ValueError as e:
            # couldnt find hrefs in the toc so lets continue with different strategy
            logger.info(e, exc_info=True)
            pass
        else:
            return sections
        try:
            sections = self._split_by_table_of_content_based_on_headers(doc)
        except AttributeError as e:
            logger.info(
                (
                    "_Split_by_table_of_content_based_on_headers: Filing doesnt have a TOC or it couldnt be determined",
                    e,
                ),
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.debug(e, exc_info=True)
            return None
        return sections

    def preprocess_text(self, text):
        """removes common unicode and folds multi spaces/newlines into one"""
        text = self.replace_common_unicode(text)
        # fold multiple empty newline rows into one
        text = re.sub(RE_COMPILED["two_newlines_or_more"], "\n", text)
        text = re.sub(RE_COMPILED["one_newline"], " ", text)
        # fold multiple spaces into one
        text = re.sub(RE_COMPILED["two_spaces_or_more"], " ", text)
        return text
    
    def replace_common_unicode(self, text: str) -> str:
        return text.replace("\xa04", "").replace("\xa0", " ").replace("\u200b", " ")

    def preprocess_doc(self, doc: str):
        """preprocess the html string, by converting it to Beautifulsoup and back,
        thereby converting common html entities."""
        return str(BeautifulSoup(doc, features="html5lib"))

    def get_clean_text_from_raw_htm(self, filing: str|BeautifulSoup):
        """cleanes html filing and returns only text"""
        if isinstance(filing, BeautifulSoup):
            soup = filing
        else:
            soup = self.make_soup(filing)
        filing = self.get_text_content(soup, exclude=["title"])
        return self.preprocess_text(filing)

    def make_soup(self, doc: str):
        """creates a BeautifulSoup from string html"""
        soup = BeautifulSoup(doc, features="html5lib")
        return soup

    def get_unparsed_tables(self, soup: BeautifulSoup):
        """return all table tags in soup"""
        unparsed_tables = soup.find_all("table")
        return unparsed_tables

    def classify_table(self, table: list[list]):
        """'classify a table into subcategories so they can
        be processed further.

        Args:
            table: should be a list of lists cleaned with clean_parsed_table.
        """
        # assume header
        try:
            table_shape = (len(table), len(table[0]))
        except IndexError:
            return "empty"
        # logger.debug(f"table shape is: {table_shape}")
        # could be a bullet point table
        if table_shape[0] == 1:
            return "one_row_table"
        if table_shape[1] == 2:
            if self._is_bullet_point_table(table) is True:
                return "ul_bullet_points"
            else:
                return "unclassified"
        return "unclassified"

    # def _create_section_start_element(self, section_title: str, ele: element.Tag, meta: dict)

    def _preprocess_table(self, table: list[list]):
        """preprocess the strings in the table, removing multiple whitespaces and newlines
        Returns:
            a new (preprocessed) table with the same dimensions as the original
        """
        t = table.copy()
        for ridx, _ in enumerate(t):
            for cidx, _ in enumerate(t[ridx]):
                field_content = t[ridx][cidx]
                if isinstance(field_content, str):
                    t[ridx][cidx] = self.preprocess_text(field_content)
        return t

    def _clean_parsed_table_drop_empty_rows(
        self, table: list[list], remove_: list = ["", None]
    ):
        """clean a parsed table, removing all rows consisting only of fields in remove_"""
        drop_rows = []
        for idx, row in enumerate(table):
            if _row_is_ignore(row) is True:
                drop_rows.insert(0, idx)
        for d in drop_rows:
            table.pop(d)
        return table

    def _clean_parsed_table_fieldwise(
        self, table: list[list], remove_: list = ["", None, "None", " ", "\u200b"]
    ):
        """clean a parsed table of shape m,n removing all fields which are in remove_"""
        drop = []
        for ridx, row in enumerate(table):
            for fidx, field in enumerate(row):
                if field in remove_:
                    drop.insert(0, (ridx, fidx))
        for drop_ in drop:
            table[drop_[0]].pop(drop_[1])
        drop_row = []
        for ridx, row in enumerate(table):
            if row == []:
                drop_row.insert(0, ridx)
        for d in drop_row:
            table.pop(d)
        return table

    def _clean_parsed_table_columnwise(
        self, table: list[list], remove_identifier: list = ["", "None", None, " ", "\u200b"]
    ):
        """clean a parsed table of shape m,n by removing all columns whose row values are a combination of remove_identifier"""
        tablec = table.copy()
        nr_rows = len(tablec)
        nr_cols = len(tablec[0])
        boolean_matrix = [[True] * nr_cols for n in range(nr_rows)]
        for row in range(nr_rows):
            for col in range(nr_cols):
                if tablec[row][col] in remove_identifier:
                    boolean_matrix[row][col] = False
        cols_to_remove = []
        for col in range(nr_cols):
            all_empty = True
            for row in range(nr_rows):
                if boolean_matrix[row][col] is True:
                    all_empty = False
            if all_empty is True:
                cols_to_remove.insert(0, col)
        for rmv in cols_to_remove:
            for row in range(nr_rows):
                tablec[row].pop(rmv)
        return tablec

    def _is_bullet_point_table(self, table: list[list]):
        table_shape = (len(table), len(table[0]))
        if table_shape[1] != 2:
            return False
        bullet = "\u25cf"
        cols_unicode = [True] * table_shape[1]
        for col in range(table_shape[1]):
            for row in range(table_shape[0]):
                if table[row][col] not in ["\u25cf", "\u25cf \u25cf", "\u25cf\u25cf", "", ""]:
                    # print(table[row][col], row, col)
                    cols_unicode[col] = False
        if cols_unicode[0] is True:
            return True
        else:
            return False

    def _make_reintegrate_html_of_table(self, classification, table: list[list]):
        empty_soup = BeautifulSoup("", features="html5lib")
        base_element = empty_soup.new_tag("p")
        if classification == "ul_bullet_points":
            for idx, row in enumerate(table):
                ele = empty_soup.new_tag("span")
                ele.string = " ".join(row) + "\n"
                base_element.insert(idx + 2, ele)
            return base_element
        elif classification == "one_row_table":
            ele = empty_soup.new_tag("span")
            _string = ""
            for field in table[0]:
                _string += str(field) + "\t"
            ele.string = _string
            base_element.insert(0, ele)
            return base_element
        raise NotImplementedError(
            f"reintegration of this class of table hasnt been handled. classification: {classification}"
        )

    def get_element_text_content(self, ele):
        """gets the cleaned text content of a single element.Tag"""
        content = " ".join([s.strip().replace("\n", " ") for s in ele.strings]).strip()
        return content
    
    def _table_is_shape_and_field_length(self,
        table: list[list],
        shape_constraint: tuple[int, int] = (-1, -1),
        field_length_constraint: tuple[int] = (-1, 12)) -> bool:
        '''
        helper function to determine if the parsed table has the correct shape and field lengths.
        
        Args:
            table: list[list]
        '''
        if (len(field_length_constraint) > shape_constraint[1]) and (shape_constraint[1] != -1):
            raise ValueError(f"field_length_constraint elements can't exceed shape_constraint[1]")
        is_correct_shape = True
        if shape_constraint[0] != -1:
            if len(table) > shape_constraint[0]:
                is_correct_shape = False
        if shape_constraint[1] != -1:
            if len(table[0]) > shape_constraint[1]:
                is_correct_shape = False
        if is_correct_shape is False:
            return False
        has_correct_lengths = True
        for row in  table:
            for idx, col in enumerate(row):
                try:
                    if field_length_constraint[idx] == -1:
                        continue
                    else:
                        if len(col) > field_length_constraint[idx]:
                            has_correct_lengths = False
                except IndexError:
                    logger.debug("tried to access shape_constraint item which doesnt exist. excepted IndexError.")
        if (has_correct_lengths is True) and (is_correct_shape is True):
            return True


    def _parse_toc_table_element(self, table_element: element.Tag):
        """convert the toc table element into a list[dict].

        Returns:
            list[dict]: where each dict is a entry from the toc with keys:
                        title, href, page
            None: when the parsed and cleaned table doesnt conform to the field lengths of a TOC.
        """
        rows = table_element.find_all("tr")
        amount_rows = len(rows)
        amount_columns = 0
        table = None
        for row in rows:
            nr_columns = len(row.find_all("td"))
            if nr_columns > amount_columns:
                amount_columns = nr_columns
        # create empty table
        table = [[None] * amount_columns for each in range(amount_rows)]
        hrefs = []
        # write each row and keep separat list of hrefs for each row
        for row_idx, row in enumerate(rows):
            href = []
            for field_idx, field in enumerate(row.find_all("td")):
                content = self.get_element_text_content(field)
                table[row_idx][field_idx] = content if content else ""
            href = self.get_element_hrefs(row)
            if href != []:
                unseen = []
                for h in href:
                    if h not in hrefs:
                        unseen.append(h)
                if len(set(unseen)) > 1:
                    logger.debug(f"unhandled row: {row}")
                    logger.debug(f"unhandled field: {field}")
                    raise AttributeError(
                        f"parse_toc_table can only handle one href per toc row! More than one href found: {hrefs}"
                    )
                if unseen == []:
                    hrefs.append(None)
                    continue
                hrefs.append(unseen[0])
            else:
                hrefs.append(None)
        idx = 0
        print(len(table), len(hrefs))
        for href in hrefs:
            table[idx].append(href)
            idx += 1
        table = self._clean_parsed_table_columnwise(table)
        table = self._clean_parsed_table_fieldwise(table)
        drop_none_complete = []
        hrefs_set = set(hrefs)
        for ridx, row in enumerate(table):
            logger.debug(f"toc_row: {row}")
            if (len(hrefs_set) == 1) and (None in hrefs_set):
                if (len(row) != 2):
                    drop_none_complete.insert(0, ridx)
            else:
                if (len(row) != 3):
                    drop_none_complete.insert(0, ridx)
        for ridx in drop_none_complete:
            table.pop(ridx)
        if table == []:
            return None
        toc_table = []
        for field in table[0]:
            if re.search(re.compile("(descript.*)|(page)", re.I), field):
                table.pop(0)
                break
        if not self._table_is_shape_and_field_length(table, (-1, -1), (-1, 12, -1)):
            logger.debug(f"returning None. Discarded TOC because of incorrect field_lengths in the pages column: {table}")
            return None
        for row in table:
            title = row[0]
            try:
                page = row[1]
            except IndexError:
                page = None
            try:
                href = row[2]
            except IndexError:
                href = None
            toc_table.append({"title": title, "page": page, "href": href})
        return toc_table

    def get_element_hrefs(self, ele: element.Tag):
        hrefs = []
        containg_hrefs = ele.find_all(lambda x: x.has_attr("href"))
        if containg_hrefs:
            return [e["href"] for e in containg_hrefs]
        else:
            return []

    def primitive_htmltable_parse(self, htmltable):
        """parse simple html tables without col or rowspan"""
        rows = htmltable.find_all("tr")
        amount_rows = len(rows)
        amount_columns = 0
        table = None
        for row in rows:
            nr_columns = len(row.find_all("td"))
            if nr_columns > amount_columns:
                amount_columns = nr_columns
        # create empty table
        table = [[None] * amount_columns for each in range(amount_rows)]
        # write each row
        for row_idx, row in enumerate(rows):
            for field_idx, field in enumerate(row.find_all("td")):
                content = self.get_element_text_content(field)
                # adjust row_idx for header
                table[row_idx][field_idx] = content
        return table

    def parse_htmltable_with_header(
        self, htmltable, colspan_mode="separate", merge_delimiter=" "
    ):
        # parse the header
        header, colspans, body = self.parse_htmltable_header(htmltable)
        # get number of rows (assuming rowspan=1 for each row)
        rows = htmltable.find_all("tr")
        amount_rows = len(rows)
        amount_columns = None
        table = None

        # mode to write duplicate column names for colspan > 1
        if colspan_mode == "separate":
            # get total number columns including colspan
            amount_columns = sum([col for col in colspans])
            # create empty table
            table = [[None] * amount_columns for each in range(amount_rows)]
            # parse, write each row (ommiting the header)
            for row_idx, row in enumerate(rows[1:]):
                for field_idx, field in enumerate(row.find_all("td")):
                    content = self.get_element_text_content(field)
                    # adjust row_idx for header
                    table[row_idx + 1][field_idx] = content

        # mode to merge content of a colspan column under one unique column
        # splitting field content by the merge_delimiter
        elif colspan_mode == "merge":
            # get number of columns ignoring colspan
            amount_columns = sum([1 for col in colspans])
            # create empty table
            table = [[None] * amount_columns for each in range(amount_rows)]

            # parse, merge and write each row excluding header
            for row_idx, row in enumerate(rows[1:]):
                # get list of content in row
                unmerged_row_content = [
                    self.get_element_text_content(td) for td in row.find_all("td")
                ]
                merged_row_content = [None] * amount_columns
                # what we need to shift by to adjust for total of colspan > 1
                colspan_offset = 0
                for idx, colspan in enumerate(colspans):
                    unmerged_idx = idx + colspan_offset
                    merged_row_content[idx] = merge_delimiter.join(
                        unmerged_row_content[unmerged_idx : (unmerged_idx + colspan)]
                    )
                    colspan_offset += colspan - 1
                # write merged rows to table
                for r in range(amount_columns):
                    # adjust row_idx for header
                    table[row_idx + 1][r] = merged_row_content[r]

            # change colspans to reflect the merge and write header correctly
            colspans = [1] * len(colspans)

        # write header
        cspan_offset = 0
        for idx, h in enumerate(header):
            colspan = colspans[idx]
            if colspan > 1:
                for r in range(colspan - 1):
                    table[0][idx + cspan_offset] = h
                    cspan_offset = cspan_offset + 1

            else:
                table[0][idx + cspan_offset] = h

        return table

    def parse_htmltable_header(self, htmltable):
        parsed_header = []
        unparsed_header = []
        body = []
        colspans = []
        # first case with actual table header tag <th>
        if htmltable.find_all("th") != (None or []):
            for th in htmltable.find_all("th"):
                colspan = 1
                if "colspan" in th.attrs:
                    colspan = int(th["colspan"])
                parsed_header.append(self.get_element_text_content(th))
                colspans.append(colspan)
            body = [row for row in htmltable.find_all("tr")]
        # second case where first <tr> is header
        else:
            unparsed_header = htmltable.find("tr")
            for td in unparsed_header.find_all("td"):
                colspan = 1
                if "colspan" in td.attrs:
                    colspan = int(td["colspan"])
                parsed_header.append(self.get_element_text_content(td))
                colspans.append(colspan)
            body = [row for row in htmltable.find_all("tr")][1:]
        return parsed_header, colspans, body

    def preprocess_section_text_content(self, section_content: str):
        """cleanes section_content and returns only text with unwanted characters removed"""
        section_content.replace("\xa04", "")
        section_content.replace("\xa0", " ")
        section_content.replace("\u200b", " ")
        # fold multiple empty newline rows into one
        section_content = re.sub(
            re.compile(r"(\n){2,}", re.MULTILINE), "\n", section_content
        )
        
        # fold multiple spaces into one
        section_content = re.sub(
            re.compile(r"(\s){2,}", re.MULTILINE), " ", section_content
        )
        section_content = re.sub(re.compile(r"(?<!(\.|\?|!))(\n)(?![A-Z0-9])", re.MULTILINE), " ", section_content)
        section_content = re.sub(re.compile(r"(?<!(\.|\?|!)(\s))(\n)((\s)?![A-Z0-9])", re.MULTILINE), " ", section_content)
        section_content = re.sub(re.compile(r"(?<!(\.|\?|!)(\s))(\n)(?![A-Z0-9])", re.MULTILINE), " ", section_content)
        section_content = re.sub(re.compile(r"(?<!(\.|\?|!))(\n)((\s)?![A-Z0-9])", re.MULTILINE), " ", section_content)
        section_content = re.sub(
            re.compile(r"(\s){2,}", re.MULTILINE), " ", section_content
        )
        section_content = re.sub(
            re.compile(r"(\s)", re.MULTILINE), " ", section_content
        )
        return section_content

    def table_has_header(self, htmltable):
        has_header = False
        # has header if it has <th> tags in table
        if htmltable.find("th"):
            has_header = True
            return has_header
        else:
            # finding first <tr> element to check if it is empty
            # otherwise consider it a header
            possible_header = htmltable.find("tr")
            try:
                header_fields = [
                    self.get_element_text_content(td)
                    for td in possible_header.find_all("td")
                ]
            except AttributeError:
                return False
            for h in header_fields:
                if (h != "") and (h != []):
                    has_header = True
                    break
        return has_header

    def _get_colspan_of_element(self, element):
        if "attrs" not in element.__dict__:
            return 1
        if "colspan" in element.attrs:
            return int(element["colspan"])
        else:
            return 1

    def _get_rowspan_of_element(self, element):
        if "attrs" not in element.__dict__:
            return 1
        if "rowspan" in element.attrs:
            return int(element["rowspan"])
        else:
            return 1

    def _get_possible_headers_based_on_style(
        self,
        doc: BeautifulSoup,
        start_ele: element.Tag | BeautifulSoup = None,
        max_distance_multiline: int = 400,
        ignore_toc: bool = True,
    ):
        """look for possible headers based on styling of the elements.

        Ignores elements in the TOC table if found.

        approach:
            1) filter all common types of styles the headers could be in
            2) sort into mainheader(all caps) and subheader(not all caps)
            3) if we dont have a lot of mainheaders try and sort by font-size/size
               mostlikely need to check if parents have that property
               [NOT IMPLEMENTED]
            4) assign subheaders based on position relative to mainheaders
               eg: subheaders between two mainheaders are assigned to the earlier
               mainheader

        Args:
            start_ele: what element to select from [optional] if not selected start_ele will be doc

        Raises:
            ValueError: if the position of an element couldnt be determined
        """
        str_doc = str(doc)
        # weird start end of conesecutive elements (50k vs 800k ect)? why?
        toc_start_end = None
        if start_ele is None:
            start_ele = doc  # .find("title") if doc.find("title") else doc
        if ignore_toc is True:
            try:
                close_to_toc = doc.find(
                    string=re.compile(
                        "(toc|table..?.?of..?.?contents)", re.I | re.DOTALL
                    )
                )
                if isinstance(close_to_toc, NavigableString):
                    close_to_toc = close_to_toc.parent
                if "href" in close_to_toc.attrs:
                    name_or_id = close_to_toc["href"][-1]
                    close_to_toc = doc.find(True, {"name": name_or_id})
                    if close_to_toc is None:
                        close_to_toc = doc.find(True, {"id": name_or_id})
                toc_table = close_to_toc.find_next("table")
                found_toc = False
                while found_toc is False:
                    dirty_toc = self.parse_htmltable_with_header(toc_table)
                    if len(dirty_toc) < 10:
                        toc_table = toc_table.find_next("table")
                        if not toc_table:
                            print("couldnt find a toc")
                            after_toc = start_ele
                    else:
                        toc_start_end = self.get_span_of_element(str_doc, toc_table)
                        found_toc = True

                after_toc = self.find_next_by_position(str_doc, toc_table, True)
                logger.debug(
                    f"toc_table span: {self.get_span_of_element(str_doc, toc_table)}"
                )
                logger.debug(
                    f"after_toc span: {self.get_span_of_element(str_doc, after_toc)}"
                )
                # ignore_before = re.search(re.escape(str(after_toc)), str_doc).span()[1]

            except Exception as e:
                logger.debug(
                    "handle the following exception in the ignore_toc=True block of _get_possible_headers_based_on_style"
                )
                raise e

        # use i in css selector for case insensitive match
        style_textalign = [
            "[style*='text-align: center' i]",
            "[style*='text-align:center' i]",
        ]
        style_weight = [
            "[style*='font-weight: bold' i]",
            "[style*='font-weight:bold' i]",
            "[style*='font-weight:700']",
            "[style*='font-weight: 700']",
        ]
        style_font = ["[style*='font: bold' i]", "[style*='font:bold' i]"]
        attr_align = "[align='center' i]"
        # add a total count and thresholds for different selector groups
        # eg: we dont find enough textcenter_b and textcenter_eigth candidates go to next group
        selectors = {
            "center_b": [" ".join([attr_align, "b"])],
            "textcenter_strong": [" ".join([s, "> strong"]) for s in style_textalign],
            "textcenter_b": [" ".join([s, " b"]) for s in style_textalign],
            "textcenter_weight": sum(
                [
                    [" ".join([s, ">", w]) for s in style_textalign]
                    for w in style_weight
                ],
                [],
            ),
            "td_textcenter_font_b": [
                " ".join(["td" + s, "> font > b"]) for s in style_textalign
            ],
            "td_textcenter_font_weigth": sum(
                [
                    [" ".join(["td" + s, "> font" + w]) for s in style_textalign]
                    for w in style_weight
                ],
                [],
            ),
            "b_u": ["b > u"],
            "font_bold_text_align_center_parent": sum(
                [
                    [" ".join([a + f, "font"]) for a in style_textalign]
                    for f in style_font
                ],
                [],
            ),
        }

        matches = {}
        entry_to_ignore = set()
        multiline_matches = 0
        for name, selector in selectors.items():
            if isinstance(selector, list):
                match = {"main": [], "sub": []}
                for s in selector:
                    last_entry = None
                    ele_group = []
                    # group together elements that are within close range of each other (how many chars?)
                    # -> multiline headers

                    # try this
                    selected_elements = start_ele.select(selector=s)
                    elements = []
                    for idx, e in enumerate(selected_elements):
                        if idx > 0:
                            start_pos = elements[idx - 1][1][1]
                        else:
                            start_pos = 0
                        span = self.get_span_of_element(str_doc, e, pos=start_pos)
                        elements.append((e, span))
                    elements_sorted = sorted(elements, key=lambda x: x[1][1])

                    for entry in elements_sorted:
                        ele = entry[0]
                        if entry in entry_to_ignore:
                            continue
                        else:
                            entry_to_ignore.add(entry)
                        if toc_start_end is not None:
                            if self._ele_is_between(
                                str_doc, ele, toc_start_end[0], toc_start_end[1]
                            ):
                                # if (0 <= entry[1][0] <= toc_start_end[1]):
                                continue
                        text_content = (
                            ele.string
                            if ele.string
                            else " ".join([s for s in ele.strings])
                        )
                        t_is_upper = (
                            text_content.isupper()
                            if len(text_content.split()) < 2
                            else " ".join(text_content.split()[:1]).isupper()
                        )
                        if t_is_upper:
                            if last_entry is None:
                                last_entry = entry
                                logger.debug(f"entry (last_entry was None): {entry}")
                            else:
                                logger.debug(
                                    f"entry: {entry} \t last_entry: {last_entry} \t distance: {(entry[1][0] - last_entry[1][1])}"
                                )
                                if (
                                    entry[1][0] - last_entry[1][1]
                                ) <= max_distance_multiline:
                                    if ele_group == []:
                                        ele_group.append(last_entry)
                                        entry_to_ignore.add(entry)
                                    if entry != last_entry:
                                        ele_group.append(entry)
                                    last_entry = entry
                                    continue
                                else:
                                    # finished possible multline title
                                    logger.debug(f"ELE_GROUP (main): {ele_group}")
                                    if ele_group != []:
                                        if len(ele_group) > 1:
                                            match["main"].append([e for e in ele_group])
                                            multiline_matches += 1
                                        else:
                                            match["main"].append(ele_group[0])
                                        ele_group = [entry]
                                    else:
                                        if entry not in match["main"]:
                                            match["main"].append(entry)
                                            logger.debug(
                                                "entry was missing from main so we added it"
                                            )
                                        if last_entry not in match["main"]:
                                            logger.debug(
                                                "last_entry was missing from main so we added it"
                                            )
                                            match["main"].append(last_entry)
                                    last_entry = None
                                    continue
                            last_entry = entry
                        else:
                            match["sub"].append(entry)
                            if (last_entry is not None) and (
                                last_entry not in ele_group
                            ):
                                match["main"].append(last_entry)
                                last_entry = None
                            if ele_group != []:
                                logger.debug(f"ELE_GROUP (sub): {ele_group}")
                                match["main"].append([e for e in ele_group])
                                ele_group = []

                    if ele_group == []:
                        if last_entry is not None:
                            match["main"].append(last_entry)
                    else:
                        # append ele group
                        match["main"].append([e for e in ele_group])
                        multiline_matches += 1
                matches[name] = match
            else:
                raise TypeError(
                    f"selectors should be wrapped in a list: got {type(selector)}"
                )
            # logger.debug(f"found {multiline_matches} multiline matches")
        return matches

    def _format_matches_based_on_style(self, matches, header_style="main"):
        formatted_matches = []
        for k, v in matches.items():
            for i in v[header_style]:
                if i:
                    if isinstance(i, list):
                        pos = i[0][1]
                        full_text = self._normalize_toc_title(
                            " ".join(
                                sum(
                                    [
                                        [
                                            i[idx][0].string
                                            if i[idx][0].string
                                            else " ".join(
                                                [s for s in i[idx][0].strings]
                                            )
                                            for idx in range(len(i))
                                        ]
                                    ],
                                    [],
                                )
                            )
                        )
                        formatted_matches.append(
                            {
                                "elements": [f[0] for f in i],
                                "start_pos": pos[0],
                                "end_pos": pos[1],
                                "selector": k,
                                "full_norm_text": full_text,
                            }
                        )
                    else:
                        pos = i[1]
                        text = (
                            i[0].string
                            if i[0].string
                            else " ".join([s for s in i[0].strings])
                        )
                        text = self._normalize_toc_title(text)
                        # make this dict with start_ele, end_ele, first_ele_text_norm, full_text_norm, pos_start, pos_end
                        formatted_matches.append(
                            {
                                "elements": [i[0]],
                                "start_pos": pos[0],
                                "end_pos": pos[1],
                                "selector": k,
                                "full_norm_text": text,
                            }
                        )
        return formatted_matches

    def _get_table_elements_containing(
        self, start_element: element.Tag, required_items: list[re.Pattern] = [], shape_constraint: tuple[int, int] = (-1, -1), field_length_constraint: tuple[int] = (-1, -1)
    ):
        """
        Get all the <table> items parsed after start_element
        which have all required_items in them.
        
        Args:
            start_element: what element to start the search from,
                           checks elements parsed after start_element for tables.
            required_items: check the table fields against this filter.
            shape_constraint: table of shape m,n must be smaller than shape_constraint.
                              shape_constraint[0] = number of rows
                              shape_constraint[1] = number of columns
                             -1 means no constraint in that dimension.
            field_length_constraint: field length in that column must be smaller than
                                     field_length_constraint.
                                     -1 means no length constraint for that column.
                                     Element count cant exceed shape_constraint[1].
        Returns:
            list[element.Tag] or []
        """
        if (len(field_length_constraint) > shape_constraint[1]) and (shape_constraint[1] != -1):
            raise ValueError(f"field_length_constraint elements can't exceed shape_constraint[1]")
        tables = start_element.find_all_next("table")
        found_tables = []
        for table in tables:
            parsed_table = self.primitive_htmltable_parse(table)
            if shape_constraint[0] != -1:
                if len(parsed_table) > shape_constraint[0]:
                    continue
            if shape_constraint[1] != -1:
                if len(parsed_table[0]) > shape_constraint[1]:
                    continue
            ritems = required_items.copy()
            for entry in sum(parsed_table, []):
                try:
                    remove_idx = None
                    for idx, ritem in enumerate(ritems):
                        if re.search(ritem, str(entry)):
                            remove_idx = idx
                            break
                    if remove_idx is not None:
                        ritems.pop(remove_idx)
                except TypeError as e:
                    logger.debug(e, exc_info=True)
            if ritems == []:
                is_correct_field_length = True
                for row in  parsed_table:
                    for idx, col in enumerate(row):
                        try:
                            if field_length_constraint[idx] == -1:
                                continue
                            else:
                                if len(col) > field_length_constraint[idx]:
                                    is_correct_field_length = False
                        except IndexError:
                            logger.debug("tried to access shape_constraint item which doesnt exist. excepted IndexError.")
                if is_correct_field_length is True:
                    found_tables.append(table)
            else:
                if len(ritems) < len(required_items):
                    logger.debug(
                        f"at least one required item present, following were missing to qualify: {ritems}"
                    )
        return found_tables

    def _get_cover_page_start_ele_from_toc(self, toc_element: element.Tag):
        cover_page_start_ele = None
        cover_page_end_ele = None
        start_re_term = re.compile(r"(^\s*PROSPECTUS)|(^\s*PRELIMINARY\s*PROSPECTUS)", re.MULTILINE)
        alternative_start_re_term = re.compile(r"subject(\s)*to(\s)*completion,", re.I)
        end_re_term = re.compile(
            r"(the(?:\s){,3}date(?:\s){,3}of(?:\s){,3}this(?:\s){,3}prospectus(?:\s){,3}(?:supplement)?(?:\s){,3}is)|(this(?:\s){,3}prospectus(?:\s){,3}is(?:\s){,3}dated)|(Prospectus(?:\s){0,3}dated)",
            re.I,
        )
        ele = toc_element
        while True:
            ele = ele.previous_element
            if ele is None:
                return None
            string = None
            if isinstance(ele, NavigableString):
                string = str(ele)
            else:
                string = ele.string if ele.string else None
            if string:
                if re.search(start_re_term, string) or re.search(
                    alternative_start_re_term, string
                ):
                    cover_page_start_ele = (
                        ele if not isinstance(ele, NavigableString) else ele.parent
                    )
                    logger.debug(f"found cover_page_start_ele: {cover_page_start_ele}")
                if cover_page_end_ele is None:
                    if re.search(end_re_term, string):
                        cover_page_end_ele = (
                            ele if not isinstance(ele, NavigableString) else ele.parent
                        )
                        logger.debug(f"found cover_page_end_ele: {cover_page_end_ele}")
            if (cover_page_end_ele is not None) and (cover_page_start_ele is not None):
                if (cover_page_end_ele.sourceline > cover_page_start_ele.sourceline
                    ) or (
                        (
                            cover_page_end_ele.sourceline == cover_page_start_ele.sourceline
                        ) and (
                            cover_page_end_ele.sourcepos > cover_page_start_ele.sourcepos
                        )
                ):
                    logger.debug(
                        f"found cover page: {cover_page_start_ele}, {cover_page_end_ele}"
                    )
                    # get soup between start, end ele and return it
                    # i might need to add this aswell as the tocs to the section_start_elements and then just pass it to the existing splitter?
                    return cover_page_start_ele

    def _get_front_page_sections_from_first_cover_page(
        self,
        start_ele: element.Tag,
        re_terms: list[re.Pattern] = [re.compile(r"EXPLANATORY(\s)*NOTE")],
    ):
        """

        Args:
            re_terms: list of re.Patterns that should be split of from the front page into their own section_start_elements
            start_ele: the start element of the first cover page
        """
        section_start_elements = []
        ele = start_ele
        while True:
            prev_ele = ele
            ele = ele.previous_element
            if (ele.name == "title") or (ele.name == "text"):
                if isinstance(prev_ele, NavigableString):
                    while isinstance(prev_ele, NavigableString):
                        prev_ele = prev_ele.next_element
                section_start_elements.append(
                    {"section_title": "front page", "ele": prev_ele}
                )
                break
            if not isinstance(ele, NavigableString):
                string = ele.string if ele.string else None
                if string:
                    for term in re_terms:
                        if re.search(term, string):
                            section_start_elements.append(
                                {
                                    "section_title": self._normalize_toc_title(string),
                                    "ele": ele,
                                }
                            )
                            re_terms.remove(term)
        # logger.debug(
        #     f"section_start_elements from  _get_front_page...: {section_start_elements}"
        # )
        return section_start_elements

    # def _get_toc_list(self, doc: BeautifulSoup, start_table: element.Tag = None):
    #     """gets the elements of the TOC as a list.

    #     currently doesnt account for multi table TOC.
    #     Returns:
    #         list of [description, page_number]
    #     """
    #     if start_table is None:
    #         try:
    #             close_to_toc = doc.find(
    #                 string=re.compile("(table.?.?of.?.?contents)", re.I | re.DOTALL),
    #                 recursive=True,
    #             )
    #             if isinstance(close_to_toc, NavigableString):
    #                 close_to_toc = close_to_toc.parent
    #             if "href" in close_to_toc.attrs:
    #                 name_or_id = close_to_toc["href"][-1]
    #                 close_to_toc = doc.find(name=name_or_id)
    #                 if close_to_toc is None:
    #                     close_to_toc = doc.find(id=name_or_id)
    #             toc_table = close_to_toc.find_next("table")
    #         except AttributeError as e:
    #             logger.info(e, exc_info=True)
    #             logger.info(
    #                 "This filing mostlikely doesnt have a TOC. Couldnt find TOC from close_to_toc tag"
    #             )
    #             return None
    #     else:
    #         toc_table = start_table
    #     if toc_table is None:
    #         logger.info("couldnt find TOC from close_to_toc tag")
    #     dirty_toc = self.parse_htmltable_with_header(toc_table, colspan_mode="separate")
    #     if dirty_toc is None:
    #         logger.info("couldnt get toc from toc_table")
    #         return None
    #     if len(dirty_toc) < 10:
    #         # assume we missed the toc
    #         try:
    #             toc_table = toc_table.find_next("table")
    #             toc = self._get_toc_list(doc, toc_table)
    #             return toc
    #         except AttributeError:
    #             return None
    #     toc = []
    #     for row in dirty_toc:
    #         offset = 0
    #         for idx in range(len(row)):
    #             field = row[idx - offset]
    #             if (field is None) or (field == ""):
    #                 row.pop(idx - offset)
    #                 offset += 1

    #         if row != []:
    #             if len(row) == 1:
    #                 row.append("None")
    #             toc.append(row)
    #     return toc
    # toc_table2 = toc_table.find_next("table")
    # toc2 = self.parse_htmltable_with_header(toc_table2, colspan_mode="separate")
    # if toc2 is not None:
    #     try:
    #         if (len(toc2[0][-1]) == (2 or 3)) and (len(toc2[0]) == len(dirty_toc[0])):
    #             [dirty_toc.append(t) for t in toc2]
    #             logger.info("doc had a multipage toc!")
    #     except TypeError:
    #         pass
    # toc = []

    # for row in dirty_toc:
    #     print(row)

    def _look_for_toc_matches_after(
        self,
        start_ele: element.Tag,
        re_toc_titles: list[(re.Pattern, int)],
        min_distance: int = 5,
        stop_ele: element.Tag = None,
    ):
        """
        looks for regex matches in the string of the tags of the element tree.

        Avoids matches that are descendants of the last match and matches that are too close
        together.

        Args:
            start_ele: from what element to start the search
            re_toc_titles: should be a list of tuples consisting of (the regex pattern, the max length to match for)
            max length to match for should be equal to the  length + some whitespace margin of the toc_title string
            min_distance: minimum number of elements between matches
            stop_ele: at what element to stop the search at the latest.
        """
        title_matches = []
        min_distance_count = 0
        matched_ele = None
        for ele in start_ele.next_elements:
            if stop_ele is not None:
                if ele == stop_ele:
                    break
            if matched_ele == ele.previous_element:
                if min_distance_count >= min_distance:
                    matched_ele = None
                    min_distance_count = 0
                else:
                    matched_ele = ele
                    min_distance_count += 1
                    continue
            if ele.string:
                content = ele.string
                for re_term, term_length in re_toc_titles:
                    if re.search(re_term, content):
                        content_length = (
                            len(ele)
                            if isinstance(ele, (NavigableString, str))
                            else len(ele.string)
                        )
                        if isinstance(ele, str):
                            ele = ele.parent
                        if content_length < term_length:
                            title_matches.append(ele)
                            # print(re_term, ele)
                            matched_ele = ele
                            break
        return title_matches

    def _search_toc_match_in_list_of_tags(self, tags, re_toc_titles):
        """
        looks for regex matches in the string of the tags and their descendants.

        Avoids matches that are descendants of the last match.
        """
        matches = []
        for tag in tags:
            matched = False
            if tag.string:
                content = tag.string
                if content == "":
                    pass
                else:
                    for re_term, _ in re_toc_titles:
                        if re.search(re_term, content):
                            matches.append(tag)
                            matched = True
                            break
            if matched is False:
                try:
                    for child in tag.descendants:
                        if child.string:
                            content = child.string
                            if content is not (None or ""):
                                for re_term, _ in re_toc_titles:
                                    if re.search(re_term, content):
                                        matches.append(child)
                                        break
                except AttributeError as e:
                    raise e
        return matches

    def _ele_is_between(self, doc: str, ele: element.Tag, x1, x2):
        """check if ele is between x1 and x2 in the document (by regex span)"""
        ele_span = self.get_span_of_element(doc, ele)
        if (x1 > ele_span[0]) and (x2 < ele_span[1]):
            return True
        else:
            return False

    def _normalize_toc_title(self, title):
        return re.sub("\s{1,}", " ", re.sub("\n{1,}", " ", title)).lower().strip()

    def _create_toc_re(self, search_term, max_length=None):
        if max_length is None:
            max_length = len(search_term) + 4
        else:
            pass
        return (
            re.compile(
                r"^\s*" + re.sub(r"(\s){1,}", r"(?:.){0,4}", search_term) + r"\s*$",
                re.I | re.DOTALL | re.MULTILINE,
            ),
            max_length,
        )

    def _split_into_sections_by_tags(
        self, doc: BeautifulSoup, section_start_elements: list[dict]
    ):
        """
        splits html doc into malformed html strings by section_start_elements.
        Args:
            section_start_elements: [{"section_title": section_title, "ele": element.Tag}]
        """
        sections = []
        # make sure that the section_start_elements are sorted in ascending order by sourceline
        sorted_section_start_elements = sorted(
            section_start_elements, key=lambda x: (x["ele"].sourceline, x["ele"].sourcepos)
        )
        # logger.debug(sorted_section_start_elements)
        for idx, section_start_element in enumerate(sorted_section_start_elements):
            sorted_section_start_elements[idx] = _add_unique_id_to_dict(
                section_start_element
            )
        logger.debug([s["section_title"] for s in list(sorted_section_start_elements)])
        for section_nr, start_element in enumerate(sorted_section_start_elements):

            ele = start_element["ele"]
            logger.debug(
                f'{ele.sourceline, ele.sourcepos} inserted start ele {"-START_SECTION_TITLE_" + start_element["section_title"] + start_element["UUID"]}'
            )
            ele.insert_before(
                "-START_SECTION_TITLE_"
                + start_element["section_title"]
                + start_element["UUID"]
            )
            if section_nr == len(sorted_section_start_elements) - 1:
                while True:
                    prev_ele = ele
                    ele = ele.next_element
                    if ele is None:
                        prev_ele.insert_before(
                            "-STOP_SECTION_TITLE_"
                            + start_element["section_title"]
                            + start_element["UUID"]
                        )
                        break
            else:
                while ele != sorted_section_start_elements[section_nr + 1]["ele"]:
                    next_ele = ele.next_element
                    if not next_ele:
                        logger.debug(
                            f"element that should be reached before advancing with inserting stop marker: {section_start_elements[section_nr + 1]['ele']}"
                        )
                        # logger.debug(f"sourceline stop ele: {section_start_elements[section_nr + 1]['ele'].sourceline}")
                        logger.debug(
                            f"_split_into_sections_by_tag: next_element was none. start_element['ele']: {start_element['ele']} \t section_nr: {section_nr}"
                        )
                        logger.debug(
                            f"sourceline start ele: {start_element['ele'].sourceline}"
                        )
                        # --> sort by sourceline at start to avoid fuckupy
                        break
                    ele = next_ele
                logger.debug(
                    f'{ele.sourceline if not isinstance(ele, NavigableString) else ele.parent.sourceline} inserted stop ele {"-STOP_SECTION_TITLE_" + start_element["section_title"] + start_element["UUID"]}'
                )
                ele.insert_before(
                    "-STOP_SECTION_TITLE_"
                    + start_element["section_title"]
                    + start_element["UUID"]
                )
        text = str(doc)
        for idx, sec in enumerate(sorted_section_start_elements):
            logger.debug(
                f'looking for: {"-START_SECTION_TITLE_" + re.escape(sec["section_title"] + sec["UUID"])}'
            )
            start = re.search(
                re.compile(
                    "-START_SECTION_TITLE_"
                    + re.escape(sec["section_title"] + sec["UUID"]),
                    re.MULTILINE,
                ),
                text,
            )

            end = re.search(
                re.compile(
                    "-STOP_SECTION_TITLE_"
                    + re.escape(sec["section_title"] + sec["UUID"]),
                    re.MULTILINE,
                ),
                text,
            )
            try:

                sections.append(
                    HTMFilingSection(
                        title=self._normalize_toc_title(sec["section_title"]),
                        content=text[start.span()[1] : end.span()[0]],
                        extension=self.extension,
                        form_type=self.form_type,
                    )
                )
            except Exception as e:
                print("FAILURE TO SPLIT SECTION BECAUSE:")
                print(
                    f"start: {start}, end: {end}, section_title: {sec['section_title']}, section_idx/total: {idx}/{len(section_start_elements)-1}"
                )
                logger.debug(e, exc_info=True)
                print("----------------")
        return sections if sections != [] else None

    def _split_by_table_of_content_based_on_headers(self, doc: BeautifulSoup):
        """split the filing by the element strings of the TOC"""
        close_to_toc = doc.find(
            string=re.compile("(table.?.?of.?.?contents)", re.I | re.DOTALL),
            recursive=True,
        )
        toc_table = close_to_toc.find_next("table")
        # still need to account for multi table TOC, while close_to_toc -> check if toc table
        if toc_table is None:
            logger.info("couldnt find TOC table from close_to_toc tag")
        toc = self.parse_htmltable_with_header(toc_table, colspan_mode="separate")
        # check if we have desc, page header or not and remove it if so
        possible_header = toc[0]
        for head in possible_header:
            if re.search(re.compile("(descript.*)|(page)", re.I), head):
                toc.pop(0)
                break
        toc_titles = []
        for entry in toc:
            if entry[0] != "":
                toc_titles.append(entry[0])

        re_toc_titles = [
            (
                re.compile(
                    r"^\s*" + re.sub(r"(\s|\n)", r"(?:.){0,4}", t) + r"\s*$",
                    re.I | re.DOTALL | re.MULTILINE,
                ),
                len(t) + 3,
            )
            for t in toc_titles
        ]
        ele_after_toc = toc_table.next_sibling
        title_matches = self._look_for_toc_matches_after(
            ele_after_toc, re_toc_titles, min_distance=5
        )
        stil_missing_toc_titles = [
            s
            for s in toc_titles
            if self._normalize_toc_title(s)
            not in [
                self._normalize_toc_title(f.string)
                if f.string
                else self._normalize_toc_title(" ".join([t for t in f.strings]))
                for f in title_matches
            ]
        ]
        # assume that the still missing toc titles are multi tag titles or malformed after the 4th word
        # so lets take take the first 4 words of the title and look for those
        norm_four_word_titles = [
            " ".join(self._normalize_toc_title(title).split(" ")[:4])
            for title in stil_missing_toc_titles
        ]
        re_four_word_titles = [
            self._create_toc_re(t, max_length=len(t) + 10)
            for t in norm_four_word_titles
        ]
        four_word_matches = self._look_for_toc_matches_after(
            ele_after_toc, re_four_word_titles
        )
        [title_matches.append(m) for m in four_word_matches]
        norm_title_matches = [
            self._normalize_toc_title(
                t.string if t.string else " ".join([s for s in t.strings])
            )
            for t in title_matches
        ]
        unique_toc_titles = set([self._normalize_toc_title(t) for t in toc_titles])
        unique_match_titles = set(norm_title_matches)

        alternative_matches = []
        for failure in unique_toc_titles - unique_match_titles:
            if failure in TOC_ALTERNATIVES.keys():
                alternative_options = TOC_ALTERNATIVES[failure]
                for option in alternative_options:
                    matches = self._look_for_toc_matches_after(
                        ele_after_toc, [self._create_toc_re(option)]
                    )
                    if matches:
                        alternative_matches.append(matches)
                        break
        for match in alternative_matches:
            if len(match) > 1:
                logger.warning(alternative_matches)
                logger.warning("--------->>>!! address this now !!<<<-------------")
            else:
                title_matches.append(match[0])
        section_start_elements = []
        headers_based_on_style = self._format_matches_based_on_style(
            self._get_possible_headers_based_on_style(doc, doc), header_style="main"
        )
        # print()
        # print("ALL MATCHES BASED ON HEADERS")
        # print(title_matches, headers_based_on_style)
        replace_headers = []
        add_headers = []
        for idx, headers in enumerate(headers_based_on_style):
            for tm_idx, tm in enumerate(title_matches):
                if isinstance(tm, list):
                    ele = tm[0]
                else:
                    ele = tm
                if ele in headers["elements"]:
                    replace_headers.append((tm_idx, headers["elements"]))
                else:
                    if headers["elements"] not in add_headers:
                        add_headers.append(headers["elements"])
        for idx, replace_header in replace_headers:
            title_matches[idx] = replace_header
        for add_header in add_headers:
            # print(add_header)
            title_matches.append(add_header)
        for tm_idx, tm in enumerate(title_matches):
            if not isinstance(tm, list):
                title_matches[tm_idx] = [tm]

        title_matches.sort(key=lambda x: x[0].sourceline)

        offset = 0
        copy_title_matches = title_matches.copy()
        for idx, tm in enumerate(copy_title_matches):
            if idx < (len(title_matches) - 1):
                next_item_sourcelines = [
                    t.sourceline for t in copy_title_matches[idx + 1]
                ]
                current_sourcelines = [t.sourceline for t in tm]
                # print(tm, "\t" ,copy_title_matches[idx + 1])
                for csourceline in current_sourcelines:
                    if csourceline in next_item_sourcelines:
                        if len(tm) == len(copy_title_matches[idx + 1]):
                            title_matches.pop(idx + 1 - offset)
                            offset += 1
                        else:
                            if len(tm) > len(copy_title_matches[idx + 1]):
                                title_matches.pop(idx + 1 - offset)
                            else:
                                title_matches.pop(idx - offset)
                            offset += 1
                            break

        for match in title_matches:
            _title = " ".join(
                sum(
                    [
                        [
                            match[idx].string
                            if match[idx].string
                            else " ".join([s for s in match[idx].strings])
                            for idx in range(len(match))
                        ]
                    ],
                    [],
                )
            )
            section_title = self._normalize_toc_title(_title)
            if section_title not in HEADERS_TO_DISCARD:
                # print({"section_title": section_title, "ele": match[0]})
                section_start_elements.append(
                    {"section_title": section_title, "ele": match[0]}
                )
            else:
                logger.info(
                    f"_split_by_table_of_content_based_on_headers: \t Discarded section: {section_title} because it was in HEADERS_TO_DISCARD"
                )

        return self._split_into_sections_by_tags(
            doc, section_start_elements=section_start_elements
        )

    def _split_by_table_of_contents_based_on_hrefs(self, doc: BeautifulSoup):
        """split the filing based on the hrefs, linking to different parts of the filing, from the TOC."""
        #
        #!
        #!
        #  ADD HEADERS BASED ON STYLE TO THIS FUNCTION TO MAKE IT MORE EQUIVALENT TO SPLITTING BY HEADERS
        #!
        #!
        #

        # get close to TOC
        try:
            close_to_toc = doc.find(
                string=re.compile("(table..?.?of..?.?contents)", re.I | re.DOTALL)
            )
            logger.debug(f"inital close_to_toc in split with hrefs: {close_to_toc}")
            if isinstance(close_to_toc, NavigableString):
                close_to_toc = close_to_toc.parent
                # logger.debug("close_to_toc was navigable string")
            if "href" in close_to_toc.attrs:
                # logger.debug(f"found href in close_to_toc")
                name_or_id = close_to_toc["href"][1:]
                # logger.debug(f"close_to_toc href attr: {close_to_toc['href']}")
                # logger.debug(f"name_or_id: {name_or_id}")
                close_to_toc = doc.find(True, {"name": name_or_id})
                if close_to_toc is None:
                    close_to_toc = doc.find(True, {"id": name_or_id})
                # logger.debug(
                #     f"close_to_toc after looking for the name and id: {close_to_toc}"
                # )
            toc_table = close_to_toc.find_next("table")
            hrefs = toc_table.findChildren("a", href=True)
            logger.debug(f"hrefs from toc: {hrefs}")
            first_ids = [h["href"] for h in hrefs]
            n = 0
            if first_ids == []:
                while (first_ids == []) or (n < 5):
                    close_to_toc = close_to_toc.find_next(
                        string=re.compile(
                            "(toc|table..?.?of..?.?contents)", re.I | re.DOTALL
                        )
                    )
                    toc_table = close_to_toc.find_next("table")
                    hrefs = toc_table.findChildren("a", href=True)
                    first_ids = [h["href"] for h in hrefs]
                    n += 1
        except AttributeError as e:
            logger.info(e, exc_info=True)
            return None

        # determine first section so we can check if toc is multiple pages
        id = first_ids[0]
        id_match = doc.find(id=id[1:])
        if id_match is None:
            name_match = doc.find(attrs={"name": id[1:]})
            first_toc_element = name_match
        else:
            first_toc_element = id_match
        # check that we didnt miss other pages of toc
        if not first_toc_element:
            raise ValueError("couldnt find section of first element in TOC")

        # get all tables before the first header found in toc
        tables = first_toc_element.find_all_previous("table")
        track_ids_done = []
        section_start_elements = []
        section_start_ids = []
        for idx in range(len(tables) - 1, -1, -1):
            table = tables[idx]
            hrefs = table.findChildren("a", href=True)
            ids = []
            for a in hrefs:
                id = a["href"]
                toc_title = " ".join([s for s in a.strings]).lower()
                ids.append((id, toc_title))
            section_start_ids.append(ids)
        for entry in sum(section_start_ids, []):
            id = entry[0]
            if id not in track_ids_done:
                id_match = doc.find(id=id[1:])
                if id_match is None:
                    id_match = doc.find(attrs={"name": id[1:]})
                track_ids_done.append(id)
                if id_match:
                    # print(id_match, entry)
                    section_start_elements.append(
                        {"ele": id_match, "section_title": entry[1]}
                    )
                else:
                    print("NO ID MATCH FOR", entry)

        return self._split_into_sections_by_tags(
            doc, section_start_elements=section_start_elements
        )


class ParserS3(HTMFilingParser):
    form_type = "S-3"
    extension = ".htm"
    """process and parse s-3 filing."""

    def split_into_sections(self, doc: BeautifulSoup | str):
        if isinstance(doc, str):
            doc_ = BeautifulSoup(doc, features="html5lib")
        else:
            doc_ = doc
        sections = self.split_by_table_of_contents(doc_)
        logger.debug(
            f"found sections: {len(sections) if sections is not None else None, [sec.title if sec.title else None for sec in sections]}"
        )
        if sections is not None:
            return sections
        else:
            raise AttributeError(
                f"unhandled case of split_into_sections of the ParserS3"
            )

    def split_by_table_of_contents(self, doc: BeautifulSoup):
        """split a filing with a TOC into sections based on the TOC.

        Args:
            doc: html parsed with bs4

        Returns:
            list[HTMFilingSection]
        """

        section_start_elements = []
        # get TOCs
        tocs = self._get_table_elements_containing(doc, REQUIRED_TOC_ITEMS_S3)
        logger.debug(
            f"found following TOCs: {[self.primitive_htmltable_parse(toc) for toc in tocs]}"
        )
        cover_page_list = []
        drop = []
        for idx, toc in enumerate(tocs):
            _toc = self._parse_toc_table_element(toc)
            logger.debug(f"toc after parse: {_toc}")
            if _toc is None:
                drop.insert(0, idx)
        if drop != []:
            for idx in drop:
                tocs.pop(idx)
        for idx, toc in enumerate(tocs):
            logger.debug(f"working on toc number: {idx}")
            section_start_elements.append(
                {"section_title": "toc " + str(idx), "ele": toc}
            )
            cover_page_start_ele = self._get_cover_page_start_ele_from_toc(toc)
            if cover_page_start_ele:
                section_start_elements.append(
                    {
                        "section_title": "cover page " + str(idx),
                        "ele": cover_page_start_ele,
                    }
                )
                if idx == 0:
                    front_page_sections = (
                        self._get_front_page_sections_from_first_cover_page(
                            cover_page_start_ele
                        )
                    )
                    if front_page_sections != []:
                        [section_start_elements.append(x) for x in front_page_sections]
            cover_page_list.append(
                cover_page_start_ele if cover_page_start_ele else None
            )
        logger.debug(f"section_start_elements before doing sections: {[sec['section_title'] for sec in section_start_elements]}")
        for idx, toc in enumerate(tocs):
            href_start_elements = self._get_section_start_elements_from_toc_hrefs(
                doc, toc
            )
            logger.debug(f"href_start_elements: {href_start_elements}")
            if (href_start_elements is not None) and (href_start_elements != []):
                for x in href_start_elements:
                    if x not in section_start_elements:
                        section_start_elements.append(x)
            else:
                try:
                    stop_ele = cover_page_list[idx]
                except IndexError:
                    stop_ele = None
                header_start_elements = (
                    self._get_section_start_elements_from_toc_headers(
                        doc, toc, stop_ele=stop_ele
                    )
                )
                logger.debug(f"header_start_elements: {header_start_elements}")
                if header_start_elements != []:
                    for x in header_start_elements:
                        if x not in section_start_elements:
                            section_start_elements.append(x)
        return self._split_into_sections_by_tags(
            doc, section_start_elements=section_start_elements
        )

    def _get_section_start_elements_from_toc_headers(
        self, doc: BeautifulSoup, toc_table: element.Tag, stop_ele: element.Tag = None
    ):
        """split the filing by the element strings of the TOC"""
        toc = self._parse_toc_table_element(toc_table)
        toc_titles = []
        for entry in toc:
            if entry["title"] != "":
                toc_titles.append(entry["title"])
        re_toc_titles = [
            (
                re.compile(
                    r"^\s*" + re.sub(r"(\s|\n)", r"(?:.){0,4}", t) + r"\s*$",
                    re.I | re.DOTALL | re.MULTILINE,
                ),
                len(t) + 3,
            )
            for t in toc_titles
        ]
        ele_after_toc = toc_table.next_sibling
        title_matches = self._look_for_toc_matches_after(
            ele_after_toc, re_toc_titles, min_distance=5, stop_ele=stop_ele
        )
        stil_missing_toc_titles = [
            s
            for s in toc_titles
            if self._normalize_toc_title(s)
            not in [
                self._normalize_toc_title(f.string)
                if f.string
                else self._normalize_toc_title(" ".join([t for t in f.strings]))
                for f in title_matches
            ]
        ]
        # assume that the still missing toc titles are multi tag titles or malformed after the 4th word
        # so lets take take the first 4 words of the title and look for those
        norm_four_word_titles = [
            " ".join(self._normalize_toc_title(title).split(" ")[:4])
            for title in stil_missing_toc_titles
        ]
        re_four_word_titles = [
            self._create_toc_re(t, max_length=len(t) + 10)
            for t in norm_four_word_titles
        ]
        four_word_matches = self._look_for_toc_matches_after(
            ele_after_toc, re_four_word_titles, stop_ele=stop_ele
        )
        [title_matches.append(m) for m in four_word_matches]
        norm_title_matches = [
            self._normalize_toc_title(
                t.string if t.string else " ".join([s for s in t.strings])
            )
            for t in title_matches
        ]
        unique_toc_titles = set([self._normalize_toc_title(t) for t in toc_titles])
        unique_match_titles = set(norm_title_matches)

        alternative_matches = []
        for failure in unique_toc_titles - unique_match_titles:
            if failure in TOC_ALTERNATIVES.keys():
                alternative_options = TOC_ALTERNATIVES[failure]
                for option in alternative_options:
                    matches = self._look_for_toc_matches_after(
                        ele_after_toc, [self._create_toc_re(option)], stop_ele=stop_ele
                    )
                    if matches:
                        alternative_matches.append(matches)
                        break
        for match in alternative_matches:
            if len(match) > 1:
                logger.warning(alternative_matches)
                logger.warning("--------->>>!! address this now !!<<<-------------")
            else:
                title_matches.append(match[0])
        section_start_elements = []
        for match in title_matches:
            _title = self.get_joined_text_of_tag(match)
            section_title = self._normalize_toc_title(_title)
            if section_title not in HEADERS_TO_DISCARD:
                section_start_elements.append(
                    {"section_title": section_title, "ele": match}
                )
            else:
                logger.info(
                    f"_split_by_table_of_content_based_on_headers: \t Discarded section: {section_title} because it was in HEADERS_TO_DISCARD"
                )
        return section_start_elements

    def get_joined_text_of_tag(self, match):
        _title = " ".join(
            sum(
                [
                    [
                        m.string
                        if m.string
                        else " ".join([s for s in m.strings])
                        for m in match
                    ]
                ],
                [],
            )
        )
        return _title

    def _get_section_start_elements_from_toc_hrefs(
        self, doc: BeautifulSoup, toc_table: element.Tag
    ):
        """split the filing based on the hrefs, linking to different parts of the filing, from the TOC."""
        table = self._parse_toc_table_element(toc_table)
        id = table[0]["href"]
        if id is None:
            return None
        id_match = doc.find(attrs={"id": id[1:]})
        if id_match is None:
            name_match = doc.find(attrs={"name": id[1:]})
            first_toc_element = name_match
        else:
            first_toc_element = id_match
        if not first_toc_element:
            raise ValueError("couldnt find section of first element in TOC")
        track_ids_done = []
        section_start_elements = []
        for entry in table:
            logger.debug(f"looking for section start element of entry: {entry}")
            id = entry["href"]
            # logger.debug(f"working on id: {id}")
            if id not in track_ids_done:
                id_match = doc.find(attrs={"id": id[1:]})
                if id_match is None:
                    id_match = doc.find(attrs={"name": id[1:]})
                track_ids_done.append(id)
                if id_match:
                    # print(id_match.sourceline, id_match.sourcepos, entry)
                    # logger.debug(f"id_match found: {type(id_match), id_match}")
                    section_start_elements.append(
                        {"ele": id_match, "section_title": entry["title"]}
                    )
                else:
                    logger.debug(("NO ID MATCH FOR", entry))
        return section_start_elements

    def extract_tables(
        self, soup: BeautifulSoup, reintegrate=["ul_bullet_points", "one_row_table"]
    ):
        """see HTMFilingParser"""
        unparsed_tables = self.get_unparsed_tables(soup)
        tables = {"reintegrated": [], "extracted": []}
        for t in unparsed_tables:
            parsed_table = None
            if t is None:
                continue
            if self.table_has_header(t):
                parsed_table = self.parse_htmltable_with_header(t)
            else:
                parsed_table = self.primitive_htmltable_parse(t)
            try:
                cleaned_table = self._clean_parsed_table_drop_empty_rows(
                    self._clean_parsed_table_columnwise(
                        self._preprocess_table(parsed_table)
                    ),
                    remove_=["", None],
                )
            except IndexError:
                continue
            if cleaned_table == []:
                continue
            classification = self.classify_table(cleaned_table)
            if classification == "unclassified":
                classification = super().classify_table(cleaned_table)
            if classification in reintegrate:
                reintegrate_html = self._make_reintegrate_html_of_table(
                    classification, cleaned_table
                )
                t.replace_with(reintegrate_html)
                tables["reintegrated"].append(
                    {
                        "classification": classification,
                        "reintegrated_as": reintegrate_html,
                        "table_meta": {"table_elements": [t]},
                        "parsed_table": cleaned_table,
                    }
                )
            else:
                # further reformat the table
                tables["extracted"].append(
                    {
                        "classification": classification,
                        "table_meta": {"table_elements": [t]},
                        "parsed_table": cleaned_table,
                    }
                )
        return tables

    def classify_table(self, table: list[list]):
        try:
            table_shape = (len(table), len(table[0]))
        except IndexError:
            return "empty"
        if table_shape[0] > 1:
            for row in table:
                if _row_is_ignore(row=row) is False:
                    if table_header_has_fields(row, REGISTRATION_TABLE_HEADERS_S3):
                        return "registration_table"
                    else:
                        break
        return "unclassified"


class Parser8K(HTMFilingParser):
    form_type = "8-K"
    extension = ".htm"
    """
    process and parse 8-k filing.

    Usage:
        1) open file and read() file with utf-8 and in "r"
        or
        1) call parser.get_doc(path)
        2) call split_into_sections on the result from 1)
    """

    def __init__(self):
        self.soup = None
        self.match_groups = self._create_match_group()

    def split_into_sections(self, doc: str|BeautifulSoup) -> list[FilingSection]:
        """
        split the filing into FilingSections.

        Returns:
            list[HTMFilingSection] or []
        """
        clean_doc = self.get_clean_text_only_from_raw_htm(doc)
        items = self._parse_items(clean_doc)
        logger.debug(f"8-K items found: {len(items)}")
        if items == []:
            return []
        else:
            sections = []
            for item in items:
                for k, v in item.items():
                    sections.append(
                        HTMFilingSection(
                            title=k,
                            content=v,
                            extension=self.extension,
                            form_type=self.form_type,
                        )
                    )
            return sections

    def split_into_items(self, path: str, get_cik=True):
        """
        split the 8k into the individual items.

        Args:
            path: path to the 8-k filing
            get_cik: if we want to get the CIK from the folder structure

        Returns:
                {
                "cik": cik or None,
                "file_date": the date of report in filing,
                "items": list[
                              {section_title: str : section_content: str}
                             ]
                }
        """
        _path = path
        with open(path, "r", encoding="utf-8") as f:
            if get_cik is True:
                if isinstance(path, str):
                    _path = Path(path)
                cik = _path.parent.name.split("-")[0]
            else:
                cik = None
            file = f.read()
            filing = self.get_clean_text_only_from_raw_htm(file)
            try:
                items = self._parse_items(filing)
            except AttributeError as e:
                logger.info((e, path, filing), exc_info=True)
                return None
            except IndexError as e:
                logger.info((e, path, filing), exc_info=True)
                return None
            try:
                date = self.get_date_of_report_matches(filing)
            except AttributeError as e:
                logger.info((e, path, filing), exc_info=True)
                return None
            except ValueError as e:
                logger.info((e, filing), exc_info=True)
                return None
            date_group = date.groups()
            valid_dates = []
            for d in date_group:
                if d is not None:
                    valid_dates.append(d)
            if len(valid_dates) > 1:
                logger.error(f"valid_dates found: {valid_dates}, filing: {filing}")
                raise AttributeError(
                    "more than one valid date of report for this 8k found"
                )
            else:
                try:
                    valid_date = self._parse_date_of_report(valid_dates[0])
                except Exception as e:
                    logging.info(
                        f"couldnt parse date of filing: {valid_dates}, filing: {filing}"
                    )
                    return None
            return {"cik": str(cik), "file_date": valid_date, "items": items}

    def get_item_matches(self, filing: str):
        """get matches for the 8-k items.

        Args:
            filing: should be a cleaned 8-k filing (only text content, no html ect)"""
        matches = []
        for match in re.finditer(self.match_groups, filing):
            matches.append([match.start(), match.end(), match.group(0)])
        return matches

    def get_signature_matches(self, filing: str):
        """get matches for the signatures"""
        signature_matches = []
        for smatch in re.finditer(re.compile("(signatures|signature)", re.I), filing):
            signature_matches.append(
                [smatch.start(), smatch.end(), smatch.start() - smatch.end()]
            )
        return signature_matches

    def get_date_of_report_matches(self, filing: str):
        date = re.search(COMPILED_DATE_OF_REPORT_PATTERN, filing)
        if date is None:
            raise ValueError
        return date

    def _parse_items(self, filing: str):
        """extract the items from the filing and their associated paragraph"""
        # first get items and signatures
        extracted_items = []
        items = self.get_item_matches(filing)
        signatures = self.get_signature_matches(filing)
        # ensure that there is one signature which comes after all the items
        # otherwise discard the signature
        last_idx = len(items) - 1
        for idx, sig in enumerate(signatures):
            # logger.debug(sig)
            for item in items:
                if sig[1] < item[1]:
                    try:
                        signatures.pop(idx)
                    except IndexError as e:
                        raise e
                    except TypeError as e:
                        logger.debug(
                            (f"unhandled case of TypeError: ", sig, signatures, e),
                            exc_info=True,
                        )
            if len(signatures) == 0:
                continue
        for idx, item in enumerate(items):
            # last item
            if idx == last_idx:
                try:
                    body = filing[item[1] : signatures[0][0]]
                except Exception as e:
                    # logger.debug(f"filing{filing}, item: {item}, signatures: {signatures}")
                    if len(signatures) == 0:
                        # didnt find a signature so just assume content is until EOF
                        body = filing[item[1] :]
                # normalize item
                normalized_item = (
                    item[2]
                    .lower()
                    .replace(" ", "")
                    .replace(".", "")
                    .replace("\xa0", " ")
                )
                extracted_items.append({normalized_item: body})
            else:
                body = filing[item[1] : items[idx + 1][0]]
                normalized_item = (
                    item[2]
                    .lower()
                    .replace(" ", "")
                    .replace(".", "")
                    .replace("\xa0", " ")
                )
                extracted_items.append({normalized_item: body})
        return extracted_items

    def _parse_date_of_report(self, fdate):
        try:
            date = pd.to_datetime(fdate.replace(",", ", "))
        except Exception as e:
            logging.info(f"couldnt parse date of filing; date found: {fdate}")
            raise e
        return date

    def _create_match_group(self):
        reg_items = "("
        for key, val in ITEMS_8K.items():
            reg_items = reg_items + "(" + val + ")|"
        reg_items = reg_items[:-2] + "))"
        return re.compile(reg_items, re.I | re.DOTALL)


class ParserSC13D(HTMFilingParser):
    form_type = "SC 13D"
    extension = ".htm"

    def __init__(self):
        self.match_groups = self._create_match_group(ITEMS_SC13D)

    def split_into_sections(self, doc: str):
        """
        split the filing into FilingSections.

        Returns:
            list[HTMFilingSection] or []
        """
        items = self._parse_items(doc)
        logger.debug(f"SC 13D items found: {len(items)}")
        if items == []:
            return []
        else:
            sections = []
            for item in items:
                for k, v in item.items():
                    sections.append(
                        HTMFilingSection(
                            title=k,
                            content=v,
                            extension=self.extension,
                            form_type=self.form_type,
                        )
                    )
            return sections

    def _create_match_group(self, items_dict: dict):
        reg_items = "(?:"
        for key, val in items_dict.items():
            reg_items = reg_items + "(?:" + val + ")|"
        reg_items = reg_items[:-2] + "))"
        return re.compile(reg_items, re.I | re.MULTILINE)

    def get_item_matches(self, filing: str):
        """get matches for the sc 13d items.

        Args:
            filing: should be a cleaned sc 13d filing (only text content, no html ect)"""
        matches = []
        for match in re.finditer(self.match_groups, filing):
            matches.append([match.start(), match.end(), match.group(0)])
        return matches

    def get_signature_matches(self, filing: str):
        """get matches for the signatures"""
        signature_matches = []
        for smatch in re.finditer(re.compile("(signatures|signature)", re.I), filing):
            signature_matches.append(
                [smatch.start(), smatch.end(), smatch.start() - smatch.end()]
            )
        return signature_matches

    def _parse_items(self, filing: str):
        """extract the items from the filing and their associated paragraph"""
        # first get items and signatures
        extracted_items = []
        items = self.get_item_matches(filing)
        signatures = self.get_signature_matches(filing)
        # ensure that there is one signature which comes after all the items
        # otherwise discard the signature
        last_idx = len(items) - 1
        for idx, sig in enumerate(signatures):
            # logger.debug(sig)
            for item in items:
                if sig[1] < item[1]:
                    try:
                        signatures.pop(idx)
                    except IndexError as e:
                        raise e
                    except TypeError as e:
                        logger.debug(
                            (f"unhandled case of TypeError: ", sig, signatures, e),
                            exc_info=True,
                        )
            if len(signatures) == 0:
                continue
        for idx, item in enumerate(items):
            # last item
            if idx == last_idx:
                try:
                    body = filing[item[1] : signatures[0][0]]
                except Exception as e:
                    # logger.debug(f"filing{filing}, item: {item}, signatures: {signatures}")
                    if len(signatures) == 0:
                        # didnt find a signature so just assume content is until EOF
                        body = filing[item[1] :]
                # normalize item
                normalized_item = (
                    item[2]
                    .lower()
                    .replace(" ", "")
                    .replace(".", "")
                    .replace("\xa0", " ")
                )
                extracted_items.append({normalized_item: body})
            else:
                if idx == 0:
                    body = filing[: item[0]]
                    normalized_item = "before items"
                    extracted_items.append({normalized_item: body})

                body = filing[item[1] : items[idx + 1][0]]
                normalized_item = (
                    item[2]
                    .lower()
                    .replace(" ", "")
                    .replace(".", "")
                    .replace("\xa0", " ")
                )
                extracted_items.append({normalized_item: body})
        return extracted_items

    def classify_table(self, table: list[list]):
        """'classify a table into subcategories so they can
        be processed further.

        Args:
            table: should be a list of lists cleaned with clean_parsed_table.
        """
        table_shape = (len(table), len(table[0]))
        if table_shape[0] == 1:
            return "one_row_table"
        return None

    def _make_reintegrate_html_of_table(self, classification, table: list[list]):
        return None

    def extract_tables(
        self, soup: BeautifulSoup, reintegrate=["ul_bullet_points", "one_row_table"]
    ):
        return self._extract_tables(
            soup, MAIN_TABLE_ITEMS_SC13D, reintegrate=reintegrate
        )

    def _extract_tables(
        self,
        soup: BeautifulSoup,
        items_dict: dict,
        reintegrate=["ul_bullet_points", "one_row_table"],
    ):
        """
        extract the tables, parse them and return them as a dict of nested lists.

        side effect: modifies the soup attribute if reintegrate isnt an empty list

        Args:
            soup: BeautifulSoup of content
            reintegrate: which tables to reintegrate as text. if this is an empty list
                         all tables will be returned in the "extracted" section of the dict
                         and the "reintegrated" section will be an empty list
        Returns:
            a dict of form: {
                "reintegrated": [
                        {
                        "classification": classification,
                        "reintegrated_as": new elements that were added to the original doc
                        "table_meta": meta data dict for the extracted tables. contains at least
                                      a key of 'table_elements' which contains the original <table>
                                      elements as a list.
                        "parsed_table": parsed representation of the table before reintegration
                        }
                    ],
                "extracted": [
                        {
                        "classification": classification,
                        "table_meta": meta data dict for the extracted tables. contains at least
                                      a key of 'table_elements' which contains the original <table>
                                      elements as a list.
                        "parsed_table": parsed representation of table,
                        }
                    ]
                }
        """
        unparsed_tables = self.get_unparsed_tables(soup)
        tables = {"reintegrated": [], "extracted": []}
        current_main_table_item = 0
        multi_element_table_items = []
        multi_element_table_meta = {"table_elements": [], "items": []}
        for t in unparsed_tables:
            parsed_table = None
            if self.table_has_header(t):
                parsed_table = self.parse_htmltable_with_header(t)
            else:
                parsed_table = self.primitive_htmltable_parse(t)
            cleaned_table = self._clean_parsed_table_columnwise(
                self._preprocess_table(parsed_table)
            )
            if cleaned_table == []:
                continue
            classification = self.classify_table(cleaned_table)
            if classification == "unclassified":
                classification = super().classify_table(cleaned_table)
            if classification in reintegrate:
                reintegrate_html = self._make_reintegrate_html_of_table(
                    classification, cleaned_table
                )
                if reintegrate_html is None:
                    try:
                        reintegrate_html = super()._make_reintegrate_html_of_table(
                            classification, cleaned_table
                        )
                    except NotImplementedError:
                        tables["extracted"].append(
                            {
                                "classification": classification,
                                "table_meta": {"table_elements": [t]},
                                "parsed_table": cleaned_table,
                            }
                        )
                        logger.info(
                            f"couldnt reintegrate table, because this class of table isnt handled in the base class or this class with _make_reintegrate_html_of_table function. Extracted table instead. classification not handled: {classification}"
                        )
                        continue
                t.replace_with(reintegrate_html)
                tables["reintegrated"].append(
                    {
                        "classification": classification,
                        "reintegrated_as": reintegrate_html,
                        "table_meta": {"table_elements": [t]},
                        "parsed_table": cleaned_table,
                    }
                )
            else:
                if (_re_is_main_table_start(parsed_table, items_dict) is True) or (
                    current_main_table_item > 0
                ):
                    logger.debug(f"found part of a main table")
                    # logger.debug(cleaned_table)
                    new_current_item, extracted_items = _re_get_key_value_table(
                        parsed_table, items_dict, current_main_table_item
                    )
                    if extracted_items == [] or _list_is_true(
                        [
                            True if list(e.values())[0] == "" else False
                            for e in extracted_items
                        ]
                    ):
                        parsed_table = _parse_sc13_main_table_alternative(t)
                        for ridx, row in enumerate(parsed_table):
                            for fidx, field in enumerate(row):
                                if isinstance(field, str):
                                    parsed_table[ridx][fidx] = self.preprocess_text(
                                        field
                                    )

                        logger.debug(f"went alternative for table: {parsed_table}")
                        new_current_item, extracted_items = _re_get_key_value_table(
                            parsed_table, items_dict, current_main_table_item
                        )
                        logger.debug(extracted_items)
                        if extracted_items == []:
                            raise ValueError("incomplete main table")
                    current_main_table_item = new_current_item
                    multi_element_table_meta["table_elements"].append(t)
                    multi_element_table_meta["items"].append(extracted_items)
                    [multi_element_table_items.append(e) for e in extracted_items]
                    if current_main_table_item == len(items_dict.keys()):
                        logger.debug(f"completed the parse of a main table")
                        current_main_table_item = 0
                        tables["extracted"].append(
                            {
                                "classification": "main_table",
                                "table_meta": multi_element_table_meta,
                                "parsed_table": multi_element_table_items,
                            }
                        )
                        multi_element_table_items = []
                        current_main_table_item = 0
                tables["extracted"].append(
                    {
                        "classification": classification,
                        "table_meta": {"table_elements": [t]},
                        "parsed_table": cleaned_table,
                    }
                )
        if current_main_table_item > 0:
            logger.debug(
                (
                    f"failed to parse the main table completely."
                    f"items found so far: {multi_element_table_items}"
                )
            )
        return tables


class ParserSC13G(ParserSC13D):
    form_type = "SC 13G"
    extension = ".htm"

    def __init__(self):
        self.match_groups = self._create_match_group(ITEMS_SC13G)

    def split_into_sections(self, doc: str):
        """
        split the filing into FilingSections.

        Returns:
            list[HTMFilingSection] or []
        """
        items = self._parse_items(doc)
        logger.debug(f"SC 13G items found: {len(items)}")
        if items == []:
            return []
        else:
            sections = []
            for item in items:
                for k, v in item.items():
                    sections.append(
                        HTMFilingSection(
                            title=k,
                            content=v,
                            extension=self.extension,
                            form_type=self.form_type,
                        )
                    )
            return sections

    def extract_tables(
        self, soup: BeautifulSoup, reintegrate=["ul_bullet_points", "one_row_table"]
    ):
        return self._extract_tables(
            soup, MAIN_TABLE_ITEMS_SC13G, reintegrate=reintegrate
        )


class XMLFilingParser(AbstractFilingParser):
    form_type = None
    extension = ".xml"
    def split_into_sections(self, doc: ElementTree.ElementTree):
        raise NotImplementedError(f"This is meant to be a Base Class to be subclassed. Implement split_into_sections in the subclass.")
    
    def get_doc(self, path: str) -> ElementTree.ElementTree:
        return ElementTree.parse(path)

class XMLFilingSection(FilingSection):
    def __init__(self, content_dict: dict, **kwargs):
        super().__init__(**kwargs)
        self.content_dict = content_dict

class BaseFiling(Filing):
    def __init__(
        self,
        accession_number: str,
        path: str,
        cik: str,
        form_type: str,
        filing_date: str = None,
        file_number: str = None,
        extension: str = None,
        doc=None,
        sections: list[FilingSection] = None
    ):
        super().__init__(
            path=path,
            filing_date=filing_date,
            accession_number=accession_number,
            cik=cik,
            file_number=file_number,
            form_type=form_type,
            extension=extension
        )
        self.parser: AbstractFilingParser = parser_factory.get_parser(
            extension=self.extension,
            form_type=self.form_type
        )
        self.doc = self.parser.get_doc(self.path) if doc is None else doc
        self.sections = (
            self.parser.split_into_sections(self.doc) if sections is None else sections
        )

    

class ParserEFFECT(XMLFilingParser):
    form_type = "EFFECT"
    extension = ".xml"
    
    def split_into_sections(self, doc: ElementTree.ElementTree) -> list[XMLFilingSection]:
        try:
            effective_data: ElementTree.Element = doc.find(".//effectiveData")
            # print([i for i in effective_data])
            # print(effective_data.get('form'))
            # print(effective_data.find(".//form"))
            for_form = self._get_for_form(effective_data)
            print(f"for_form value: {for_form}")
            content_dict = {
                "for_form": for_form,
                "effective_date": doc.find(".//finalEffectivenessDispDate").text,
                "file_number": doc.find(".//fileNumber").text,
                "cik": doc.find(".//cik").text,
            }
        except AttributeError as e:
            logger.warning(f"failed to split this filing correctly, returning []. Exception caught: {e}", exc_info=True)
            return []
        else:
            sections = [
                XMLFilingSection(
                    content_dict=content_dict,
                    title="main",
                    content=str(doc)
                    )
                ]
            return sections

    def _get_for_form(self, element: ElementTree.Element) -> str|None:
        # find correct way to access the children
        print(f"checking for form element in {[(i.text, i) for i in element]}")
        # print(f"elemnt form: {element.find('form').text}")
        if isinstance(element.find("form"), ElementTree.Element):
            return element.find("form").text
        if isinstance(element.find('submissionType'), ElementTree.Element):
            return element.find("submissionType").text
        return None

def _re_is_main_table_start(table: list[list], items_dict: dict):
    # check if this is the start of the main table
    for row in table:
        for field in row:
            if table_field_contains_content(
                field, re.compile(items_dict[list(items_dict.keys())[0]], re.I)
            ):
                return True
    return False


def _row_is_ignore(row: list, ignore: set = set(["", None])):
    row_set = set(row)
    if len(row_set - ignore) == 0:
        return True
    else:
        return False


def _re_get_key_value_table(table: list[list], items_dict: dict, current_item: int):
    """extract the key value from items dict from table.
    Args:
        current_item: index of item in items_dict to start from
    Returns:
        current_item: item idx of last item
        items: list[dicts]
    """
    items = []
    keys = list(items_dict.keys())
    for row in table:
        for field in row:
            if current_item > (len(keys) - 1):
                return current_item, items
            if field is not None:
                match = _extract_field_value(
                    field,
                    re.compile(
                        items_dict[keys[current_item]] + "(.*)", re.I | re.DOTALL
                    ),
                )
                if match is not None:
                    items.append({keys[current_item]: match})
                    current_item += 1
    return current_item, items


def _parse_sc13_main_table_alternative(htmltable: element.Tag):
    """alternative way to parse a sc13 main table"""
    rows = htmltable.find_all("tr")
    offset = None
    table = []
    rowcount = -1
    for row in rows:
        cells = row.find_all(["td", "th"], recursive=False)
        if (offset is None) or (offset == 0):
            rowcount += 1
            if re.search(
                re.compile("NUMBER(?:.){,3}OF(?:.){,3}SHARES", re.I | re.DOTALL),
                cells[0].get_text(strip=True),
            ):
                offset = int(cells[1].get("rowspan", 1))
            else:
                offset = int(cells[0].get("rowspan", 1))
            table.append([])
        [table[rowcount].append(c.get_text(strip=True)) for c in cells]
        offset -= 1
    return [[" ".join(x)] for x in table]


def _get_cusip(field: str):
    """check if field contains cusip and extract if so."""
    match = re.search(
        field,
        re.compile(
            "([a-z0-9]{6}(?:\s{0,3})(?:(?:[a-z0-9]{3})|(?:[a-z0-9]{2}(?:\s){0,3})[a-z0-9]{1}))(?:\s)*\((?:\s){0,3}cusip"
        ),
    )
    if match:
        return match.group(1)
    else:
        return None


def _extract_field_value(field, re_term):
    """extract a key value pair from a re term where groups()[0] is key and groups()[1] is value."""
    match = re.search(re_term, field)
    if match:
        return match.group(2)
    else:
        # print(f"no match found in: {field} with re_term: {re_term}")
        return None


def _list_is_true(entries: list[bool]):
    """checks if list has only True for values"""
    for entry in entries:
        if entry is True:
            pass
        else:
            return False
    return True


def table_field_contains_content(field, re_term):
    """checks if re.search matches in field and returns boolean"""
    if not isinstance(field, str):
        return False
    else:
        if re.search(re_term, field):
            return True
        else:
            return False


def table_header_has_fields(table_header: list, re_terms: list[re.Pattern]) -> bool:
    """checks if all fields are present in the header"""
    header_matches = []
    for field in table_header:
        for re_term in re_terms:
            if table_field_contains_content(field, re_term):
                header_matches.append(True)
    return _list_is_true(header_matches) and (len(header_matches) >= len(re_terms))


parser_factory_default = [
    (".htm", "8-K", Parser8K),
    (".htm", "SC 13D", ParserSC13D),
    (".htm", "SC 13G", ParserSC13G),
    (".htm", "S-3", ParserS3),
    (".xml", "EFFECT", ParserEFFECT)

]
parser_factory = ParserFactory(defaults=parser_factory_default)


class HTMFilingSection(FilingSection):
    def __init__(self, title, content, extension: str = None, form_type: str = None):
        super().__init__(title=title, content=content)
        self.parser: HTMFilingParser = parser_factory.get_parser(
            extension=extension, form_type=form_type
        )
        self.soup: BeautifulSoup = self.parser.make_soup(self.content)
        self.tables: dict = self.parser.extract_tables(self.soup)
        self.text_only = self.parser.preprocess_section_text_content(
            self.parser.get_text_content(
                self.soup, exclude=["table", "script", "title", "head"]
            )
        )

    def quick_summary(self) -> dict:
        """returns a short summary of the section."""
        return {
            "title": self.title,
            "text_only_length": len(self.text_only),
            "tables_extracted": len(self.tables["extracted"]),
            "tables_reintegrated": len(self.tables["reintegrated"]),
        }

    def get_tables(
        self, classification: str = "unclassified", table_type: str = "extracted"
    ) -> list:
        """
        Gets tables by table_type and classification.

        Args:
            classification: "all" or any classification declared with classify_table
            table_type: either "extracted" or "reintegrated"

        Returns:
            list: if tables matching the classification and table_type where found
            None: if no tables were found
        """
        tables = []
        for table in self.tables[table_type]:
            if (table["classification"] == classification) or (classification == "all"):
                tables.append(table)
        if len(tables) > 0:
            return tables
        else:
            return None


class BaseHTMFiling(BaseFiling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soup: BeautifulSoup = self.parser.make_soup(self.doc)

    def get_preprocessed_text_content(self) -> str:
        """get all the text content of the Filing"""
        return self.parser.preprocess_text(self.doc)

    def get_text_only(self):
        text = " ".join([sec.text_only for sec in self.sections])
        return text

    def get_preprocessed_section(self, identifier: str | int):
        section = self.get_section(identifier=identifier)
        if section != []:
            return self.preprocess_section(section)
        else:
            return []

    def get_section(self, identifier: str | int | re.Pattern):
        """
        gets a section either by index or by exact title match.

        Returns:
            FilingSection or [], if no section was found."""
        if not isinstance(identifier, (str, int, re.Pattern)):
            raise ValueError
        if self.sections == []:
            return []
        if isinstance(identifier, int):
            try:
                self.sections[identifier]
            except IndexError:
                logger.info("no section with that identifier found")
                return []
        elif isinstance(identifier, str):
            for sec in self.sections:
                if sec.title == identifier:
                    return sec
        elif isinstance(identifier, re.Pattern):
            for sec in self.sections:
                if re.search(identifier, sec.title):
                    return sec
        return []

    def get_sections(self, identifier: str | re.Pattern):
        """gets sections based on a re.search of identifier.
        Returns:
            list[FilingSection] or [], if no matching sections were found."""
        sec_matches = []
        for sec in self.sections:
            if re.search(identifier, sec.title):
                sec_matches.append(sec)
        return sec_matches

    def preprocess_section(self, section: HTMFilingSection):
        text_content = self.parser.make_soup(section.content).getText(
            separator=" ", strip=True
        )
        return self.parser.preprocess_section_content(text_content)


class HTMFilingBuilder:
    def _split_into_filings(
        self,
        form_type: str,
        extension: str,
        path: str,
        filing_date: str,
        accession_number: str,
        cik: str,
        file_number: str,
    ) -> list[Filing]:
        parser: AbstractFilingParser = parser_factory.get_parser(
            extension=extension, form_type=form_type
        )
        doc = parser.get_doc(path)
        sections = parser.split_into_sections(doc)
        (
            is_multiprospectus_filing,
            cover_pages,
        ) = self._is_multiprospectus_registration_statement(sections)
        if is_multiprospectus_filing is True:
            logger.debug("This is a multiprospectus filing.")
            filings = []
            start_idx = 0
            # skip first cover page so we include front page in the base prospectus
            for idx, cover_page in enumerate(cover_pages[1:]):
                for sidx, section in enumerate(sections):
                    if section == cover_page:
                        filings.append(
                            BaseHTMFiling(
                                path=path,
                                filing_date=filing_date,
                                accession_number=accession_number,
                                cik=cik,
                                file_number=file_number,
                                form_type=form_type,
                                extension=extension,
                                doc=doc,
                                sections=sections[start_idx:sidx],
                            )
                        )
                        start_idx = sidx
            # add last prospectus
            filings.append(
                BaseHTMFiling(
                    path=path,
                    filing_date=filing_date,
                    accession_number=accession_number,
                    cik=cik,
                    file_number=file_number,
                    form_type=form_type,
                    extension=extension,
                    doc=doc,
                    sections=sections[start_idx:],
                )
            )
            return filings
        else:
            logger.debug("This is a single prospectus filing.")
            return BaseHTMFiling(
                path=path,
                filing_date=filing_date,
                accession_number=accession_number,
                cik=cik,
                file_number=file_number,
                form_type=form_type,
                extension=extension,
                doc=doc,
                sections=sections,
            )

    def _select_sections(self, re_term: re.Pattern, sections: list[HTMFilingSection]):
        selected = []
        for section in sections:
            if re.search(re_term, section.title):
                selected.append(section)
        return selected

    def _is_multiprospectus_registration_statement(
        self, sections: list[HTMFilingSection]
    ):
        cover_pages = self._select_sections(re.compile("cover page", re.I), sections)
        if len(cover_pages) > 1:
            return True, cover_pages
        else:
            return False, None


def create_htm_filing(
    form_type: str,
    extension: str,
    path: str,
    filing_date: str,
    accession_number: str,
    cik: str,
    file_number: str,
) -> list[Filing]:
    """handle multi prospectus filings"""
    builder = HTMFilingBuilder()
    return builder._split_into_filings(
        form_type, extension, path, filing_date, accession_number, cik, file_number
    )


filing_factory_default = [
    ("S-1", ".htm", BaseHTMFiling),
    ("DEF 14A", ".htm", BaseHTMFiling),
    ("8-K", ".htm", BaseHTMFiling),
    ("SC 13D", ".htm", BaseHTMFiling),
    ("SC 13G", ".htm", BaseHTMFiling),
    ("S-3", ".htm", create_htm_filing),
    ("EFFECT", ".xml", BaseFiling)
]

filing_factory = FilingFactory(defaults=filing_factory_default)
