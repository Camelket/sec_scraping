from downloader import Downloader #added to extra_path with vscode



'''aggregate following data:

resources: 
https://xbrl.us/data-rule/guid-cashflowspr/#5
https://asc.fasb.org/viewpage

- Outstanding shares [value, instant] us-gaap:CommonStockSharesOutstanding, EntityCommonStockSharesOutstanding ?

- cash and equiv. NET at end of period [value, instant] us-gaap:[
    CashAndCashEquivalentsPeriodIncreaseDecrease,
    CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect,
    CashAndCashEquivalentsPeriodIncreaseDecreaseExcludingExchangeRateEffect,
    CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseExcludingExchangeRateEffect,
    CashPeriodIncreaseDecrease,
    CashPeriodIncreaseDecreaseExcludingExchangeRateEffect
]
capital rais in a period seen on sheet as proceeds from issuance of...
look for us-gaap elements that represent atms, warrants, notes ect.
- s-1 shelves
- s-3 shelves
- warrants - keys: ProceedsFromIssuanceOfWarrants
- notes
- ATM's
- what is considered to be a private placement - keys us-gaap: ProceedsFromIssuanceOfPrivatePlacement
- issuance of common stock [value, from, to] us-gaap:ProceedsFromIssuanceOfCommonStock
 CommonStockSharesIssued
- 

Unit key I want is USD so a fact i want is accessed by using json["facts"][taxonomy][fact_name][unit]
for issuance of inital public offering take the filing date as the instant of the value
for issuance of 
'''
import re

class Fact:
    def __init__(self):
        self.taxonomy = None
        self.name = None
        self.value = None
        self.unit = None
        self.instant = None
        self.period = None
        self.filing_date = None
        self.fp = None
        self.fy = None
        self.accn = None
        self.form = None
    
    def __repr__(self):
        return self.name +":"+ str(self.value) +":"+ self.unit +"\n"


def get_fact_data(companyfacts, name, taxonomy, unit="USD"):
    facts = []
    data_points = companyfacts["facts"][taxonomy] #[name] #[unit]
    searchterm = None
    if isinstance(name, re.Pattern):
        for d in data_points:
            fname = re.search(name, d)
            if fname:
                facts.append(companyfacts["facts"][taxonomy][fname])
    else:
        for d in data_points:
            fname = re.search(re.compile("(.*)("+name+")(.*)", re.I), d)
            if fname:
                fstring = fname.string
                for unit in companyfacts["facts"][taxonomy][fstring]["units"]:
                    for single_fact in companyfacts["facts"][taxonomy][fstring]["units"][unit]:
                        print(single_fact)
                        f = Fact()
                        f.taxonomy = taxonomy
                        f.name = fstring
                        f.value = single_fact["val"]
                        f.unit = unit
                        f.filing_date = single_fact["filed"]
                        f.fy = single_fact["fy"]
                        f.fp = single_fact["fp"]
                        f.form = single_fact["form"]
                        f.accn = single_fact["accn"]
                        if "start" in single_fact.keys():
                            f.period = {"start": single_fact["start"], "end": single_fact["end"]}
                        else:
                            f.instant = single_fact["instant"]
                        
                        facts.append(f)
            # f = Fact()
            # f.taxonomy = taxonomy
            # f.name = name
            # f.unit = unit
            # if ("start" and "end") in d.keys():
            #     f.period = {"start": d["start"], "end": d["end"]}
            #     # f.period = Period(start=d["start"], end=d["end"])
            # print(d)
    return facts

'''what: a fact that contains taxonomy, name, value, period, filing_date
--> how will i use it?
    --> pass it to db to use in calculations
    
    --> what do i need to write it to db?
        the facts attributes and that is it: construct a
        function that inserts into the right tables based
        on fact name!
        --> what to consider when inserting?
        what time interests me? start, end or the filing_date (unless it is an instance where i only have to decide between filing_date and instance time)
    does it make sense to use a Fact class or am i better
    served with a dictionary or do i need to save space and
    use list of lists ?
    perform validation of period or instance where?
    

    ---> dictionary will suffice
    

'''


        
import json
import re
from pathlib import Path

dl = Downloader(r"C:\Users\Olivi\Testing\sec_scraping_testing\pysec_downloader\companyfacts", user_agent="john smith js@test.com")
print(len(dl._lookuptable_ticker_cik.keys()))
# symb = ["PHUN", "GNUS", "IMPP"]
# for s in symb:
#     j = dl.get_xbrl_companyfacts(s)
#     with open((dl.root_path / (s +".json")), "w") as f:
#         json.dump(j, f)

# with open(Path(r"C:\Users\Olivi\Testing\sec_scraping\pysec_downloader\companyfacts") / ("PHUN" + ".json"), "r") as f:
#     j = json.load(f)
#     f = get_fact_data(j, "ProceedsFromIssuance", "us-gaap")
#     print(f)

# for s in symb:
#         # j = dl.get_xbrl_companyfacts(s)
#         # with open((dl.root_path / (s +".json")), "w") as f:
#         #     json.dump(j, f)
#         with open(Path(r"C:\Users\Olivi\Testing\sec_scraping_testing\pysec_downloader\companyfacts") / (s + ".json"), "r") as f:
#             j = json.load(f)
#             matches = []
#             for each in j["facts"].keys():
#                 for possible in j["facts"][each]:
                    
#                     if re.search(re.compile("ProceedsFromIssuance(.*)", re.I), possible):
#                         matches.append(possible)
#             print([j["facts"]["us-gaap"][p] for p in matches])