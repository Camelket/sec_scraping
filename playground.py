from cgitb import text
from sec_edgar_downloader import Downloader
from pathlib import Path
import re
from bs4 import BeautifulSoup

from xbrl_parser import *
from xbrl_structure import *
from filing_handler import *

from datetime import datetime


# dl = Downloader(Path(r"C:\Users\Olivi\Testing\sec_scraping_testing\filings"))
# aapl_10ks = dl.get("10-Q", "PHUN", amount=10)

r""" with open(r"C:\Users\Olivi\Testing\sec_scraping\filings\sec-edgar-filings\PHUN\424B3\0001213900-18-015809\full-submission.txt", "r") as f:
    text = f.read()
    metaparser = TextMetaParser()
    metadata_doc = metaparser.process_document_metadata(text)
    logging.debug(metadata_doc)
    pass """

# TEST FOR ParserXBRL
# path = r"C:\Users\Olivi\Testing\sec_scraping\filings\sec-edgar-filings\PHUN\10-Q\0001213900-19-008896"
paths = sorted(Path(r"C:\Users\Olivi\Testing\sec_scraping\filings\sec-edgar-filings\PHUN\10-Q").glob("*"))
handler = FilingHandler()
shares_outstanding = []
for p in paths[:2]:
# path = r"C:\Users\Olivi\Testing\sec_scraping_testing\filings\sec-edgar-filings\PHUN\10-Q\0001628280-21-023228" 
    # fts = FullTextSubmission(p)
    now = datetime.now()
    p = p / "full-submission.txt"
    with open(p, "r") as f:
        full_text = f.read()
        doc = handler.preprocess_documents(full_text)
        file = handler.process_file(doc)
        
        
        facts = file.search_fact("Commonstocksharesoutstanding", None, namespace="us-gaap")
        # print([str(f) for f in facts])
        # if file:
        #     matches = file.search_for_key(re.compile("sharesoutstanding", re.I))
        #     for m in matches:
        #         print(file.facts[m])
        #         print([f.get_members() for f in file.facts[m]])
        elapsed = datetime.now() - now
        # print([f.convert_to_dict() for f in facts])
        # print(f"time for processing: {elapsed}")
    # else:
    #     print(p, mode)

fact1 = Fact(
            Tag("us", "common"),
            Context(id="1", entity=Entity(identifier="1"), period=Instant("2020-09-03")),
            Value("29", unit=Unit("usd"))
            )

fact2 = Fact(
            Tag("us", "common"),
            Context(id="1", entity=Entity(identifier="1"), period=Instant("2020-09-03")),
            Value("29", unit=Unit("usd"))
            )

print(f"fact1: {fact1.__repr__}, fact2: {fact2.__repr__}")
print((fact1 == fact2))
print(fact1.__eq__(fact2))
print(fact1.__dict__)
print(fact2.__dict__)

# TEST FOR ParserS1
# S1_path = r"C:\Users\Olivi\Testing\sec_scraping_testing\filings\sec-edgar-filings\PHUN\S-1\0001213900-16-014630\filing-details.html"
# with open(S1_path, "rb") as f:
#     text = f.read()
#     parser = ParserS1()
#     parser.make_soup(text)
#     parser.get_unparsed_tables()
#     registration_fee_table = parser.get_calculation_of_registration_fee_table()
#     print(registration_fee_table[0])
#     df = pd.DataFrame(columns=registration_fee_table[0], data=registration_fee_table[1:])
#     print(df)

# TEST FOR Parser424B5
# with open(r"C:\Users\Olivi\Testing\sec_scraping\filings\sec-edgar-filings\PHUN\424B5\0001628280-20-012767\filing-details.html", "rb") as f:
#     text = f.read()
#     parser = Parser424B5()
#     parser.make_soup(text)
#     # print(parser.get_offering_table())
#     print(parser.get_dilution_table())
#         # print(parser.get_tables())


