from cgitb import text
from sec_edgar_downloader import Downloader
from pathlib import Path
import re
from bs4 import BeautifulSoup

from xbrl_parser import *
from xbrl_structure import *
from filing_handler import *

import time


# dl = Downloader(Path(r"C:\Users\Olivi\Testing\sec_scraping_testing\filings"))
# aapl_10ks = dl.get("10-Q", "PHUN", amount=10)

r""" with open(r"C:\Users\Olivi\Testing\sec_scraping\filings\sec-edgar-filings\PHUN\424B3\0001213900-18-015809\full-submission.txt", "r") as f:
    text = f.read()
    metaparser = TextMetaParser()
    metadata_doc = metaparser.process_document_metadata(text)
    logging.debug(metadata_doc)
    pass """

# TEST FOR FullTextSubmission
# path = r"C:\Users\Olivi\Testing\sec_scraping\filings\sec-edgar-filings\PHUN\10-Q\0001213900-19-008896"
paths = sorted(Path(r"C:\Users\Olivi\Testing\sec_scraping\filings\sec-edgar-filings\PHUN\10-Q").glob("*"))
handler = FilingHandler()
for p in paths:
# path = r"C:\Users\Olivi\Testing\sec_scraping_testing\filings\sec-edgar-filings\PHUN\10-Q\0001628280-21-023228" 
    # fts = FullTextSubmission(p)
    p = p / "full-submission.txt"
    with open(p, "r") as f:
        now = time.now()
        full_text = f.read()
        doc = handler.preprocess_documents(full_text)
        file = handler.process_file(doc)
        elapsed = time.now() - now
        print(f"elapsed: {elapsed}")
    # if file:
    #     matches = file.search_for_key(re.compile("sharesoutstanding", re.I))
    #     for m in matches:
    #         print(file.facts[m])
    # else:
    #     print(p, mode)

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


