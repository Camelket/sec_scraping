
import queue
from typing import Optional
from bs4 import BeautifulSoup
from matplotlib import docstring
from numpy import number
from urllib3 import connection_from_url
from dilution_db import DilutionDBUpdater
from main.adapters.repository import SqlAlchemyCompanyRepository
from main.parser.extractors import UnhandledClassificationError
# from dilution_db import DilutionDB
# from main.data_aggregation.polygon_basic import PolygonClient
# from main.configs import cnf

# from main.data_aggregation.bulk_files import update_bulk_files


from main.parser.parsers import HTMFilingParser, Parser8K, ParserSC13D, BaseHTMFiling
from pathlib import Path

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.errors import ProgrammingError, UniqueViolation, ForeignKeyViolation
from psycopg_pool import ConnectionPool

import pandas as pd
import logging
import webbrowser

from main.services.unit_of_work import SqlAlchemyCompanyUnitOfWork
logging.basicConfig(level=logging.DEBUG)



class GenericDB:
    def __init__(self, connectionString):
        self.connectionString = connectionString
        self.pool = ConnectionPool(
            self.connectionString, kwargs={"row_factory": dict_row}
        )
        self.conn = self.pool.connection
    
    def execute_sql(self, path):
        with self.conn() as c:
            with open(path, "r") as sql:
                res = c.execute(sql.read())
                try:
                    for row in res:
                        print(row)
                except ProgrammingError:
                    pass
    
    def read(self, query, values):
        with self.conn() as c:
            res = c.execute(query, values)
            rows = [row for row in res]
            return rows

class FilingDB(GenericDB):
    def __init__(self, *args):
        super().__init__(*args)
        self.parser8k = Parser8K()
        self.items8k = [
            'item101',
            'item102',
            'item103',
            'item104',
            'item201',
            'item202',
            'item203',
            'item204',
            'item205',
            'item206',
            'item301',
            'item302',
            'item303',
            'item401',
            'item402',
            'item501',
            'item502',
            'item503',
            'item504',
            'item505',
            'item506',
            'item507',
            'item508',
            'item601',
            'item602',
            'item603',
            'item604',
            'item605',
            'item701',
            'item801',
            'item901']
    
    def normalize_8kitem(self, item: str):
        '''normalize and extract the first 7 chars to assign to one specific item'''
        return item.lower().replace(" ", "").replace(".", "").replace("\xa0", " ").replace("\n", "")[:7]
    
    def get_items_count_summary(self):
        entries = self.read("SELECT f.item_id as item_id, f.file_date, i.item_name as item_name FROM form8k as f JOIN items8k as i ON i.id = f.item_id", [])
        summary = {}
        for e in entries:
            if e["item_name"] not in summary.keys():
                summary[e["item_name"]] = 0
            else:
                summary[e["item_name"]] += 1

        return summary
    
    def init_8k_items(self):
        with self.conn() as connection:
            for i in self.items8k:
                connection.execute("INSERT INTO items8k(item_name) VALUES(%s)",[i])
    
    
    def add_8k_content(self, cik, file_date, item, content):
        normalized_item = self.normalize_8kitem(item)
        if normalized_item not in self.items8k:
            logging.info(f"skipping 8-k section, because couldnt find a valid item in it, item found: {item}. cik:{cik}; file_date:{file_date}; content:{content}")
            return
        with self.conn() as connection:
            # print(item)
            connection.execute("INSERT INTO form8k(cik, file_date, item_id, content) VALUES(%(cik)s, %(file_date)s, (SELECT id from items8k WHERE item_name = %(item)s), %(content)s) ON CONFLICT ON CONSTRAINT unique_entry DO NOTHING",
            {"cik": cik,
            "file_date": file_date,
            "item": normalized_item,
            "content": content})
    
    def parse_and_add_all_8k_content(self, paths):
        for p in tqdm(paths, mininterval=1):
            split_8k = self.parser8k.split_into_items(p, get_cik=True)
            if split_8k is None:
                print(f"failed to split into items or get date: {p}")
                continue
            cik = split_8k["cik"]
            file_date = split_8k["file_date"]
            for entries in split_8k["items"]:
                for item, content in entries.items():
                    self.add_8k_content(
                        cik,
                        file_date,
                        item,
                        content)


def get_all_8k(root_path):
        '''get all .htm files in the 8-k subdirectories. entry point is the root path of /filings'''
        paths_folder = [r.glob("8-K") for r in (Path(root_path)).glob("*")]
        paths_folder = [[f for f in r] for r in paths_folder]
        # print(paths_folder)
        paths_files = [[f.rglob("*.htm") for f in r] for r in paths_folder]
        paths = []
        for l in paths_files:
            # print(l)
            for each in l:
                # print(each)
                for r in each:
                    # print(r)
                    paths.append(r)
        return paths

def flatten(lol):
    '''flatten a list of lists (lol).'''
    if len(lol) == 0:
        return lol
    if isinstance(lol[0], list):
        return flatten(lol[0]) + flatten(lol[1:])
    return lol[:1] + flatten(lol[1:])

def get_all_filings_path(root_path: Path, form_type: str):
    '''get all files in the "form_type" subdirectories. entry point is the root path /filings'''
    paths_folder = [r.glob(form_type) for r in (Path(root_path)).glob("*")]
    print(list(paths_folder))
    form_folders = flatten([[f for f in r] for r in paths_folder])
    file_folders = flatten([list(r.glob("*")) for r in form_folders])
    paths = flatten([list(r.glob("*")) for r in file_folders])
    return paths




if __name__ == "__main__":
    

    from main.configs import cnf
    from pysec_downloader.downloader import Downloader
    from tqdm import tqdm
    from dilution_db import DilutionDB
    import json
    import csv
    import spacy
    import datetime
    from spacy import displacy
    import main.parser.extractors as extractors

    # nlp = spacy.load("en_core_web_sm")

    # db = DilutionDB(cnf.DILUTION_DB_CONNECTION_STRING)
    # dl = Downloader(cnf.DOWNLOADER_ROOT_PATH, retries=100, user_agent=cnf.SEC_USER_AGENT)
    # import logging
    # logging.basicConfig(level=logging.INFO)
    # dlog = logging.getLogger("urllib3.connectionpool")
    # dlog.setLevel(logging.CRITICAL)
    # with open("./resources/company_tickers.json", "r") as f:
    #     tickers = list(json.load(f).keys())
    #     for ticker in tqdm(tickers):
    #         db.util.get_filing_set(dl, ticker, cnf.APP_CONFIG.TRACKED_FORMS, "2017-01-01", number_of_filings=10)

    # with open("./resources/company_tickers.json", "r") as f:
    #     tickers = list(json.load(f).keys())
    #     db.util.get_overview_files(cnf.DOWNLOADER_ROOT_PATH, cnf.POLYGON_OVERVIEW_FILES_PATH, cnf.POLYGON_API_KEY, tickers)

# delete and recreate tables, populate 8-k item names
# extract item:content pairs from all 8-k filings in the downloader_root_path
# and add them to the database (currently not having filing_date, 
# important for querying the results later) if not other way, get filing date 
# by using cik and accn to query the submissions file 

    def init_fdb():
        connection_string = "postgres://postgres:admin@localhost/postgres"
        fdb = FilingDB(connection_string)

        fdb.execute_sql("./main/sql/db_delete_all_tables.sql")
        fdb.execute_sql("./main/sql/filings_db_schema.sql")
        fdb.init_8k_items()
        paths = []
        with open(r"E:\pysec_test_folder\paths.txt", "r") as pf:
            while len(paths) < 20000:
                paths.append(pf.readline().strip("\n").lstrip("."))
        # paths = get_all_8k(Path(r"E:\sec_scraping\resources\datasets") / "filings")
        fdb.parse_and_add_all_8k_content(paths)
        return fdb

    def retrieve_data_set():
        data = fdb.read("SELECT f.item_id as item_id, f.file_date, i.item_name as item_name, f.content FROM form8k as f JOIN items8k as i ON i.id = f.item_id WHERE item_name = 'item801' AND f.file_date > %s ORDER BY f.file_date LIMIT 1", [datetime.datetime(2021, 1, 1)])
        j = []
        for d in data:
            pass
        with open(r"E:\pysec_test_folder\k8s1v1.txt", "w", encoding="utf-8") as f:
            # json.dump(j, f)
            for r in data:
                    f.write(r["content"].replace("\n", " ") + "\n")

    # init_fdb()
    # items = parser.split_into_items(r"E:\sec_scraping\resources\datasets\filings\0000023197\8-K\0000023197-20-000144\cmtl-20201130.htm")
    # print(items)
    # retrieve_data_set()
    # print(fdb.get_items_count_summary())

    def try_spacy():
        texts = fdb.read("SELECT f.item_id as item_id, f.file_date, i.item_name as item_name, f.content FROM form8k as f JOIN items8k as i ON i.id = f.item_id WHERE item_name = 'item801' ORDER BY f.file_date LIMIT 30", [])
        text = ""
        contents = [text["content"] for text in texts]
        for content in nlp.pipe(contents, disable=["attribute_ruler", "lemmatizer", "ner"]):
            # text = text + "\n\n" + content
            print([s for s in content.sents])
        # displacy.serve(doc, style="ent")
            
    # try_spacy()

    import re
    from datetime import timedelta
    import pickle
    def download_samples(root, forms=["S-1", "S-3", "SC13D"], num_tickers=100, max_filings=20):
        import random
        random.seed(a=32525)
        dl = Downloader(root, user_agent="P licker p@licker.com")
        def get_filing_set(downloader: Downloader, ticker: str, forms: list, after: str, number_of_filings: int = 250):
            # # download the last 2 years of relevant filings
            if after is None:
                after = str((datetime.now() - timedelta(weeks=104)).date())
            for form in forms:
            #     # add check for existing file in pysec_donwloader so i dont download file twice
                try:
                    downloader.get_filings(ticker, form, after, number_of_filings=number_of_filings)
                except Exception as e:
                    print((ticker, form, e))
                    pass
        with open("./resources/company_tickers.json", "r") as f:
            all_tickers = list(json.load(f).keys())
            tickers = random.sample(all_tickers, k=num_tickers)
            for ticker in tqdm(tickers):
                get_filing_set(dl, ticker, forms, "2017-01-01", number_of_filings=max_filings)
    
    def open_filings_in_browser(root: str, form: str, max=100):
        import webbrowser
        paths = get_all_filings_path(root, form_type=form)
        for idx, p in enumerate(paths):
            if idx > max:
                return
            webbrowser.open(p, new=2)
            

    def store(path: str, obj: list):
        with open(path, "wb") as f:
                pickle.dump(obj, f)

    def retrieve(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def try_htmlparser():
        root_lap = Path(r"C:\Users\Public\Desktop\sec_scraping_testsets\example_filing_set_100_companies\filings")
        root_des = Path(r"F:\example_filing_set_100_companies\filings")
        parser = HTMFilingParser()
        root_filing = root_des
        file_paths = get_all_filings_path(Path(root_filing), "DEF 14A")
        # file_paths2 = get_all_filings_path(Path(root_filing), "S-3")
        # # file_paths3 = get_all_filings_path(Path(root_filing), "S-1")
        # file_paths.append(file_paths2)
        # # file_paths.append(file_paths3)
        file_paths = flatten(file_paths)
        # store(r"F:\example_filing_set_100_companies\s1s3paths.txt", file_paths)$
        # file_paths = retrieve(root_lap)
        # for p in [r"F:\example_filing_set_100_companies\filings\0001556266\S-3\0001213900-20-018486\ea124224-s3a1_tdholdings.htm"]:
        file_count = 0
    
        # main = []

        html = ('<p style="font: 10pt Times New Roman, Times, Serif; margin-top: 0pt; margin-bottom: 0pt; text-align: center"><font style="font-family: Times New Roman, Times, Serif"><b>THE BOARD OF DIRECTORS UNANIMOUSLY RECOMMENDS A VOTE “FOR” THE ELECTION OF <br> ALL FIVE NOMINEES LISTED BELOW.</b></font></p>')
        # soup = BeautifulSoup(html, features="html5lib")
        # print(soup.select("[style*='text-align: center' i]  b"))
        # print(parser._get_possible_headers_based_on_style(soup, ignore_toc=False))
        # print(soup)
        
        # soup2 = BeautifulSoup("", features="html5lib")
        # print(soup2)
        # from bs4 import element
        
        # new_tag = soup2.new_tag("p")
        # new_tag.string = "START TAG"
        # first_line = soup2.new_tag("p")
        # first_line.string = "this is the first line"
        # new_tag.insert(1,first_line)

        # soup.find("table").replace_with(new_tag)
        # print(soup)
        
        # for p in [r"C:/Users/Public/Desktop/sec_scraping_testsets/example_filing_set_100_companies/filings/0001731727/DEF 14A/0001213900-21-063709/ea151593-def14a_lmpautomot.htm"]:
        

        for p in file_paths[0:1]:
            file_count += 1
            with open(p, "r", encoding="utf-8") as f:
                file_content = f.read()
                print(p)
                # soup = parser.make_soup(file_content)
                # headers = parser._format_matches_based_on_style(parser._get_possible_headers_based_on_style(soup, ignore_toc=False))
                # for h in headers:
                #     print(h)
                filing = HTMFiling(file_content, path=p, form_type="S-1")

                # ??? why does it skip the adding of ele group ?
                print(f"FILING: {filing}")
                print([(s.title, len(s.text_only)) for s in filing.sections])
                try:
                    sections = sum([filing.get_sections(ident) for ident in ["share ownership", "beneficial"]], [])
                    for sec in sections:
                        print(sec.title)
                        print([pd.DataFrame(s["parsed_table"]) for s in sec.tables["extracted"]])
                        # print(sec.text_only)
                        print("\n")
                        # print([pd.DataFrame(s["parsed_table"]) for s in sec.tables["extracted"]])
                except KeyError:
                    print([s.title for s in filing.sections])
                except IndexError:
                    print([s.title for s in filing.sections])

                # print(filing.get_sections("beneficial"))
                # print([fil.title for fil in filing.sections])
                # pro_summary = filing.get_section("prospectus summary")
                # print([(par["parsed_table"], par["reintegrated_as"]) for par in pro_summary.tables["reintegrated"]])
                # print(pro_summary.soup.getText())
                # print(pro_summary.tables[0])
                # test_table = pro_summary.tables[0]
        #       
        # with open(root_filing.parent / "headers.csv", "w", newline="", encoding="utf-8") as f:
        #     # print(main)
        #     df = pd.DataFrame(main)
        #     df.to_csv(f)


    def init_DilutionDB():
        # connect 
        db = DilutionDB(cnf.DILUTION_DB_CONNECTION_STRING)
        db.util.inital_company_setup(db, cnf.DOWNLOADER_ROOT_PATH, cnf.POLYGON_OVERVIEW_FILES_PATH, cnf.POLYGON_API_KEY, cnf.APP_CONFIG.TRACKED_FORMS, ["CEI", "HYMC", "GOEV", "SKLZ", "ACTG", "INO", "GNUS"])

    # init_DilutionDB()
    def test_database(skip_bulk=True):
        db = DilutionDB(cnf.DILUTION_DB_CONNECTION_STRING)
        db.updater.dl.index_handler.check_index()
        db.util.reset_database()
        from datetime import datetime
        
        # print(db.read("SELECT * FROM company_last_update", []))
        if skip_bulk is True:
            with db.conn() as conn:
                db._update_files_lud(conn, "submissions_zip_lud", datetime.utcnow())
                db._update_files_lud(conn, "companyfacts_zip_lud", datetime.utcnow())
        db.util.inital_company_setup(
            cnf.DOWNLOADER_ROOT_PATH,
            cnf.POLYGON_OVERVIEW_FILES_PATH,
            cnf.POLYGON_API_KEY,
            ["DEF 14A", "S-1"],
            ["CEI"])
        with db.conn() as conn:
            db._update_company_lud(conn, 1, "filings_download_lud", datetime(year=2022, month=1, day=1))
        with db.conn() as conn:    
            db.updater.update_ticker("CEI")
    
    
    
    def test_spacy():
        from main.parser.filing_nlp import SpacyFilingTextSearch
        spacy_text_search = SpacyFilingTextSearch()
        # import spacy
        # from spacy.matcher import Matcher

        # nlp = spacy.load("en_core_web_sm")
        # nlp.remove_pipe("lemmatizer")
        # nlp.add_pipe("lemmatizer").initiali312qzu6ze()

        # matcher = Matcher(nlp.vocab)
        # pattern1 = [{"LEMMA": "base"},{"LEMMA": "onE:\\test\\sec_scraping\\resources\\datasets\\0001309082"},{"ENT_TYPE": "CARDINAL"},{"IS_PUNCT":False, "OP": "*"},{"LOWER": "shares"}, {"IS_PUNCT":False, "OP": "*"}, {"LOWER": {"IN": ["outstanding", "stockoutstanding"]}}, {"IS_PUNCT":False, "OP": "*"}, {"LOWER": {"IN": ["of", "on"]}}, {"ENT_TYPE": "DATE"}, {"ENT_TYPE": "DATE", "OP": "*"}]
        # pattern2 = [{"LEMMA": "base"},{"LEMMA": "on"},{"ENT_TYPE": "CARDINAL"},{"IS_PUNCT":False, "OP": "*"},{"LOWER": "outstanding"}, {"LOWER": "shares"}, {"IS_PUNCT":False, "OP": "*"},{"LOWER": {"IN": ["of", "on"]}}, {"ENT_TYPE": "DATE"}, {"ENT_TYPE": "DATE", "OP": "+"}]
        # matcher.add("Test", [pattern1])
        text = (" The number of shares and percent of class stated above are calculated based upon 399,794,291 total shares outstanding as of May 16, 2022")
        matches = spacy_text_search.match_outstanding_shares(text)
        print(matches)
        # doc = nlp(text)
        # for ent in doc.ents:
        #     print(ent.label_, ent.text)
        # for token in doc:
        #     print(token.ent_type_)
        
    def create_htm_filing():
        fake_filing_info = {
            # "path": r"C:\Users\Olivi\Desktop\test_set\set_s3/filings/0000002178/S-3/000000217820000138/a2020forms-3.htm",
            # "path": r"C:\Users\Olivi\Desktop\test_set\set_s3/filings/0001325879/S-3/000119312520289207/d201932ds3.htm",
            "path": r"F:/example_filing_set_S3/filings/0000002178/S-3/000000217820000138/a2020forms-3.htm",
            # "path": r"F:/example_filing_set_S3/filings/0001175680/S-3/000119312518056145/d531632ds3.htm",
            "filing_date": "2022-01-05",
            "accession_number": "000147793221000113",
            "cik": "0001477932",
            "file_number": "001-3259",
            "form_type": "S-3",
            "extension": ".htm"
        }
        from main.parser.parsers import filing_factory
        filing = filing_factory.create_filing(**fake_filing_info)
        return filing
    
    def _create_filing(form_type, path, extension=".htm"):
        fake_filing_info = {
            "path": path,
            # "path": r"F:/example_filing_set_S3/filings/0001175680/S-3/000119312518056145/d531632ds3.htm",
            "filing_date": "2022-01-05",
            "accession_number": "000147793221000113",
            "cik": "0001477932",
            "file_number": "001-3259",
            "form_type": form_type,
            "extension": extension
        }
        from main.parser.parsers import filing_factory
        filing = filing_factory.create_filing(**fake_filing_info)
        return filing
    # 
    # create_htm_filing()
    def test_s3_splitting_by_toc_hrefs():
        s3_path = r"C:\Users\Olivi\Testing\sec_scraping\tests\test_resources\filings\0001325879\S-3\000119312518218817\d439397ds3.htm"
        parser = HTMFilingParser()
        doc = parser.get_doc(s3_path)
        sections = parser._split_by_table_of_contents_based_on_hrefs(parser.make_soup(doc))
        print([s.title for s in sections])
        
    # test_s3_splitting_by_toc_hrefs()

    def test_spacy_secu_matches():
        from main.parser.filing_nlp import SpacyFilingTextSearch
        spacy_text_search = SpacyFilingTextSearch()
        # text = "1,690,695 shares of common stock issuable upon exercise of stock options outstanding as of September 30, 2020 at a weighted-average exercise price of $12.86 per share."
        # doc = spacy_text_search.nlp(text)
        filing = create_htm_filing()
        doc = spacy_text_search.nlp(filing.get_section(re.compile("summary", re.I)).text_only)
        # doc = spacy_text_search.nlp("2,500,000 shares of common stock issuable upon exercise at an exercise price of $12.50 per share;")
        # doc = spacy_text_search.nlp("1,690,695 shares of common stock issuable upon exercise of stock options outstanding as of September 30, 2020 at a weighted-average exercise price of $12.86 per share.")

        displacy.serve(doc, style="ent")
        # for ent in doc.ents:
        #     if ent.label_ == "SECU":
        #         print(ent.label_, ": " ,ent.text)
        
        # for t in doc:
        #     print(t)
    # test_spacy_secu_matches()

    def get_secu_list():
        from main.parser.filing_nlp import SpacyFilingTextSearch
        spacy_text_search = SpacyFilingTextSearch()
        root = r"F:\example_filing_set_100_companies"
        paths = [f for f in (Path(root) /"filings").rglob("*.htm")]
        parser = HTMFilingParser()
        secus = set()
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()
                soup = BeautifulSoup(raw, features="html5lib")
                text = parser.get_text_content(soup, exclude=["script"])
                if len(text) > 999999:
                    print("SKIPPING TOO LONG FILE")
                    continue
                doc = spacy_text_search.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "SECU":
                        secus.add(ent.text)
        secus_list = list(secus)
        pd.DataFrame(secus_list).to_clipboard()
    # get_secu_list()
    def get_relates_to_list():
        from main.parser.filing_nlp import SpacyFilingTextSearch
        from main.parser.parsers import ParserS3
        spacy_text_search = SpacyFilingTextSearch()
        root_d = Path(r"F:\example_filing_set_S3")
        root = root_d / "filings"
        paths = get_all_filings_path(root, "S-3")
        parser = ParserS3()
        relates = set()
        unmatched_files = set()
        try:
            for p in tqdm(paths[14:]):
                filing = _create_filing("S-3", p)
                if isinstance(filing, list):
                    for f in filing:
                        cp = f.get_section(re.compile("cover page", re.I))
                        if cp:
                            matches = spacy_text_search.match_prospectus_relates_to(cp.text_only)
                            if matches:
                                for m in matches:
                                    relates.add(m)
                else:
                    cp = filing.get_section(re.compile("cover page", re.I))
                    if cp:
                        matches = spacy_text_search.match_prospectus_relates_to(cp.text_only)
                        if matches:
                            for m in matches:
                                relates.add(m)
                        else:
                            unmatched_files.add(p)
        finally:
            relating_to = list(relates)
            unmatched = list(unmatched_files)
            pd.DataFrame(relating_to).to_csv(root_d / "relating_to.csv")
            pd.DataFrame(unmatched).to_csv(root_d / "failed_to_match_relating_to.csv")
    # get_relates_to_list()


    def test_parser_sc13d():
        parser = ParserSC13D()
    
    def create_sc13g_filing():
        fake_filing_info = {
            "path": r"F:/example_filing_set_sc13/filings/0001578621/SC 13G/000101143821000186/form_sc13g-belong.htm",
            "filing_date": "2022-01-05",
            "accession_number": "000149315220008831",
            "cik": "0001812148",
            "file_number": "001-3259",
            "form_type": "SC 13G",
            "extension": ".htm"
        }
        
        from main.parser.parsers import filing_factory
        filing: BaseHTMFiling = filing_factory.create_filing(**fake_filing_info)
        # b = filing.get_section("before items")
        # print([t["parsed_table"] for t in b.tables])
    # test_parser_sc13d()
    # create_sc13g_filing()

    def test_sc13d_main_table():
        filings = get_all_filings_path(r"F:\example_filing_set_sc13\filings", "SC 13D")
        from main.parser.parsers import filing_factory
        for path in filings:
            path1 = Path(path)
            info = {
            "path": path,
            "filing_date": None,
            "accession_number": path1.parents[0].name,
            "cik": path1.parents[2].name,
            "file_number": None,
            "form_type": "SC 13d",
            "extension": ".htm"
            }
            print(path)
            filing: BaseHTMFiling = filing_factory.create_filing(**info)

    # test_sc13d_main_table()

    def test_sc13g_main_table():
        filings = get_all_filings_path(r"F:\example_filing_set_sc13\filings", "SC 13G")
        from main.parser.parsers import filing_factory
        for path in filings:
            path1 = Path(path)
            info = {
            "path": path,
            "filing_date": None,
            "accession_number": path1.parents[0].name,
            "cik": path1.parents[2].name,
            "file_number": None,
            "form_type": "SC 13G",
            "extension": ".htm"
            }
            print(path)
            filing: BaseHTMFiling = filing_factory.create_filing(**info)
    
    def get_s3_resale_filings(root_path, max_num=100):
        unhandled = []
        filing_creation_failed = []
        other_exceptions = []
        # front_pages = []
        from main.parser.extractors import HTMS3Extractor
        extractor = HTMS3Extractor()
        paths = get_all_filings_path(root_path, "S-3")
        resale_paths = []
        if len(paths) <= max_num-1:
            pass
        else:
            paths = paths[:max_num]
        for path in tqdm(paths):
            try:
                filings = _create_filing("S-3", path)
            except Exception as e:
                filing_creation_failed.append(path)
                continue
            if not isinstance(filings, list):
                filings = [filings]
            for filing in filings:
                # try:
                #     front_page = filing.get_section(re.compile("front page"))
                #     front_pages.append(front_page.text_only)
                # except Exception:
                #     front_pages.append(None)
                try:
                    form_case = extractor.classify_s3(filing)
                    if "resale" in form_case["classifications"]:
                        resale_paths.append(path)
                except UnhandledClassificationError:
                    unhandled.append(path)
                except Exception as e:
                    other_exceptions.append(e)
                    print("opening file in browser")
                    webbrowser.open(path, new=2)
        print(f"unhandled classification for paths: {unhandled}")
        print(f"other_exceptions: {other_exceptions}")
        print(f"filing_creation_failed: {filing_creation_failed}")
        # print(f"front_pages: {front_pages}")
        pd.Series([str(p) for p in resale_paths]).to_clipboard()
        return resale_paths

    def get_s3_AMT_filings(root_path, max_num=100):
        unhandled = []
        filing_creation_failed = []
        other_exceptions = []
        # front_pages = []
        from main.parser.extractors import HTMS3Extractor
        extractor = HTMS3Extractor()
        paths = get_all_filings_path(root_path, "S-3")
        ATM_paths = []
        if len(paths) <= max_num-1:
            pass
        else:
            paths = paths[:max_num]
        for path in tqdm(paths):
            try:
                filings = _create_filing("S-3", path)
            except Exception as e:
                filing_creation_failed.append(path)
                continue
            if not isinstance(filings, list):
                filings = [filings]
            for filing in filings:
                # try:
                #     front_page = filing.get_section(re.compile("front page"))
                #     front_pages.append(front_page.text_only)
                # except Exception:
                #     front_pages.append(None)
                try:
                    form_case = extractor.classify_s3(filing)
                    if "ATM" in form_case["classifications"]:
                        ATM_paths.append(path)
                except UnhandledClassificationError:
                    unhandled.append(path)
                except Exception as e:
                    other_exceptions.append(e)
                    print("opening file in browser")
                    webbrowser.open(path, new=2)
        print(f"unhandled classification for paths: {unhandled}")
        print(f"other_exceptions: {other_exceptions}")
        print(f"filing_creation_failed: {filing_creation_failed}")
        # print(f"front_pages: {front_pages}")
        pd.Series([str(p) for p in ATM_paths]).to_clipboard()
        return ATM_paths
    # test_sc13g_main_table()

    # def test_sc13d_main_table_alt():
    #     t =  [['1 NAME OF REPORTING PERSON   Qatar Airways Investments (UK) Ltd    '], ['2 CHECK THE APPROPRIATE BOX IF A MEMBER OF A GROUP (SEE INSTRUCTIONS) (a) ☐  (b) ☒  '], ['3 SEC USE ONLY       '], ['4 SOURCE OF FUNDS (SEE INSTRUCTIONS)   WC    '], ['5 CHECK IF DISCLOSURE OF LEGAL PROCEEDINGS IS REQUIRED PURSUANT TO ITEM 2(D) OR ITEM 2(E)  ☐ N/A    '], ['6 CITIZENSHIP OR PLACE OF ORGANIZATION   United Kingdom    '], ['NUMBER OF SHARES BENEFICIALLY OWNED BY EACH REPORTING PERSON WITH 7 SOLE VOTING POWER   0    '], ['8 SHARED VOTING POWER   60,837,452    '], ['9 SOLE DISPOSITIVE POWER   0    '], ['10 SHARED DISPOSITIVE POWER   60,837,452    '], ['11 AGGREGATE AMOUNT BENEFICIALLY OWNED BY EACH REPORTING PERSON   60,837,452    '], ['12 CHECK IF THE AGGREGATE AMOUNT IN ROW (11) EXCLUDES CERTAIN SHARES (SEE INSTRUCTIONS)  ☐     '], ['13 PERCENT OF CLASS REPRESENTED BY AMOUNT IN ROW (11)   10% (1)    '], ['14 TYPE OF REPORTING PERSON (SEE INSTRUCTIONS)   CO    ']]
    #     from main.parser.parsers import MAIN_TABLE_ITEMS_SC13D, _re_get_key_value_table
    #     _, items = _re_get_key_value_table(t, MAIN_TABLE_ITEMS_SC13D, 0)
    #     print(items)
    
    # test_sc13d_main_table_alt()




    # t = [['2.', 'Check the Appropriate Box if a Member of a Group (See Instructions):'], ['', '(a) (b)'], ['3.', 'SEC Use Only'], ['4.', 'Source of Funds (See Instructions): OO'], ['5.', 'Check if Disclosure of Legal Proceedings Is Required Pursuant to Items 2(d) or 2(e): Not Applicable'], ['6.', 'Citizenship or Place of Organization: Ireland'], ['', '']]
    # term = re.compile("SEC(?:\s){0,2}Use(?:\s){0,2}Only(.*)", re.I | re.DOTALL)
    # for row in t:
    #     for field in row:
    #         match = re.search(term, field)
    #         if match:
    #             print(match.groups())
    # table = [['CUSIP No. G3728V 109', None], ['', None], ['1.', 'Names of Reporting Person. Dermot Smurfit']]
    # parser = ParserSC13D()
    # print(parser._is_main_table_start(table))


   

    
    # dl = Dä¨$ä¨$$$$$ings("0001175680", "S-3", after_date="2000-01-01", number_of_filings=100)
    # dl.get_filings("CEI", "DEF 14A", after_date="2021-01-01", number_of_filings=10)
    # dl.index_handler.check_index()        
    
    # test_database()

    
    # print(Path(url))
    # db = DilutionDB(cnf.DILUTION_DB_CONNECTION_STRING)
    # db.updater.update_ticker("CEI")
    # test_spacy()

    # db._update_files_lud("submissions_zip_lud", (datetime.utcnow()-timedelta(days=2)).date())
    # print(db.read("SELECT * FROM files_last_update", []))
    # dbu.update_bulk_files()            
    # try_htmlparser()

    # download_samples(r"F:\example_filing_set_100_companies")


    # from main.data_aggregation.bulk_files import update_bulk_files
    # update_bulk_files()
    # from spacy.tokens import DocBin, Doc
    # Doc.set_extension("rel", default={}, force=True)

    # docbin = DocBin(store_user_data=True)
    # docbin.from_disk(r"C:\Users\Olivi\Desktop\spacy_example2.spacy")
    # doc = list(docbin.get_docs(nlp.vocab))[0]



        # with open(p, "r", encoding="utf-8") as f:
        #     filing = parser.preprocess_filing(f.read())
        #     # print(filing)
        #     print(len(parser.parse_items(filing)))

    # open_filings_in_browser(r"F:\example_filing_set_S3\filings", "S-3", max=30)

    # text = " prospectus provides, describes general description or terms of securities. Each time we sell or offer securities or  securities are offered or sold we will provide you with prospectus supplement | supplement to this prospectus | supplement."
    # text = "The shares of common stock being offered include: 1)       6,406,000 shares of common stock issuable upon conversion of Series C Convertible Preferred Stock to shares issued to certain selling stockholders of certain private transactions occurring on certain dates between October 21, 2019 and December 6, 2019 (the “Series C Offering”); 2) 8,007,500 shares of common stock issuable upon exercise, at an exercise price of $0.30 per share, of warrants issued to certain selling stockholders in connection with the Series C Offering; 3)       1,620,000 shares of common stock issued to certain selling stockholders in connection with financial advisory fees arising from a transaction in November 2018 (the “November 2018 Transaction”); 4)      16,904,000 shares of common stock issuable upon conversion of Series D Convertible Preferred Stock to shares issued to certain selling stockholders of certain private transactions occurring on January 31, 2020 and March 13, 2020 (the “Series D Offering”); 5)      4,060,625 shares of common stock issuable upon exercise, at an exercise price of $1.00 per share, of warrants issued to certain selling stockholders in connection with the Series D Offering; 6) 3,800,000 shares of common stock issuable upon conversion, at an exercise price of $4.50 per share, of a long-term convertible note issued by a certain selling stockholder in connection with a long-term convertible note transaction on March 31, 2020 (the “Long-term Convertible Note Transaction”); 7)      2,500,000 shares of common stock issued in connection with the exercise, at an exercise price of $0.30, of warrants by a certain selling shareholder on February 4, 2020 and February 12, 2020, and the Long-term Convertible Note Transaction; 8)      2,500,000 shares of common stock issuable upon exercise, at exercise prices ranging from $0.57 per share to $0.83 per share, of warrants issued on March 6, 2015, February 15, 2018, November 8, 2018, and December 19, 2019 in connection with services provided by the Company’s Chief Executive Officer and certain consultants to the Company (the “Compensatory Warrant Grants”); 9) 600,000 shares of common stock issuable upon exercise, at exercise prices ranging from $0.39 per share to $0.63 per share, of stock options issued to certain selling stockholders on September 12, 2019, October 7, 2019, and December 19, 2019 in connection with services provided by consultants to the Company (the “Consultant Stock Option Grants”)."
    # text = "This prospectus relates to the sale from time to time by the selling stockholders identified in this prospectus for their own account of up to a total of 12,558,795 shares of our common stock, including up to an aggregate of 3,588,221 shares of our common stock issuable upon the exercise of warrants."
    # from main.parser.parsers import ParserS3
    # p = ParserS3()
    # text = p.preprocess_section_text_content(text)

    def create_absolute_path(rel, root):
        return str(Path(root).joinpath(rel))

    # download_samples(r"C:\Users\Olivi\Desktop\test_set\set2_s3", forms=["S-3"], num_tickers=300, max_filings=30)

    from main.parser.filing_nlp import SpacyFilingTextSearch
    from main.parser.extractors import BaseHTMExtractor
    search = SpacyFilingTextSearch()
    # filing = _create_filing("S-3", r"C:\Users\Olivi\Desktop\test_set\set_s3/filings/0001175680/S-3/000119312520128998/d921147ds3a.htm")
    # # text = 'The selling shareholders named in this prospectus may use this prospectus to offer and resell from time to time up to 22,093,822 shares of our common stock, par value $0.0001 per share, which are comprised of (i) 6,772,000 shares (the “Shares”) of our common stock issued in a private placement on November 22, 2021 (the “Private Placement”), pursuant to that certain Securities Purchase Agreement by and among us and certain investors (the “Purchasers”), dated as of November 17, 2021 (the “Securities Purchase Agreement”), (ii) 4,058,305 shares (the “Pre-funded Warrant Shares”) of our common stock issuable upon the exercise of the pre-funded warrants (the “Pre-funded Warrants”) issued in the Private Placement pursuant to the Securities Purchase Agreement, (iii) 10,830,305 shares (the “Common Stock Warrant Shares” and together with the Pre-funded Warrant Shares, the “Warrant Shares”) of our common stock issuable upon the exercise of the warrants (the “Common Stock Warrants” and together with the Pre-funded Warrants, the “Warrants”) issued in the Private Placement pursuant to the Securities Purchase Agreement we issued to such investor and (iv) 433,212 shares (the “Placement Agent Warrant Shares”) of our common stock issuable upon the exercise of the placement agent warrants (the “Placement Agent Warrants”) issued in connection with the Private Placement.'

    # texts = ["This prospectus relates to the offer and sale by the selling stockholders identified in this prospectus of up to 79,752,367 shares of our common stock, par value $0.001 per share, issued and outstanding or issuable upon exercise of warrants. The shares of common stock being offered include: 1)	35,286,904 shares issued to the selling stockholders in certain private transactions occurring between November 2, 2017 and February 16, 2018 (the “February 2018 Placement”); 2)	35,286,904 shares issuable upon exercise, at an exercise price of $0.75 per share, of warrants issued to the selling stockholders in the February 2018 Placement; 3)	2,813,490 shares issuable upon exercise, at an exercise price of $0.55 per share, of warrants issued to our placement agent and its employees in the February 2018 Placement;", "This prospectus relates to the sale from time to time by the selling stockholders identified in this prospectus for their own account of up to a total of 12,558,795 shares of our common stock, including up to an aggregate of 3,588,221 shares of our common stock issuable upon the exercise of warrants. The selling stockholders acquired their shares in a private placement of shares of common stock and warrants to purchase shares of common stock completed on August 29, 2008."]
    # root = r"F:\example_filing_set_S3\filings"
    root =  r"C:\Users\Olivi\Desktop\test_set\set2_s3"
    filing_root = str(Path(root)/"filings")
    # resale_paths = get_s3_resale_filings(filing_root, max_num=200)
    # import pandas as pd
    
    resale_paths = [
        r"0001089907\S-3\000155278120000140\e20085_swkh-s3.htm",
        r"0001109189\S-3\000119312517123212\d367699ds3.htm",
        r"0001138978\S-3\000149315221013807\forms3.htm",
        r"0001138978\S-3\000149315222010232\forms-3.htm",
        r"0001178727\S-3\000121390021050060\ea147604-s3_comsovereign.htm",
        r"0001421517\S-3\000143774919018317\erii20190823_s3.htm",
        r"0001453593\S-3\000149315221008120\forms-3.htm",
        r"0001453593\S-3\000149315221010942\forms-3.htm",
        r"0001595527\S-3\000110465920111083\tm2032148-2_s3.htm",
        r"0001830033\S-3\000110465922053533\tm2213671d1_s3.htm",
        r"0001830033\S-3\000110465922053550\tm2213671d2_s3.htm"
        ]
    
    extractor = BaseHTMExtractor()
    docs = []
    def get_absolute_paths_from_rel_paths(rel_paths, root_path):
        return [create_absolute_path(f, root_path) for f in rel_paths]

    def get_cover_pages_from_paths(paths, skip_first=True):
        for p in paths:
            filings = _create_filing("S-3", p)
            if not isinstance(filings, list):
                filings = [filings]
            for idx, filing in enumerate(filings):
                if skip_first is True and len(filings) > 1:
                    if idx == 0:
                        continue
                docs.append(extractor.doc_from_section(filing.get_section(re.compile("cover page"))))
        return docs
    
    # test_filing = _create_filing("S-3", r"F:/example_filing_set_S3/filings/0000002178/S-3/000000217820000138/a2020forms-3.htm")
    from main.parser.extractors import HTMS3Extractor
    extractor = HTMS3Extractor()
    # for filing in test_filing:
    #     print(extractor.classify_s3(filing))

    # atm_filings = get_s3_AMT_filings(filing_root, max_num=300)

    failed_creation = [
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0000092230/S-3/000119312520186933/d922227ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001176495/S-3/000007632117000036/forms-36302017.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000095012322003104/d238665ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000095012322003106/d224632ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312517033056/d340149ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312517170650/d384504ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312517297534/d460791ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312519069067/d702010ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312520063305/d854165ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312520132535/d917962ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312520139237/d923407ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312521041857/d222360ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312521045244/d32186ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312521274106/d223710ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001415311/S-3/000119312521274112/d185409ds3.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001466538/S-3/000146653817000119/cowenincs-3november2017xv8.htm', 
        r'C:/Users/Olivi/Desktop/test_set/set2_s3/filings/0001495932/S-3/000149593219000018/expi-20190131xs3.htm'
    ]

    ATM_paths = [
        r"0001571498\S-3\000119312521152469\d385293ds3.htm",
        r"0001623526\S-3\000119312520191098\d13608ds3.htm",
        r"0001623526\S-3\000119312522156134\d350430ds3.htm",
        r"0001708599\S-3\000149315221000635\forms-3.htm"
        ]
    # docs = get_cover_pages_from_paths(
    #     get_absolute_paths_from_rel_paths(ATM_paths, filing_root)
    #     )
    # for doc in docs:
    #     print(extractor.extract_aggregate_offering_amount(doc))
    
    # docs = get_cover_pages_from_paths(
    #     get_absolute_paths_from_rel_paths(resale_paths, filing_root)
    #     )

    # for doc in docs:
    #     print("ALIAS MAP:")docs = get_cover_pages_from_paths(atm_filings)
    #     print(doc._.single_secu_alias)
    #     print("SECU SPANS:")
    #     print(doc.spans["SECU"])

    # displacy.serve(docs, style="ent", options={
    #     "ents": ["SECU", "SECUREF", "SECUQUANTITY"],
    #     "colors": {"SECU": "#e171f0", "SECUREF": "#03fcb1", "SECUQUANTITY": "grey"}
    #     })


    
    
    # # text = "up to $ 75,000,000 of Common Stock  issued with exercise price of 1$."
    # text = "from time to time, of up to 119,007,618 shares of our common stock"
    
    texts = [
        ## EXERCISE PRICE SENTENCES
        # "An aggregate of 8,555,804 shares of common stock issuable upon the exercise of stock purchase warrants outstanding as of September 20, 2021 with a weighted average exercise price of $4.20 per share that expire between November 9, 2021 and January 10, 2026"
        # "The Warrants to purchase 33,334 shares of common stock at any time on or prior to September 26, 2022 at an initial exercise price of $3.00 per share.",
        # "The Warrants have an exercise price of $11.50 per share.",
        # "The Warrant has an exercise price of $2.25 per share.",
        # "Warrants to purchase 10,000 shares of common stock at any time on or prior to December 15, 2021 at an initial exercise price of $1.50 per share.",
        # "Warrants to purchase 96,668 shares of common stock and remain outstanding at any time on or prior to December 31, 2022 at an initial exercise price of $3.00 per share.",
        # "Warrants to purchase 560,192 shares of common stock at any time on or prior to July 6, 2025 at an initial exercise price of $0.15 per share for 315,689 of the warrants and $0.72 per share for 244,503 of the warrants."
        # "The Placement Agent Warrant have an exercise price of $2.8125 per share, subject to customary anti-dilution, but not price protection, adjustments.",
        # "We will be prohibited from effecting an exercise of the Investor Warrant to the extent that, as a result of such exercise, the Investor would beneficially own more than 9.99% of the number of shares of our common stock outstanding immediately after giving effect to the Warrant Shares."
        "Warrants to purchase 33,334 shares of common stock at any time on or prior to September 26, 2022 at an initial exercise price of $3.00 per share.",
        
        ## EXPIRY SENTENCES
        # "The Warrants are immediately exercisable and expire on the five-year anniversary of the date of issuance.",
        # "The Warrants expire on the five-year anniversary of the date of issuance.",
        # "The Warrants expires on August 6, 2025.",
        # "The Warrants are exercisable at an exercise price of $2.00 per share and expire on the fourth year anniversary of December 14, 2021, the initial issuance date of the Warrants",
        # "The Warrants have an exercise price of $11.50 per share will be exercisable beginning on the calendar day following the six month anniversary of the date of issuance, will expire on March 17, 2026.",
        # "Holders of our Series A Warrants may exercise their Series A Warrants at any time beginning on the first anniversary of the date of issuance of such shares up to 5:00 p.m., New York time, on the date that is the fifth anniversary of such date of issuance (the “Series A Warrant Expiration Date”).",
        
        # "The option fully vested on the date of grant and expires on August 6, 2025.",
        # "The Investor Warrant is immediately exercisable and will expire on the five-year anniversary of the date of issuance",
        # "Each Company Warrant become exercisable on May 4, 2021 and will expire five years after the completion of the Business Combination, or earlier upon redemption.",
        
        ## SECUQUANTITY SENTENCES
        "This prospectus relates to the resale, from time to time, of up to an aggregate of 16,000,002 shares of common stock, par value $0.000001 per share, of Xtant Medical Holdings, Inc. by the selling stockholders named in this prospectus, including their respective donees, pledgees, transferees, assignees or other successors-in-interest. The selling stockholders acquired these shares from us pursuant to a (i) Securities Purchase Agreement, dated February 22, 2021 pursuant to which we issued 8,888,890 shares of common stock, par value $0.000001 per share, at a purchase price of $2.25 per share, and a warrant to purchase up to 6,666,668 shares of common stock in a private placement, and (ii) Placement Agent Agreement, dated February 22, 2021, with A.G.P./Alliance Global Partners pursuant to which we issued warrants to purchase up to an aggregate of 444,444 shares of common stock.",

 
        ]
    secu_idxs = [
        # (1, 2),
        # (1, 2),
        # (1, 2),
        # (1, 2),
        # (1, 2)
        # (3, 6),
        (0, 1),
    ]
    docs = []
    ex = []
    # for text, idxs in zip(texts, secu_idxs):
    #     doc = search.nlp(text)
    #     print(doc)
    # #     print(ex.append(extractor.spacy_text_search.match_secu_expiry(doc, doc[idxs[0]:idxs[1]])))
    # #     for token in doc:
    # #         print(token.lemma_, token.ent_type_)
    #     ex.append(extractor.spacy_text_search.match_secu_exercise_price(doc, doc[idxs[0]:idxs[1]]))
    # #     # ex.append(extractor.spacy_text_search.match_secu_with_dollar_CD(doc, doc[0:1]))
    #     docs.append(doc)
    # print(ex)
    def get_unique_secu_text_from_alias(secu_map):
        unique_secus = []
        unique_secus_text = set()
        for key, value in secu_map.items():
            for each in ["alias", "no_alias"]:
                for secu_span in value[each]:
                    secu_text = secu_span.text
                    if secu_text not in unique_secus_text:
                        unique_secus_text.add(secu_text)
                        unique_secus.append(secu_span)
        return unique_secus

    def test_get_secuquantities(search):    
        for text in texts:
            doc = search.nlp(text)
            secu_map = doc._.single_secu_alias_tuples
            unique_secus = get_unique_secu_text_from_alias(secu_map)
            print(f"unique secus text: {unique_secus}")
            for secu in unique_secus:
                print(f"looking for quants for secu: {secu}")
                quants = search.get_secuquantities(doc, secu)
            docs.append(doc)
    texts = [
        # RESALE SECURITY REGISTRATION SENTENCES (ONLY COVER PAGE)
        # "The selling stockholders may offer and sell from time to time up to 9,193,766 shares of our common stock. Certain of such shares of common stock are issuable upon exercise of a warrant to purchase common stock issued to one of the selling stockholders.",
        # "This prospectus relates to an aggregate of up to 9,497,051 shares of common stock of Basic Energy Services, Inc. (“Basic”) that may be resold from time to time by the selling stockholders named on page 5 of this prospectus for their own account.",
        # # "This prospectus covers the sale of an aggregate of 2,388,050 shares (the “shares”) of our common stock , $0.001 par value per share (the “ common stock ”), by the selling stockholders identified in this prospectus (collectively with any of the holder’s transferees, pledgees, donees or successors, the “selling stockholders”). The shares are issuable upon the exercise of warrants (the “ warrants ”) purchased by the selling stockholders in a private placement transaction exempt from registration under Section 4(a)(2) of the Securities Act of 1933, as amended (the “Securities Act”), pursuant to a Securities Purchase Agreement dated April 9, 2021 (the “Purchase Agreement”).",
        # "his prospectus covers the sale of an aggregate of 223,880 shares (the “shares”) of our common stock , $0.001 par value per share (the “ common stock ”), by the selling stockholders identified in this prospectus (collectively with any of the holder’s transferees, pledgees, donees or successors, the “selling stockholders”). The shares are issuable upon exercise of warrants (the “ warrants ”) purchased by the selling stockholders in private placement transactions exempt from registration under Section 4(a)(2) of the Securities Act of 1933, as amended (the “Securities Act”), pursuant to Securities Purchase Agreements, dated November 17, 2021 (the “Purchase Agreements”), with the selling stockholders.",
        "This prospectus relates to the resale, from time to time, by the selling stockholders named herein (the “Selling Stockholders”) of (i) an aggregate of 5,600,001 shares of our common stock , par value $0.0001 per share, issuable upon the conversion of certain outstanding convertible promissory notes and (ii) an aggregate of 3,135,789 shares of common stock issuable upon exercise of certain outstanding warrants (the “Warrants”).",
        # "Further, the selling stockholders identified in this prospectus (the “selling stockholders”) may offer and sell from time to time, in one or more offerings, up to 16,022,824 shares of our common stock as described in this prospectus.",
        # "This prospectus relates to the resale, from time to time, of up to an aggregate of 16,000,002 shares of common stock , par value $0.000001 per share, of Xtant Medical Holdings, Inc. by the selling stockholders named in this prospectus, including their respective donees, pledgees, transferees, assignees or other successors-in-interest.",
        # "In addition, this prospectus relates to the resale, from time to time, of up to an aggregate of 18,218,374 shares of our common stock by the selling stockholders named in this prospectus, including their respective donees, pledgees, transferees, assignees or other successors-in-interest. ",
        ## "This prospectus relates to the possible resale, from time to time, by the selling stockholders named in this prospectus, of up to 52,435 shares of our Class A common stock currently outstanding and up to 13,100 shares of Class A common stock issuable upon redemption of units of limited partnership designated as “Class A Units” (“Class A units”) in New York City Operating Partnership, L.P., a Delaware limited partnership that is our operating partnership and of which we are the sole general partner (the “OP”). The shares of our Class A common stock covered by this prospectus include (i) 52,398 shares of Class A common stock that were issued to the Advisor upon the conversion of an equal number of limited partnership interests designated as “Class B Units” (“Class B units”) in the OP into an equal number of Class A units, and the redemption of those Class A units for an equal number of shares of 	our Class A common stock when trading of Class A common stock on The New York Stock Exchange (“NYSE”) commenced on August 18, 2020, (ii) 37 shares that were issued by us to the Advisor upon the redemption of an equal number of Class A units held by the Advisor when trading of Class A common stock on the NYSE commenced on August 18, 2020, and (iii) 13,100 shares of Class A common stock issuable upon redemption of Class A units that were issued to a former holder of equity interests in the entities that own and control the Advisor upon the conversion of an equal number of Class B units in accordance with their terms into Class A units when trading of Class A common stock on the NYSE commenced on August 18, 2020.",
        # "This prospectus relates to the offer and sale from time to time, on a resale basis, by the selling stockholders identified herein (the “Selling Stockholders”) or their permitted transferees, of up to an aggregate of 33,132,056 shares of our common stock , par value $0.001 per share (“ Common Stock ”), consisting of: (i) 31,438,253 shares of Common Stock issued in connection with the Business Combination (as defined herein) that are currently issued and outstanding and (ii) 1,693,803 shares of Common Stock issuable in connection with the Earnout (as defined herein).",
        # "This prospectus relates to the offer and sale from time to time, on a resale basis, by the selling stockholders identified herein (the “Selling Stockholders”) or their permitted transferees, of up to an aggregate of 53,571,408 shares of our common stock , par value $0.001 per share (“ Common Stock ”), consisting of: (i) 35,714,272 shares of Common Stock that are issued and outstanding (the “Private Placement Common Stock”) and (ii) 17,857,136 shares of Common Stock issuable upon exercise of Series A warrants held by the Selling Stockholders (the “ Series A Warrants ”).",
        
        # RESALE SECURITY REGISTRATION SENTENCES (ALIAS PARANTHESES REMOVED)
        # "This prospectus covers the sale of an aggregate of 223,880 shares of our common stock, $0.001 par value per share, by the selling stockholders identified in this prospectus. The shares are issuable upon exercise of warrants purchased by the selling stockholders in private placement transactions exempt from registration under Section 4(a)(2) of the Securities Act of 1933, as amended, pursuant to Securities Purchase Agreements, dated November 17, 2021, with the selling stockholders.",
        # "This prospectus relates to the resale, from time to time, by the selling stockholders named herein of an aggregate of 5,600,001 shares of our common stock , par value $0.0001 per share, issuable upon the conversion of certain outstanding convertible promissory notes.",
        # "This prospectus relates to the resale, from time to time, by the selling stockholders named herein of an aggregate of 3,135,789 shares of common stock issuable upon exercise of certain outstanding warrants.",
        # "Further, the selling stockholders identified in this prospectus may offer and sell from time to time, in one or more offerings, up to 16,022,824 shares of our common stock as described in this prospectus.",
        # "This prospectus relates to the resale, from time to time, of up to an aggregate of 16,000,002 shares of common stock , par value $0.000001 per share, of Xtant Medical Holdings, Inc. by the selling stockholders named in this prospectus, including their respective donees, pledgees, transferees, assignees or other successors-in-interest.",
        #  "In addition, this prospectus relates to the resale, from time to time, of up to an aggregate of 18,218,374 shares of our common stock by the selling stockholders named in this prospectus, including their respective donees, pledgees, transferees, assignees or other successors-in-interest. ",
        
        # TEST PHRASES FOR ATTRIBUTE OF SECURITY
        # "The Warrants are outstanding.",
        # "The Shares, currently outstanding, represent a big amount of the total shares.",
        # "The Common Stock is issuable upon redemption of warrants.",
        # "The Shares of our Common Stock, par value $ 0.001 per share, is the only security issued."
    ]
    # secq = []
    # for text in texts:
    #     doc = search.nlp(text)
    #     for secu in doc.spans["SECU"]:
    #         root_token = search._get_compound_SECU_root(secu)
    #         if root_token:
    #             secq.append((secu, search.get_SECU_subtree_adjectives(root_token)))
                # secq.append((secu, search.get_head_verbs(root_token)))
        #     print(secu.root)
        # for x in (set([token.pos_ for token in doc])):
        #     print(x, spacy.explain(x))
        # for x in (set([token.tag_ for token in doc])):
        #     print(x, spacy.explain(x))
        # for x in (set([token.dep_ for token in doc])):
        #     print(x, spacy.explain(x))
        # for token in doc:
        #     print(token.pos_, token.tag_)

        # docs.append(doc)
    
        # print("prep_phrases: ")
        # [print("    ", x) for x  in extractor.spacy_text_search.get_prep_phrases(doc)]
        # print("verbal_phrases: ")
        # [print("    ", x) for x in extractor.spacy_text_search.get_verbal_phrases(doc)]
        # print("noun_chunks: ")
        # [print("    ", chunk, chunk.root.text, chunk.root.dep_) for chunk in doc.noun_chunks]
    # print("secus_with_dollar_CD: ", ex)
        # # deps = set([token.dep_ for token in doc])
        # # tag = set([token.tag_ for token in doc])
        # # # print([sent.root for sent in doc.sents])
        # # secu = None
        # for sent in doc.sents:
        #     for chunk in sent.noun_chunks:
        #         if chunk.root.ent_type_ == "SECU":
        #             secu = chunk.root
        #             print("->".join([i.text for i in secu.subtree]))
        #             print("head: ", secu.head.text)

    # print(secq)

            # for token in sent:
            #     if token.text == "price":
            #         t = token
            #         while (len([i for i in t.ancestors]))
            #         print("ancestors: ", [i for i in token.ancestors])
                    # print("sent.root: ", sent.root)
                    # print("token.head.children: ", [i for i in token.head.children])
                    # if sent.root.is_ancestor(token):
                    #     print("is ancestor of root: ", token)
            #     for t in token.head.children:
            #         # print(t.ent_type_, t.dep_)
            #         if t.dep_ == "nsubj" and t.ent_type_ == "SECU":

            #             print("found: ", [i if i.dep_ == "compound" else  for i in t.lefts], [i for i in t.rights], t.i) 
        # print("root: ", doc.root)
        # docs.append(doc)
    

    # output_path = Path(r"C:\Users\Olivi\Desktop\test_svg.svg") # you can keep there only "dependency_plot.svg" if you want to save it in the same folder where you run the script 
    # svg = displacy.render(docs, style="dep", options={"fine_grained": True, "compact": True})
    # output_path.open("w", encoding="utf-8").write(svg)
    def displacy_dep_with_search(text, print_tokens=False, show_lemmas=False, finegrained=False):
        search = SpacyFilingTextSearch()
        doc = search.nlp(text)
        if print_tokens is True:
            for token in doc:
                print(token)
                print(      spacy.explain(token.dep_),  token.dep_)
                print(      spacy.explain(token.pos_), token.pos_)
                print(      spacy.explain(token.tag_), token.tag_)
        displacy.serve(doc, style="dep", options={"fine_grained": finegrained, "compact": True, "add_lemma": show_lemmas, "word_spacing": 16, "arrow_spacing": 12}, port=5000)

    def displacy_dep_with_search_multi(texts: list[str], show_lemmas=False, finegrained=False):
        search = SpacyFilingTextSearch()
        docs = []
        for text in texts:
            docs.append(search.nlp(text))
        displacy.serve(docs, style="dep", options={"fine_grained": finegrained, "compact": True, "add_lemma": show_lemmas, "word_spacing": 16, "arrow_spacing": 12}, port=5000)

    def displacy_ent_with_search(text):
        search = SpacyFilingTextSearch()
        doc = search.nlp(text)
        print("ENTS:")
        print([(i, i.label_) for i in doc.ents])
        print("TOKENS:")
        print([i for i in doc])
        displacy.serve(doc, style="ent", options={
            "ents": ["SECU", "SECUQUANTITY", "CONTRACT", "DATE", "ORG"],
            "colors": {"SECU": "#e171f0", "SECUQUANTITY": "#03fcb1", "CONTRACT": "green", "DATE": "yellow"}
            },
            port=5000
        )
    
    #     print(doc._.single_secu_alias)

        # for token in doc:
        #     print(token.lower_, token.ent_type_)
        # extractor = BaseHTMExtractor()
        # extractor.get_issuable_relation(doc, "")
        # for secu, values in doc._.single_secu_alias.items():
        #     print(secu)
        #     print(values)
    # from spacy.matcher import Matcher
    # m = Matcher(search.nlp.vocab)
    # regular_patterns = [
    #         [
    #             {"ENT_TYPE": "CARDINAL"},
    #             {"LOWER": {"IN": ["authorized", "outstanding"]}, "OP": "?"},
    #             {"LOWER": {"IN": ["share", "shares", "warrant shares"]}}
    #         ],
    #         [
    #             {"ENT_TYPE": "CARDINAL"},
    #             # {"ENT_TYPE": "MONEY", "OP": "*"},
    #             {"ENT_TYPE": "SECU"}
    #             # {"ENT_TYPE": "SECU", "OP": "*"},
    #         ]
    #     ]
    # m.add("1", [*regular_patterns])
    # matches = m(doc, as_spans=True)
    # print(matches)
    # print([token.text for token in doc])
    # print([(ent.text, ent.label_) for ent in doc.ents])
    # for ent in doc.ents:
    #     if ent.label_ == "SECUQUANTITY":
    #         print(ent._.secuquantity)



    # print([(t.text, t.ent_type_) for t in doc])
    # test = search.get_secus_and_secuquantity(doc)
    # for entry in test:
    #     if "security" in entry.keys():
    #         secu = entry["security"]
    #         is_alias = doc._.is_alias(secu)
    #         if not is_alias:
    #             print(doc._.get_alias(secu))
    # print(doc.spans)
    # print(doc._.single_secu_alias)
    
    
    # print(doc.spans)
    # section = filing.get_section("cover page 0")
    # print(search.get_mentioned_secus(search.nlp(filing.get_text_only())))
    # displacy.serve(doc.sents, style="dep", options={"fine_grained": False, "compact": True})

    # displacy.serve(doc.sents, style="ent", options={
    #     "ents": ["SECU", "SECUQUANTITY"],
    #     "colors": {"SECU": "#e171f0", "SECUQUANTITY": "#03fcb1"}
    # })

    # displacy.serve(doc, style="span", options={"spans_key":"SECU"})
    # print(search.get_mentioned_secus(search.nlp(text)))
    # text = section.text_only
    # # print(text)
    # # # doc = search.nlp(text)
    # # # for token in doc:
    # # #     print(token,"\t" , token.ent_type_, "\t")
    # matches = search.match_secu_relation(text)
    # print(f"matches: {matches}")



    # doc = search.nlp(text)
    # for token in doc:
    #     print(token.text, token._.sec_act)
    # from main.parser.extractors import HTMS3Extractor
    # extractor = HTMS3Extractor()
    # filing = create_htm_filing()
    # for f in filing:
    #     cover_page = f.get_section(re.compile("cover page", re.I))
    #     text = cover_page.text_only
    #     doc = search.nlp(text)
    #     extractor._is_base_prospectus(doc)

    # for section in filing.sections:
    #     print(section.title, len(section.content))
    # cover_pages = filing.get_sections(re.compile("cover page", re.I))
    # for cv in cover_pages:
        # print(cv.title, cv.text_only)


    # f = _create_filing("S-3", r"F:/example_filing_set_S3/filings/0001514281/S-3/000151428121000068/mittforms-3may2021.htm")
    from boot import bootstrap_dilution_db
    from main.configs import FactoryConfig, GlobalConfig
    cnf = FactoryConfig(GlobalConfig(ENV_STATE="dev").ENV_STATE)()
    def do_inital_pop(cnf):   
        db = bootstrap_dilution_db(start_orm=True, config=cnf, uow=None)
        db.inital_setup(reset_database=True)
        with db.conn() as c:
            result = c.execute("SELECT * FROM companies").fetchall()
            print(result)
    
    def boot_db(cnf):
        return bootstrap_dilution_db(start_orm=True, config=cnf, uow=None)
    
    def reparse(db, form_type):
        # db.updater.dl.index_handler.check_index()
        # db.updater.dl.index_handler.check_index()
        db.util.reparse_local_filings("CEI", form_type)
    
    def unique_filings(cnf):
        db = bootstrap_dilution_db(start_orm=True, config=cnf, uow=None)
        all_filings = db.updater.dl.index_handler.get_local_filings_by_cik("0001309082")
        fset = set([tuple(i.values()) for i in all_filings])
        for i in fset:
            if i[0] == "EFFECT":
                print(i)
    
    def readd_filing_links(db):
        id = 1
        submissions_file = db.util._get_submissions_file("0001309082")
        submissions = db.util.format_submissions_json_for_db(
            "0001309082",
            submissions_file)
        for s in submissions:
            try:
                db.create_filing_link(
                    id,
                    s["filing_html"],
                    s["accessionNumber"],
                    s["form"],
                    s["filingDate"],
                    s["primaryDocDescription"],
                    s["fileNumber"])
            except Exception as e:
                print((e, s))
    
    # reparse(cnf, "S-3")
    # db = boot_db(cnf)
    # readd_filing_links(db)
    # reparse(db, "EFFECT")
    # unique_filings(cnf)
    # 
    # do_inital_pop(cnf)
    # displacy_dep_with_search('The Warrants have an exercise price of $11.50 per share will be exercisable beginning on the calendar day following the six month anniversary of the date of issuance, will expire on March 17, 2026.')
    # displacy_ent_with_search("The Series A Warrants have an exercise price of $11.50 per share.")
    def get_span_to_span_similarity_map(secu: list, alias: list, threshold: float = 0.65):
        similarity_map = {}
        for secu_token in secu:
            for alias_token in alias:
                similarity = secu_token.similarity(alias_token)
                similarity_map[(secu_token, alias_token)] = similarity
        return similarity_map

    text = "On February 22, 2021, we entered into the Securities Purchase Agreement (the “Securities Purchase Agreement”), pursuant to which we agreed to issue the investor named therein (the “Investor”) 8,888,890 shares (the “Shares”) of our common stock, par value $0.000001 per share, at a purchase price of $2.25 per share, and a warrant to purchase up to 6,666,668 shares of our common stock (the “Investor Warrant”) in a private placement (the “Private Placement”). The closing of the Private Placement occurred on February 24, 2021."
    def compare_similarity(text):
        def BFS_non_recursive(origin, target):
            path = []
            queue = [[origin]]
            node = origin
            visited = set()
            while queue:
                path = queue.pop(0)
                node = path[-1]
                if node == target:
                    return path
                visited.add(node)
                adjacents = (list(node.children) if node.children else []) + (list(node.ancestors) if node.ancestors else [])
                for adjacent in adjacents:
                    if adjacent in visited:
                        continue
                    new_path = list(path)
                    new_path.append(adjacent)
                    queue.append(new_path)
                    
        
        def get_dep_distance_between(secu, alias):
            is_in_same_tree = False
            if secu.is_ancestor(alias):
                is_in_same_tree = True
                start = secu
                end = alias
            if alias.is_ancestor(secu):
                is_in_same_tree = True
                start = alias
                end = secu
            if is_in_same_tree:
                path = BFS_non_recursive(start, end)
                return len(path)
            else:
                return None
        
        def get_dep_distance_between_spans(secu, alias):
            distance = get_dep_distance_between(secu.root, alias.root)
            return distance

        def get_span_distance(secu, alias):
            if secu[0].i > alias[0].i:
                first = alias
                second = secu
            else:
                first = secu
                second = alias
            mean_distance = ((second[0].i - first[-1].i) + (second[-1].i - first[0].i))/2
            return mean_distance


        search = SpacyFilingTextSearch()
        doc = search.nlp(text)
        from collections import defaultdict
        from main.parser.filing_nlp import get_premerge_tokens_for_span
        possible_aliases = defaultdict(list)
        for secu in doc._.secus:
            secu_first_token = secu[0]
            secu_last_token = secu[-1]
            for sent in doc.sents:
                if secu_first_token in sent:
                    secu_counter = 0
                    token_idx_offset = sent[0].i
                    for token in sent[secu_last_token.i-token_idx_offset+1:]:
                        alias = doc._.tokens_to_alias_map.get(token.i)
                        if alias and alias not in possible_aliases[secu]:
                            possible_aliases[secu].append(alias)
        to_eval = []
        for secu, aliases in possible_aliases.items():
            premerge_tokens =  get_premerge_tokens_for_span(secu)
            for alias in aliases:
                similarity_map = get_span_to_span_similarity_map(premerge_tokens, alias)
                dep_distance = get_dep_distance_between_spans(secu, alias)
                span_distance = get_span_distance(secu, alias)
                if dep_distance:
                    print(similarity_map)
                    score = get_secu_alias_similarity_score(alias, similarity_map, dep_distance, span_distance)
                    to_eval.append((secu, alias, score))
        for t in to_eval:
            print(t)

    def get_secu_alias_similarity_score(alias, similarity_map, dep_distance, span_distance):
        very_similar = sum([v > 0.65 for v in similarity_map.values()])
        very_similar_score = very_similar / len(alias) if very_similar != 0 else 0
        dep_distance_weight = 0.7
        span_distance_weight = 0.3
        dep_distance_score = dep_distance_weight * (1/dep_distance)
        span_distance_score = span_distance_weight * (10/span_distance)
        total_score = dep_distance_score + span_distance_score + very_similar_score
        return total_score
    
    # def get_secu_state(text):
    #     search = SpacyFilingTextSearch()
    #     from main.parser.filing_nlp import SECU, SECUQuantity, QuantityRelation, SourceQuantityRelation
    #     doc = search.nlp(text)
    #     '''
    #     need to find a smart way to include root_verb and root_noun.
    #     should probably add those to the SECU object and make extensions on the span/token
    #     then i need to get the time and place/issuing context of a relation
        
    #     how to get root_verb?
    #         go from main_secu and look if sentence has a verb root
    #         or do i need to look what kind clause it is and take the verbal root
    #         of said clause?
    #     '''
    #     secus = list()
    #     for secu in doc._.secus:
    #         secu_obj = SECU(secu)
    #         # state = secu._.amods
    #         # print(f"state: {state}")
    #         quants = search.get_secuquantities(doc, secu)
    #         for quant in quants:
    #             print(quant)
    #             quant_obj = SECUQuantity(quant["quantity"])
    #             root_verb = getattr(quant, "root_verb", None)
    #             root_noun = getattr(quant, "root_noun", None)
    #             if getattr(quant_obj.original, "source_secu", None) is not None:
    #                 source = quant["source_secu"]
    #                 rel = SourceQuantityRelation(quant_obj, secu_obj, source, root_verb=root_verb, root_noun=root_noun)
    #                 secu_obj.add_relation(rel)
    #             else:
    #                 rel = QuantityRelation(quant_obj, secu_obj, root_verb=root_verb, root_noun=root_noun)
    #                 secu_obj.add_relation(rel)
    #         secus.append(secu_obj)
    #     for s in secus:
    #         print(s)
                # for each in [main_secu, quantity]:
                #     print(each, type(each))
                #     print(each._.amods)
    
    def print_SECU_objects(text):
        search = SpacyFilingTextSearch()
        doc = search.nlp(text)
        secus = search.get_SECU_objects(doc)
        print(secus)
    
    # def get_secu_amods(text):
    #     search = SpacyFilingTextSearch()
    #     doc = search.nlp(text)
    #     print(f"getting amods for secus: {doc._.secus}")
    #     for secu in doc._.secus:
    #         print(f"amods: {search.get_secu_amods(secu)}")
        
    # def get_secuquantity(text):
    #     search = SpacyFilingTextSearch()
    #     doc = search.nlp(text)
    #     from main.parser.filing_nlp import extend_token_ent_to_span
    #     for secu in doc._.secus:
    #         quants = search.get_secuquantities(doc, secu)
    #         print(f"quants: {quants}")
    #         for quant in quants:
    #             print(f'quant: {quant["quantity"], type(quant["quantity"])}')
    #             print(quant["quantity"]._.secuquantity, quant["quantity"]._.secuquantity_unit)



    # compare_similarity(text)
    # displacy_ent_with_search(text)
    # text = "This Sentence is in regard to the total outstanding common stock of the Company consisting of 1000 shares. This Sentence is in regard to the 1000 shares of common stock outstanding as of may 15, 2020."
    text = "This Prospectus relates to the 1000 shares of common stock outstanding as of may 15, 2020 and to the 2000 $ spent on expansion. This Sentence is in regard to the 1000 outstanding shares of common stock as of may 15, 2020, which were issued pursuant to a Privat Placement."
    # get_secu_amods(text)
    # get_secuquantity(text)
    # get_secu_state(text)
    # print(print_SECU_objects(text))
    # displacy_dep_with_search(text)
    # displacy_ent_with_search(text)

    def try_get_SECU_objects():
        from main.parser.parsers import filing_factory
        from main.parser.filing_nlp import SpacyFilingTextSearch
        search = SpacyFilingTextSearch()
        path = r"C:\Users\Olivi\Testing\sec_scraping\tests\test_resources\filings\0001035976\S-3\000143774918017591\fncb20180927_s3.htm"
        filing_info = {
            "path": path,
            "filing_date": None,
            "accession_number": Path(path).parents[0].name,
            "cik": Path(path).parents[2].name,
            "file_number": "333-147568",
            "form_type": "S-3",
            "extension": ".htm"
        }
        filing: BaseHTMFiling = filing_factory.create_filing(**filing_info)
        text = filing.get_text_only()
        doc = search.nlp(text)
        print("_________________done creating doc_________________")
        secus = search.get_SECU_objects(doc)
        # print("got SECU objects: ", secus)
        for secu_key, values in secus.items():
            if "common" in secu_key:
                for secu in values:
                    # print("\t date_relations: ", secu.date_relations)
                    if secu.quantity_relations != []:
                        print(f"SECU_KEY: {secu.secu_key}")
                        # print("date_relations: ", secu.date_relations)
                        # print("quantity_relations: ", secu.quantity_relations)
                        for qrel in secu.quantity_relations:
                            print("\t quantity: ", qrel.quantity)
                    if secu.exercise_price:
                        print("exercise_price: ",secu.exercise_price)
                    if secu.expiry_date:
                        print("expiry_date: ", secu.expiry_date)

                    


    def try_own_dep_matcher():
        # text = "This is being furnished into a test sentence for a dependency matcher."
    
        # result = matcher.get_possible_candidates(pattern)
        # print([i for i in result])
        secus = []
        docs = []
        eps = []
        from main.parser.parsers import filing_factory
        path = r"C:\Users\Olivi\Testing\sec_scraping\tests\test_resources\filings\0001035976\S-3\000143774918017591\fncb20180927_s3.htm"
        filing_info = {
        "path": path,
        "filing_date": None,
        "accession_number": Path(path).parents[0].name,
        "cik": Path(path).parents[2].name,
        "file_number": "333-147568",
        "form_type": "S-3",
        "extension": ".htm"
    }
        filing = filing_factory.create_filing(**filing_info)
        for each in [
            # (
            #     "As of May 4, 2021, 46,522,759 shares of our common stock were issued and outstanding.",
            #     ),
            # (
            #     "The number of shares and percent of class stated above are calculated based upon 399,794,291 total shares outstanding as of May 16, 2022",
            #     ),
            # (
            #     "based on 34,190,415 total outstanding shares of common stock of the Company as of January 17, 2020. ",
            #     ),
            # (
            #     "are based on 30,823,573 shares outstanding on April 11, 2022. ",
            #     ),
            # (
            #     "This is based on 70,067,147 shares currently not outstanding of our Common Stock as of October 18, 2021.",
            #     ),
            # (
            #     "This is based on 41,959,545 shares, not including the 10000 Authorized shares of our Series A preferred stock as of October 26, 2020. ",
            #     ),
            # (
            #     "The Warrants have an exercise price of 10.50 $ per share and are exercisable for a period of 5 years from the date of issuance. ",
            # ),
            # ("The warrants we issued pursuant to our Private Placement are exercisable as of May 5, 2021.",),
            # ("The warrants we issued pursuant to our Private Placement can be exercised as of May 5, 2021.",),
            # ("The warrants we issued pursuant to our Private Placement are exercisable as of May 5, 2021.",),
            # (filing.get_text_only(),)
            # ("The warrants were exercisable on the grant date (April 26, 2016) and remain exercisable until April 26, 2021. \
            #   The warrants were exercisable on the grant date (April 26, 2018) and remain exercisable until May 26, 2021.",)
        ]:
            text = each[0]
            doc = search.nlp(text)
            docs.append(doc)
            secus = search.get_SECU_objects(doc)
            eps.append(secus)
            # for secu in doc._.secus:
                
                # ep = search.dep_getter.get_exercise_price(secu[0])
                # dr = search.secu_attr_getter.get_date_relation(secu[0])

                # parent = search.dep_getter.get_parent_verb(secu[0])
                # root = search.dep_getter.get_root_verb(secu[0])
                # eps.append((ep, dr, parent, root))
                # eps.append(dr)
        for ep in eps:
            for key, secus in ep.items():
                for secu in secus:
                    if "common" in secu.secu_key:
                        if secu.quantity_relations != []:
                            print("expiry_date", secu.expiry_date)
                            # print("date_relations: ", secu.date_relations)
                            # print("quantity_relations: ", secu.quantity_relations)
                            for qrel in secu.quantity_relations:
                                print("quantity: ", qrel.quantity)
                    # print(secu)
                    # print(secu.date_relations)
                    # print(secu.exercise_date)
                    # print("---------------")
                    # print(key)
                    # print(secu.date_relations)
                    # for date_relation in secu.date_relations["datetime"]:
                    #     print(date_relation)
                    # quants = secu.quantity_relations
                    # if quants:
                    #     for quant in quants:
                    #         print("quant date_relations: ", quant.quantity.date_relations)
                    #         print("quant amods: ", quant.quantity.amods)
                    #         print("quant parent_verb", quant.quantity.parent_verb)


                    # get = [secu.original, secu.date_relations, secu.exercise_price, secu.quantity_relations, secu.other_relations, secu.root_verb, secu.parent_verb, secu.aux_verbs, secu.amods]
                    # name = ["original", "date_relations", "exercise_price", "quantity_relations", "other_relations", "root_verb", "parent_verb", "aux_verbs", "amods"]
                    # for x, y in zip(get, name):
                    #     print(f"{y}: {x}")
                    # print(secu.parent_verb._.negated if secu.parent_verb else None)
        
        #     secu_objects = search.get_SECU_objects(doc)
        #     secus.append(secu_objects)
        # for i in secus:
        #     print(i)
        #     for _, s in i.items():
        #         for q in s:
        #             print(f"date_relation: {q.date_relation}")
        #     print("_________________________")
        # for doc in docs:
        #     for token in doc:
        #         if token._.negated is True:
        #             print(f"negated: {token}")
        # displacy.serve(docs, style="ent", options={
        #     "ents": ["SECU", "SECUQUANTITY", "CONTRACT", "DATE"],
        #     "colors": {"SECU": "#e171f0", "SECUQUANTITY": "#03fcb1", "CONTRACT": "green", "DATE": "yellow"}
        #     })
        # displacy.serve(docs, style="dep", options={"fine_grained": False, "collapse_phrases": False})

            # root_verb = matcher.get_root_verb(doc[8])
            # print(root_verb)
            # quantity = matcher.get_quantities(doc[8])
            # print(fquant, type(fquant), fquant.ent_type_, fquant._.amods, fquant._.secuquantity, fquant._.secuquantity_unit)
        # add secuquantity_unit to individual tokens
        # create amod getter for Token
        # rework this to account correctly for optional dependency condition
    # displacy_ent_with_search("This is based on 41,959,545 shares, not including shares of our Common Stock, currently outstanding as of October 26, 2020. ")
    # try_own_dep_matcher()
    # try_get_SECU_objects()
    # displacy_dep_with_search("The warrants we issued pursuant to our Private Placement are exercisable as of May 5, 2021.", show_lemmas=True)
    # displacy_ent_with_search("The warrants we issued pursuant to our Private Placement are exercisable as of May 5, 2021.")
# TODO: to test new SECU object, invoke an htmlfilingparser, spacyfilingtextsearch
    
    def match_context_sconj_action(doc):
        from main.parser.filing_nlp import SecurityDependencyAttributeMatcher
        attr_matcher = SecurityDependencyAttributeMatcher()
        results = []
        for ent in doc.ents:
            if ent.label_ == "SECUQUANTITY":
                sample = attr_matcher._get_context_sconj_action_from_secuquantity(ent.root)
                if sample:
                    results += sample
        return results 

    def context_sconj_action_sample(paths):
        results = []
        docs = []
        for path in paths:
            docs.append(get_doc_from_path(path))
        for doc in docs:
            result = match_context_sconj_action(doc)
            if result:
                results += result
        return results
                    
    def get_SECU_objects_from_text(text: str):
        search = SpacyFilingTextSearch()
        doc = search.nlp(text)
        secu_objects = search.get_SECU_objects(doc)
        return secu_objects
    
    
    def sample_for_SECU_objects():
        filing_paths = [
        # r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0000831547/S-3/000083154720000018/cleans-3.htm",
        # r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0001325879/S-3/000119312518218817/d439397ds3.htm",
        r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0001453593/S-3/000149315221008120/forms-3.htm"]
        # filing_paths = [i for i in Path(r"C:\Users\Olivi\Testing\sec_scraping\tests\test_resources\filings").rglob("*.htm")][2:3]
        parser = HTMFilingParser()
        # create logger file, log file name and log sSECU to info
        file_logger = logging.getLogger("fl")
        fl_handler = logging.FileHandler(r"C:\Users\Olivi\Desktop\SECU_objects_log.log")
        file_logger.addHandler(fl_handler)
        exercise_date = []
        exercise_price = []
        expiry = []
        secu_keys = []
        quants = []
        for path in filing_paths:
            file_logger.warning(f"current file: {path}")
            text = parser.clean_text_only_filing(parser.get_doc(path))
            secu_objects = get_SECU_objects_from_text(text)
            for secu_key, values in secu_objects.items():
                secu_keys.append(secu_key)
                has_exercise_price, has_expiry, has_exercise_date, has_quant = 0, 0, 0, 0
                for secu in values:
                    if secu.exercise_price:
                        has_exercise_price += 1
                    if secu.expiry_date:
                        has_expiry += 1
                    if secu.exercise_date:
                        has_exercise_date += 1
                    if secu.quantity_relations != []:
                        has_quant += len(secu.quantity_relations)

                exercise_date.append(has_exercise_date)
                exercise_price.append(has_exercise_price)
                expiry.append(has_expiry)
                quants.append(has_quant)
                # if "common" in secu_key or "warrant" in secu_key:
                for secu in values:
                    # print("\t date_relations: ", secu.date_relations)
                    if secu.quantity_relations != []:
                        file_logger.info(f"SECU_KEY: {secu.secu_key}")
                        # file_logger.info(("date_relations: ", secu.date_relations))
                        # file_logger.info(("quantity_relations: ", secu.quantity_relations))
                        for qrel in secu.quantity_relations:
                            file_logger.info(("\t quantity: ", qrel))
                    if secu.exercise_price:
                        file_logger.info(("exercise_price: ",secu.exercise_price))
                    if secu.expiry_date:
                        file_logger.info(("expiry_date: ", secu.expiry_date))
        df = pd.DataFrame(data={"name": secu_keys, "quantity_relations": quants ,"exercise_date": exercise_date, "exercise_price": exercise_price, "expiry": expiry})
        print(df)
    
    def get_doc_from_path(path):
        search = SpacyFilingTextSearch()
        parser = HTMFilingParser()
        text = parser.clean_text_only_filing(parser.get_doc(path))
        return search.nlp(text)
    
 
    
    # docs = []
    # for p in [
    #     r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0000831547/S-3/000083154720000018/cleans-3.htm",
    #     r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0001325879/S-3/000119312518218817/d439397ds3.htm",
    #     r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0001453593/S-3/000149315221008120/forms-3.htm"
    #     ]:
    #     docs.append(get_doc_from_cache(p))
    # search = SpacyFilingTextSearch()
    # text = "	Includes 7,111,112 shares of common stock that may be issued upon the exercise of warrants."
    # # displacy_dep_with_search(text)
    # doc = search.nlp(text)
    # print(match_context_sconj_action(doc))
    # d = context_sconj_action_sample([
    #     r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0001453593/S-3/000149315221008120/forms-3.htm"
    #     # r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0000831547/S-3/000083154720000018/cleans-3.htm",
    #     # r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0001325879/S-3/000119312518218817/d439397ds3.htm",
    #     # r"C:/Users/Olivi/Testing/sec_scraping/tests/test_resources/filings/0001453593/S-3/000149315221008120/forms-3.htm"
    #     ])
    # print(d)
    # matches_only = [i["match"] for i in d]
    # formatted_matches = []
    # for i in matches_only:
    #     r = {}
    #     for entry in i:
    #         # for entry in x:
    #         r[entry[1]] = entry[0]
    #     formatted_matches.append(r)
    # print(formatted_matches)
    # df = pd.DataFrame(data=formatted_matches)
    # print(df)
    
    # t = search.nlp(
    #     "The common stock outstanding after the offering is based on 113,299,612 shares of our common stock outstanding as of December 31, 2019 and the sale of 36,057,692 shares of our common stock at an assumed offering price of $2.08 per share, the last reported sale price of our common stock on the NASDAQ on March 16, 2020 and excludes the following"
    #     )
    # for i in t:
    #     print(i._.certainty_info, i._.negated)
    # from main.parser.filing_nlp import _get_amods_of_target_token
    # print(t.ent_type_, t._.amods, _get_amods_of_target_token(t), list(t.children))
    # secus = get_SECU_objects_from_text("The common stock outstanding after the offering is based on 113,299,612 shares of our common stock outstanding as of December 31, 2019 and the sale of 36,057,692 shares of our common stock at an assumed offering price of $2.08 per share, the last reported sale price of our common stock on the NASDAQ on March 16, 2020 and excludes the following")
    # print(secus)
    # sample_for_SECU_objects()
    # displacy_dep_with_search("The selling stockholders acquired these shares from us pursuant to a (i) Securities Purchase Agreement, dated February 22, 2021 pursuant to which we issued 8,888,890 shares of common stock, par value $0.000001 per share, at a purchase price of $2.25 per share, and a warrant to purchase up to 6,666,668 shares of common stock in a private placement, and (ii) Placement Agent Agreement, dated February 22, 2021, with A.G.P./Alliance Global Partners pursuant to which we issued warrants to purchase up to an aggregate of 444,444 shares of common stock.")
    # displacy_dep_with_search("The number of shares consists of 8,663,890 shares of our common stock and 6,497,918 shares of our common stock issuable upon exercise of the Investor Warrant.")
    # displacy_dep_with_search("Includes 7,111,112 shares of common stock that may be issued upon the exercise of warrants. The number of shares consists of 4,444 shares of our common stock issuable upon exercise of the Placement Agent Warrant.")
    # displacy_dep_with_search("perhaps, we issue certain stock if the time is right. we have the right to issue certain stock. we could issue certain stock. this agreement is concerning stock which we may issue. the stock which may be issued at some point. Perhaps, we will buy that. In may we may be buy that.",finegrained=True)
    # displacy_dep_with_search("As of March 31, 2021, we had outstanding warrants to purchase 7,532,390 shares of our common stock, and stock options to purchase 2,176,272 shares of our common stock.")
    # displacy_dep_with_search("As of April 20, 2022, we had warrants outstanding to issue 20000 shares of common stock at 3.0 $ per share.")
    # displacy_dep_with_search_multi([
    #     "On April 20, 2022, we sold 10000 shares of our common stock at a purchase price of $2 per share.",
    #     "On April 20, 2022, we had sold 10000 shares of our common stock at a purchase price of 2$ per share through a private placement.",
    #     "On April 20, 2022, we issued 10000 shares of our common stock at a price of 2$ per share through the placement agent.",
    #     ])
    # displacy_dep_with_search("We entered into an Agreement with Cerosa Inc. on May 5, 2021, pursuant to which we issued 5000 shares of common stock.")
    def try_alias_cache():
        alias_test_text = "On February 22, 2021, we entered into the Securities Purchase Agreement (the “Securities Purchase Agreement”), pursuant to which we agreed to issue the investor named therein (the “Investor”) 8,888,890 shares (the “Shares”) of our common stock, par value $0.000001 per share, at a purchase price of $2.25 per share, and a warrant to purchase up to 6,666,668 shares of our common stock (the “Investor Warrant”) in a private placement (the “Private Placement”). The closing of the Private Placement occurred on February 24, 2021. Pursuant to the Securities Purchase Agreement we also issued 10000 shares of our common stock through the Investor Warrant."
        from main.parser.filing_nlp_alias_setter import AliasMatcher, AliasSetter
        import spacy
        # nlp = spacy.load("en_core_web_lg")
        # doc = nlp(alias_test_text)
        search = SpacyFilingTextSearch()
        doc = search.nlp(alias_test_text)
        alias_matcher = AliasMatcher(search.nlp.vocab)
        doc = alias_matcher(doc)
        p = [ent if ent.label_ in ["CONTRACT", "ORG", "PER", "SECU"] else None for ent in doc.ents]
        origins = []
        for i in p:
            if i:
                origins.append(i)
        print(f"origins: {origins}")
        alias_setter = AliasSetter(search.nlp.vocab)
        doc = alias_setter(doc, origins)
        print(doc[10:13])
        print(doc._.alias_cache._origin_base_alias_map)
        print(doc._.alias_cache._base_alias_references)
        print(doc._.alias_cache.get_all_alias_references_by_origin(doc[10:13]))
    
    def get_dilution_db_connected_to_test():
        from boot import bootstrap_dilution_db
        from main.configs import FactoryConfig
        from main.services import unit_of_work
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine, text
        cnf = FactoryConfig("dev")()
        print(cnf.DILUTION_DB_CONNECTION_STRING)
        session_factory = sessionmaker(
            bind=create_engine(
            url=cnf.DILUTION_DB_CONNECTION_STRING,
            ),
            expire_on_commit=False
        )
        uow = unit_of_work.SqlAlchemyCompanyUnitOfWork(session_factory=session_factory)
        db = bootstrap_dilution_db(
            start_orm=True,
            config=cnf,
            uow=uow)
        return db
    
    db = get_dilution_db_connected_to_test()
    # from main.domain import commands, model
    # import datetime
    # with db.uow as uow:
    #     company = uow.company.get(symbol="CEI")
    #     cmd = commands.AddOutstandingSecurityFact(company.cik, company.symbol, attributes={"name": "common stock"},
    #     outstanding=[
    #         model.SecurityOutstanding(25543, datetime.date(2018, 1, 1)),
    #         model.SecurityOutstanding(30000, datetime.date(2018, 5, 2)),
    #     ])
    # db.bus.handle(cmd)
    with db.conn() as conn:
        # conn.execute("DELETE FROM securities_outstanding USING securities WHERE securities.company_id = (SELECT company_id FROM companies WHERE companies.symbol = 'CEI')")
        # res = conn.execute("SELECT * FROM securities_outstanding AS os JOIN securities AS s ON s.id = os.security_id WHERE s.company_id = (SELECT id FROM companies WHERE companies.symbol = 'CEI')", []).fetchall()
        res = conn.execute("SELECT * FROM securities_outstanding AS os JOIN securities AS s ON s.id = os.security_id WHERE s.company_id = (SELECT id FROM companies WHERE companies.symbol = 'CEI') ORDER BY os.security_id ASC").fetchall()
        print(res)
    # need to check the quantity relations for existance of: daterelation, amount, amods of parent secu and quant 
    # displacy_dep_with_search("The common stock outstanding after the offering is based on 113,299,612 shares of our common stock outstanding as of December 31, 2019 and the sale of 36,057,692 shares of our common stock at an assumed offering price of $2.08 per share, the last reported sale price of our common stock on the NASDAQ on March 16, 2020 and excludes the following")
# get text only of filing
# create doc of text    
# call get_SECU_objects on doc
# aggregate results
# print results
# compare to filing