
from bs4 import BeautifulSoup
from numpy import number
from urllib3 import connection_from_url
from dilution_db import DilutionDBUpdater
# from dilution_db import DilutionDB
# from main.data_aggregation.polygon_basic import PolygonClient
# from main.configs import cnf

# from main.data_aggregation.bulk_files import update_bulk_files


from main.parser.parsers import HTMFilingParser, Parser8K, ParserSC13D, HTMFiling
from pathlib import Path

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.errors import ProgrammingError, UniqueViolation, ForeignKeyViolation
from psycopg_pool import ConnectionPool

import pandas as pd
import logging
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
    def download_samples(root, forms=["S-1", "S-3", "SC13D"]):
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
            tickers = list(json.load(f).keys())
            for ticker in tqdm(tickers[5000:5020]):
                get_filing_set(dl, ticker, forms, "2017-01-01", number_of_filings=50)
    
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
        db.util.inital_setup(db, cnf.DOWNLOADER_ROOT_PATH, cnf.POLYGON_OVERVIEW_FILES_PATH, cnf.POLYGON_API_KEY, cnf.APP_CONFIG.TRACKED_FORMS, ["CEI", "HYMC", "GOEV", "SKLZ", "ACTG", "INO", "GNUS"])

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
        db.util.inital_setup(
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
        matches = extractors.spacy_text_search.match_outstanding_shares(text)
        print(matches)
        # doc = nlp(text)
        # for ent in doc.ents:
        #     print(ent.label_, ent.text)
        # for token in doc:
        #     print(token.ent_type_)
        
    def create_htm_filing():
        fake_filing_info = {
            # "path": r"C:\Users\Olivi\Desktop\test_set\set_s3/filings/0000002178/S-3/000000217820000138/a2020forms-3.htm",
            "path": r"C:\Users\Olivi\Desktop\test_set\set_s3/filings/0001325879/S-3/000119312520289207/d201932ds3.htm",
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
        from main.parser.extractors import spacy_text_search
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
        from main.parser.extractors import spacy_text_search
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
        filing: HTMFiling = filing_factory.create_filing(**fake_filing_info)
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
            filing: HTMFiling = filing_factory.create_filing(**info)

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
            filing: HTMFiling = filing_factory.create_filing(**info)

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


   

    # download_samples(r"C:\Users\Olivi\Desktop\test_set\set_s3", forms=["S-3"])
    
    # dl = Downloader(cnf.DOWNLOADER_ROOT_PATH)
    # dl.get_filings("CEI", "8-K", after_date="2021-01-01", number_of_filings=10)
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

    # item count in all 8-k's of the filings-database
    # open_filings_in_browser(r"C:\Users\Olivi\Desktop\test_set\set_s3\filings", "S-3")

    filing: HTMFiling = create_htm_filing()
    # print(f"filing: {filing}")
    fp = filing.get_section("front page")
    if isinstance(fp, list):
        raise AttributeError(f"couldnt get the front page section; sections present: {[s.title for s in filing.sections]}")
    registration_table  = fp.get_tables(classification="registration_table", table_type="extracted")
    if registration_table is not None:
        if len(registration_table) == 1:
            registration_table = registration_table[0]["parsed_table"]
            print(registration_table)
            registration_df = pd.DataFrame(registration_table[1:], columns=["Title", "Amount", "Price Per Unit", "Price Aggregate", "Fee"])
            values = {}
            print(registration_df)
            if "Total" in registration_df["Title"].values:
                    row = registration_df[registration_df["Title"] == "Total"]
                    print(type(row["Amount"].item()), row["Amount"].item())
                    if row["Amount"].item() != "":
                        print(f"amount: {row['Amount']}")

    # print(ex.text_only)
    # for section in filing.sections:
    #     print(section.title, len(section.content))
    # cover_pages = filing.get_sections(re.compile("cover page", re.I))
    # for cv in cover_pages:
        # print(cv.title, cv.text_only)