from datetime import datetime, timedelta
from psycopg.rows import dict_row
from psycopg.errors import ProgrammingError, UniqueViolation, ForeignKeyViolation
from psycopg_pool import ConnectionPool
from psycopg import Connection, sql
from json import load, dump
from functools import reduce
from os import PathLike, path
import pandas as pd
import logging
from posixpath import join as urljoin
from pathlib import Path
from requests.exceptions import HTTPError
from scipy.fftpack import idct
from tqdm import tqdm
import json
from datetime import datetime

from pysec_downloader.downloader import Downloader
from main.data_aggregation.polygon_basic import PolygonClient
from main.data_aggregation.fact_extractor import get_cash_and_equivalents, get_outstanding_shares, get_cash_financing, get_cash_investing, get_cash_operating
from main.configs import cnf
from _constants import FORM_TYPES_INFO, EDGAR_BASE_ARCHIVE_URL


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



if cnf.ENV_STATE != "prod":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)





class DilutionDB:
    def __init__(self, connectionString):
        self.connectionString = connectionString
        self.pool = ConnectionPool(
            self.connectionString, kwargs={"row_factory": dict_row}
        )
        self.conn = self.pool.connection
        self.tracked_tickers = self._get_tracked_tickers_from_config()
        self.tracked_forms = self._get_tracked_forms_from_config()
        self.util = DilutionDBUtil(self)
        self.updater = DilutionDBUpdater(self)
        

    def _get_tracked_tickers_from_config(self):
        return cnf.APP_CONFIG.TRACKED_TICKERS
    
    def _get_tracked_forms_from_config(self):
        return cnf.APP_CONFIG.TRACKED_FORMS

    def execute_sql_file(self, path):
        with self.conn() as c:
            with open(path, "r") as sql:
                res = c.execute(sql.read())
                try:
                    for row in res:
                        logging.DEBUG(row)
                except ProgrammingError:
                    pass

    def _delete_all_tables(self):
        with self.conn() as c:
            with open("./main/sql/db_delete_all_tables.sql", "r") as sql:
                c.execute(sql.read())

    def _create_tables(self):
        with self.conn() as c:
            with open("./main/sql/dilution_db_schema.sql", "r") as sql:
                c.execute(sql.read())
    
    def read(self, query, values):
        with self.conn() as c:
            res = c.execute(query, values)
            rows = [row for row in res]
            return rows
    
    def read_all_companies(self):
        return self.read("SELECT id, cik, symbol, name_ as name FROM companies", [])

    def read_company_by_symbol(self, symbol: str):
        return self.read("SELECT * FROM companies WHERE symbol = %s", [symbol.upper()])
    
    def read_company_id_by_symbol(self, symbol: str):
        id = self.read("SELECT id FROM companies WHERE symbol = %s", [symbol.upper()])
        if id:
            return id
        else:
            return None
    
    def read_company_last_update(self, id: int):
        return self.read("SELECT * FROM companies_last_update WHERE company_id = %s", [id])

    def create_form_types(self):
        with self.conn() as c:
            for name, values in FORM_TYPES_INFO.items():
                category = values["category"]
                c.execute(
                    "INSERT INTO form_types(form_type, category) VALUES(%s, %s)",
                    [name, category]
                )
                print(category)

    def create_form_type(self, form_type, category):
        with self.conn() as c:
            c.execute(
                "INSERT INTO form_types(form_type, category) VALUES(%s, %s)",
                [form_type, category],
            )

    def create_sics(self):
        sics_path = path.normpath(
            path.join(path.dirname(path.abspath(__file__)), ".", "main/resources/sics.csv")
        )
        sics = pd.read_csv(sics_path)
        with self.conn() as c:
            with c.cursor().copy(
                "COPY sics(sic, industry, sector, division) FROM STDIN"
            ) as copy:
                for val in sics.values:
                    copy.write_row(val)

    def create_sic(self, sic, sector, industry, division):
        with self.conn() as c:
            c.execute(
                "INSERT INTO sics(sic, industry, sector, division) VALUES(%s, %s, %s, %s)",
                [sic, sector, industry, division],
            )

    def create_company(self, c: Connection, cik, sic, symbol, name, description, sic_description=None):
        if len(cik) < 10:
            raise ValueError("cik needs to be in 10 digit form")
        try:
            id = c.execute(
                "INSERT INTO companies(cik, sic, symbol, name_, description_) VALUES(%s, %s, %s, %s, %s) RETURNING id",
                [cik, sic, symbol, name, description],
            )
            id = id.fetchone()["id"]
            self.create_empty_company_last_update(c, id)
            return id
        except UniqueViolation:
            # logger.debug(f"couldnt create {symbol}:company, already present in db.")
            # logger.debug(f"querying and the company_id instead of creating it")
            c.rollback()
            id = c.execute(
                "SELECT id from companies WHERE symbol = %s AND cik = %s",
                [symbol, cik],
            )
            return id.fetchone()["id"]
        except Exception as e:
            if "fk_sic" in str(e):
                if sic_description is None:
                    raise ValueError(
                        f"couldnt create missing sic without sic_description, create_company called with: {locals()}"
                    )
                try:
                    self.create_sic(
                        sic, "unclassified", sic_description, "unclassified"
                    )
                except Exception as e:
                    logger.debug(
                        f"failed to add missing sic in create_company: e{e}"
                    )
                    raise e
                id = self.create_company(c, cik, sic, symbol, name, description)
                return id
            else:
                raise e
    
    def create_empty_company_last_update(self, c: Connection, id: int):
        c.execute("INSERT INTO company_last_update(company_id) VALUES(%s)", [id])

    def create_outstanding_shares(self, c: Connection, company_id, instant, amount):
        try:
            c.execute(
                "INSERT INTO outstanding_shares(company_id, instant, amount) VALUES(%s, %s, %s) ON CONFLICT ON CONSTRAINT outstanding_shares_company_id_instant_key DO NOTHING",
                [company_id, instant, amount],
            )
        except UniqueViolation as e:
            logger.debug(e)
            pass
        except Exception as e:
            raise e

    def create_net_cash_and_equivalents(self, c: Connection, company_id, instant, amount):
        try:
            c.execute(
                "INSERT INTO net_cash_and_equivalents(company_id, instant, amount) VALUES(%s, %s, %s) ON CONFLICT ON CONSTRAINT net_cash_and_equivalents_company_id_instant_key DO NOTHING",
                [company_id, instant, amount],
            )
        except UniqueViolation as e:
            logger.debug(e)
            pass
        except Exception as e:
            raise e

    def create_filing_link(
        self, c: Connection, company_id, filing_html, form_type, filing_date, description, file_number
    ):
        try:
            c.execute(
                "INSERT INTO filing_links(company_id, filing_html, form_type, filing_date, description_, file_number) VALUES(%s, %s, %s, %s, %s, %s)",
                [
                    company_id,
                    filing_html,
                    form_type,
                    filing_date,
                    description,
                    file_number,
                ],
            )
        except ForeignKeyViolation as e:
            if "fk_form_type" in str(e):
                logger.debug(f"fk_form_type violaton is trying to be resolved for {company_id, filing_html, form_type, filing_date, description, file_number}")
                self.create_form_type(form_type, "unspecified")
                self.create_filing_link(c, company_id, filing_html, form_type, filing_date, description, file_number)      
            else:
                raise e

    def create_cash_operating(self, c: Connection, company_id, from_date, to_date, amount):
        try:
            c.execute(
                "INSERT INTO cash_operating(company_id, from_date, to_date, amount) VALUES(%s, %s, %s, %s) ON CONFLICT ON CONSTRAINT cash_operating_company_id_from_date_to_date_key DO NOTHING",
                [company_id, from_date, to_date, amount],
            )
        except UniqueViolation as e:
            logger.debug(e)
            pass
        except Exception as e:
            raise e

    def create_cash_financing(self, c: Connection, company_id, from_date, to_date, amount):
        try:
            c.execute(
                "INSERT INTO cash_financing(company_id, from_date, to_date, amount) VALUES(%s, %s, %s, %s) ON CONFLICT ON CONSTRAINT cash_financing_company_id_from_date_to_date_key DO NOTHING",
                [company_id, from_date, to_date, amount],
            )
        except UniqueViolation as e:
            logger.debug(e)
            pass
        except Exception as e:
            logger.debug(f"raising e: {e}")
            raise e

    def create_cash_investing(self, c: Connection, company_id, from_date, to_date, amount):
        try:
            c.execute(
                "INSERT INTO cash_investing(company_id, from_date, to_date, amount) VALUES(%s, %s, %s, %s) ON CONFLICT ON CONSTRAINT cash_investing_company_id_from_date_to_date_key DO NOTHING",
                [company_id, from_date, to_date, amount],
            )
        except UniqueViolation as e:
            logger.debug(e)
            pass
        except Exception as e:
            raise e
    
    def create_cash_burn_summary(
        self,
        c: Connection,
        company_id,
        burn_rate,
        burn_rate_date,
        net_cash,
        net_cash_date,
        days_of_cash,
        days_of_cash_date
    ):
        try:
            c.execute("INSERT INTO cash_burn_summary("
                "company_id, burn_rate, burn_rate_date, net_cash, net_cash_date, days_of_cash, days_of_cash_date) "
                "VALUES (%(c_id)s, %(br)s, %(brd)s, %(nc)s, %(ncd)s, %(doc)s, %(docd)s) "
                "ON CONFLICT ON CONSTRAINT unique_company_id "
                "DO UPDATE SET "
                "burn_rate = %(br)s,"
                "burn_rate_date = %(brd)s,"
                "net_cash = %(nc)s,"
                "net_cash_date = %(ncd)s,"
                "days_of_cash = %(doc)s,"
                "days_of_cash_date = %(docd)s "
                "WHERE cash_burn_summary.company_id = %(c_id)s "
                "AND cash_burn_summary.burn_rate_date <= %(brd)s "
                "AND cash_burn_summary.net_cash_date <= %(ncd)s "
                "AND cash_burn_summary.days_of_cash_date <= %(docd)s"
                ,
                {"c_id": company_id,
                "br": burn_rate,
                "brd": burn_rate_date,
                "nc": net_cash,
                "ncd": net_cash_date,
                "doc": days_of_cash,
                "docd": days_of_cash_date})
        except Exception as e:
            logger.debug((
                        e, 
                        company_id,
                        burn_rate,
                        burn_rate_date,
                        net_cash,
                        net_cash_date,
                        days_of_cash,
                        days_of_cash_date))
            raise e

    def create_cash_burn_rate(
        self,
        c: Connection,
        company_id,
        burn_rate_operating,
        burn_rate_financing,
        burn_rate_investing,
        burn_rate_total,
        from_date,
        to_date,
    ):
        """UPSERT cash_burn_rate for a specific company. replaces with newer values on from_date conflict"""
        try:
            c.execute(
                (
                    "INSERT INTO cash_burn_rate("
                    "company_id, burn_rate_operating, burn_rate_financing,"
                    "burn_rate_investing, burn_rate_total, from_date, to_date) "
                    "VALUES(%(c_id)s, %(br_o)s, %(br_f)s, %(br_i)s, %(br_t)s, %(from_date)s, %(to_date)s) "
                    "ON CONFLICT ON CONSTRAINT unique_from_date "
                    "DO UPDATE SET "
                    "burn_rate_operating = %(br_o)s,"
                    "burn_rate_financing = %(br_f)s,"
                    "burn_rate_investing = %(br_i)s,"
                    "burn_rate_total = %(br_t)s,"
                    "to_date = %(to_date)s "
                    "WHERE cash_burn_rate.company_id = %(c_id)s "
                    "AND cash_burn_rate.from_date = %(from_date)s "
                    "AND cash_burn_rate.to_date < %(to_date)s"
                ),
                {
                    "c_id": company_id,
                    "br_o": burn_rate_operating,
                    "br_f": burn_rate_financing,
                    "br_i": burn_rate_investing,
                    "br_t": burn_rate_total,
                    "from_date": from_date,
                    "to_date": to_date,
                },
            )
        except UniqueViolation as e:
            logger.debug(e)
            if (
                "Unique-Constraint »cash_burn_rate_company_id_from_date_key«"
                in str(e)
            ):
                raise e
        except Exception as e:
            raise e
    
    def _init_outstanding_shares(self, connection: Connection, id: int, companyfacts):
        '''inital population of outstanding shares based on companyfacts'''
        self.updater._update_outstanding_shares_based_on_companyfacts(connection, id, companyfacts)
        self._update_company_lud(connection, id, "outstanding_shares_lud", datetime.utcnow().date())
    
    def _init_net_cash_and_equivalents(self, connection: Connection, id: int, companyfacts):
        '''inital population of net cash and equivalents based on companyfacts'''
        self.updater._update_net_cash_and_equivalents_based_on_companyfacts(connection, id, companyfacts)
        self._update_company_lud(connection, id, "net_cash_and_equivalents_lud", datetime.utcnow().date())
    
    def _init_cash_burn_summary(self, c: Connection, company_id):
        '''take the newest cash burn rate and net cash, calc days of cash left and UPSERT into summary'''
        net_cash = self.read(
            "SELECT * FROM net_cash_and_equivalents WHERE company_id = %s "
            "ORDER BY instant DESC LIMIT 1", [company_id]
            )[0]
        cash_burn = self.read(
            "SELECT * FROM cash_burn_rate WHERE company_id = %s "
            "ORDER BY to_date DESC LIMIT 1", [company_id]
            )[0]
        days_of_cash_date = datetime.now()
        cash_date = pd.to_datetime(net_cash["instant"]).date()
        burn_rate = cash_burn["burn_rate_total"]
        print(burn_rate)
        if burn_rate >= 0:
            days_of_cash = "infinity"
        else:
            abs_br = abs(burn_rate)
            prorated_cash_left = net_cash["amount"] - (abs_br*(datetime.now().date() - cash_date).days)
            if prorated_cash_left < 0:
                days_of_cash = abs(prorated_cash_left)/abs_br
            else:
                days_of_cash = prorated_cash_left/abs_br
        self.create_cash_burn_summary(
            c,
            company_id,
            burn_rate,
            cash_burn["to_date"],
            net_cash["amount"],
            net_cash["instant"],
            round(float(days_of_cash),2),
            days_of_cash_date
        )                
                

    def _init_cash_burn_rate(self, c: Connection, company_id):
        """calculate cash burn rate from db data and UPSERT into cash_burn_rate"""
        operating = self._calc_cash_burn_operating(company_id)
        investing = self._calc_cash_burn_investing(company_id)
        financing = self._calc_cash_burn_financing(company_id)
        # merge the cash burns into one dataframe
        cash_burn = reduce(
            lambda l, r: pd.merge(l, r, on=["from_date", "to_date"], how="outer"),
            [operating, investing, financing],
        )
        # remove overlapping intervals form_date, to_date keeping longest and assure they are date
        cash_burn_cleaned = self._clean_df_cash_burn(cash_burn)
        # write to db
        for r in cash_burn_cleaned.iloc:
            try:
                self.create_cash_burn_rate(
                    c,
                    company_id,
                    r.burn_rate_operating,
                    r.burn_rate_financing,
                    r.burn_rate_investing,
                    r.burn_rate_total,
                    r.from_date,
                    r.to_date,
                )
            except UniqueViolation as e:
                logger.debug(e)
            except Exception as e:
                logger.debug(e)
                raise e
    

    def _clean_df_cash_burn(self, cash_burn: pd.DataFrame):
        cash_burn["burn_rate_total"] = cash_burn[
            ["burn_rate_operating", "burn_rate_investing"]
        ].sum(axis=1)
        cleaned_idx = (
            cash_burn.groupby(["from_date"])["to_date"].transform(max)
            == cash_burn["to_date"]
        )
        cash_burn_cleaned = cash_burn[cleaned_idx]
        cash_burn_cleaned["from_date"].apply(lambda x: x.date())
        cash_burn_cleaned["to_date"].apply(lambda x: x.date())
        return cash_burn_cleaned

    def _calculate_df_cash_burn(self, cash_used: pd.DataFrame):
        df = cash_used.copy()
        df["from_date"] = pd.to_datetime(df["from_date"])
        df["to_date"] = pd.to_datetime(df["to_date"])
        df["period"] = (df["to_date"] - df["from_date"]).apply(lambda x: int(x.days))
        df["burn_rate"] = df["amount"] / df["period"]
        return df[["from_date", "to_date", "burn_rate"]]

    def _calc_cash_burn_operating(self, company_id):
        cash_used = self.read(
            "SELECT from_date, to_date, amount FROM cash_operating WHERE company_id = %s",
            [company_id],
        )
        df = pd.DataFrame(cash_used)
        logger.debug(df)
        cash_burn = self._calculate_df_cash_burn(df).rename(
            {"burn_rate": "burn_rate_operating"}, axis=1
        )
        return cash_burn

    def _calc_cash_burn_investing(self, company_id):
        cash_used = self.read(
            "SELECT from_date, to_date, amount FROM cash_investing WHERE company_id = %s",
            [company_id],
        )
        df = pd.DataFrame(cash_used)
        cash_burn = self._calculate_df_cash_burn(df).rename(
            {"burn_rate": "burn_rate_investing"}, axis=1
        )
        return cash_burn

    def _calc_cash_burn_financing(self, company_id):
        cash_used = self.read(
            "SELECT from_date, to_date, amount FROM cash_financing WHERE company_id = %s",
            [company_id],
        )
        df = pd.DataFrame(cash_used)
        cash_burn = self._calculate_df_cash_burn(df).rename(
            {"burn_rate": "burn_rate_financing"}, axis=1
        )
        return cash_burn
    
    def _assert_files_last_update_row(self):
        luds = self.read("SELECT * FROM files_last_update", [])
        if luds == []:
            with self.conn() as conn:
                conn.execute("INSERT INTO files_last_update(submissions_zip_lud, companyfacts_zip_lud) VALUES(NULL, NULL)", [])
    
    def _update_files_lud(self, c: Connection, col_name: str, lud: datetime):
        self._assert_files_last_update_row()
        old_lud = self.read("SELECT * FROM files_last_update", [])
        # print(old_lud)
        try:
            old_lud = old_lud[0][col_name]
        except KeyError as e:
            print(e)
            old_lud = None
        if old_lud is None:
        # cannot adapt type dict using placeholder %s, why?
            c.execute(sql.SQL("UPDATE files_last_update SET {} = %s").format(sql.Identifier(col_name)), [lud])
        else:
            c.execute(sql.SQL("UPDATE files_last_update SET {} = %s WHERE {} = %s").format(sql.Identifier(col_name), sql.Identifier(col_name)), [lud,  old_lud])

    def _update_company_lud(self, c: Connection, id: int, col_name: str, lud: datetime):
        res = c.execute(sql.SQL("UPDATE company_last_update SET {} = %s WHERE company_id = %s RETURNING *").format(sql.Identifier(col_name)), [lud, id])
        try:
            res.fetchall()
        except ProgrammingError as e:
            logger.debug(f"except ProgrammingError e: {e} and raise ValueError instead", exc_info=True)
            raise ValueError(f"No entry found for col_name and id. Make sure the entry for the id({id}) was created, and the col_name({col_name}) is spelled correctly.")


class DilutionDBUpdater:
    def __init__(self, db: DilutionDB):
        self.db = db
        self.dl = Downloader(root_path=cnf.DOWNLOADER_ROOT_PATH)

    def update_ticker(self, ticker: str):
        '''
        main entry point to update a ticker.

        Args:
            ticker: symbol of a company in the database 
        '''
        with self.db.connection() as conn:
            # make sure we are working with the most current bulk files
            self.update_bulk_files()
            # make sure we are working with the newest filings
            self.update_filings(conn, ticker)
            # parsing of filings goes here
            # -- parsing --
            # update all the information tables
            self.update_outstanding_shares(conn, ticker)
            self.update_net_cash_and_equivalents(conn, ticker)
            self.force_cashBurn_update(ticker)
            

    
    def _filings_needs_update(self, ticker: str, max_age=24):
        '''checks if the ticker has newer filings available with submissions.zip
        
        Args:
            ticker: symbol of a company
            max_age: max age since last update in hours
        
        Returns: 
            True or False depending if company needs an update of filings
        '''
        id = self.db.read_company_id_by_symbol(ticker)
        if id is not None:
            luds = self.db.read_company_last_update(id)
            if luds != []:
                filings_lud = luds[0]["filings_download_lud"]
                print(filings_lud)
            else:
                raise ValueError("company wasnt added to company_last_update table, make sure it was when creating the company!")
            now = datetime.utcnow()
            if (filings_lud - now) > timedelta(hours=max_age):
                return True, filings_lud
            else:
                return False, None
        else:
            return None, None

    
    def update_filings(self, connection: Connection, ticker: str):
        '''download new filings if available'''
        needs_update, filings_lud = self._filings_needs_update(ticker)
        if needs_update is None:
            raise AttributeError("ticker not found")
        if needs_update is True:
            company = self.db.read_company_by_symbol(ticker)
            if company != []:
                company = company[0]
                cik = company["cik"]
                id = company["id"]
                newer_filings = self.db.updater.dl.index_handler.get_newer_filings_meta(
                    cik,
                    str(filings_lud),
                    set(cnf.APP_CONFIG.TRACKED_FORMS))
                # add Returns: to get_newer_filings_meta
                print(newer_filings)
            # get filings newer than x from submissions
            # download said filings with downloader by accn
            pass
    
    def _file_needs_update(self, col_name: str, max_age: int):
        '''check if files need an update
        
        Args:
            col_name: the name of the column in the db files_last_update
                      table corresponding to the file we want to check
            max_age: max age that is allowed in hours
        
        Returns:
            bool: False if file age is less than max_age.
                  True if file age is more than max_age or no entry in db is present.
        '''
        lud = self.db.read(sql.SQL("SELECT {} as _lud FROM files_last_update").format(sql.Identifier(col_name)), [])
        if lud == []:
            return True
        _lud = lud["_lud"]
        if _lud is None:
            return True
        print(_lud)
        now = datetime.utcnow()
        if (now - pd.to_datetime(_lud)) >= timedelta(hours=max_age):
            return True
        else:
            return False
    
    def update_bulk_files(self):
        '''update submissions and companyfacts bulk files'''
        update = False
        with self.db.conn() as conn:
            if self._file_needs_update("submissions_zip_lud", max_age=24):
                print("updating submissions")
                self.dl.get_bulk_submissions()
                self.db._update_files_lud(conn, "submissions_zip_lud", datetime.utcnow())
                update = True
            if self._file_needs_update("companyfacts_zip_lud", max_age=24):
                self.dl.get_bulk_companyfacts()
                self.db._update_files_lud(conn, "companyfacts_zip_lud", datetime.utcnow())
                update = True
            if update is False:
                logger.info("No Bulk Files were updated as they are still current enough.")
            else:
                logger.info("Bulk Files were successfully updated")
    
    def update_net_cash_and_equivalents(self, connection: Connection, ticker: str):
        '''update the net_cash_and_equivalents information table and the cash_finacing, cash_investing and cash_operationg.
        
        Args:
            c: connection from the database connection pool
            ticker: symbol of company
        '''
        company = self.db.read_company_by_symbol(ticker)
        if company != []:
            company = company[0]
            cik = company["cik"]
            id = company["id"]
            self._update_net_cash_and_equivalents_based_on_companyfacts(
                connection, 
                id,
                self.db.util._get_companyfacts_file(cik))

            # update based on other things than companyfacts here

            # if all successfull write new lud
            self.db._update_company_lud(connection, "net_cash_and_equivalents_lud", datetime.utcnow().date())

    
    def update_outstanding_shares(self, connection: Connection, ticker: str):
        '''update the outstanding shares information table.
        
        Args:
            c: connection from the database connection pool
            ticker: symbol of company
        '''
        company = self.db.read_company_by_symbol(ticker)
        if company != []:
            company = company[0]
            cik = company["cik"]
            id = company["id"]
            self._update_outstanding_shares_based_on_companyfacts(
                connection,
                id,
                self.db.util._get_companyfacts_file(cik))

            # update outstanding shares by other means
                # no other ways yet
            # if all successfull write new lud
            self.db._update_company_lud(connection, "outstanding_shares_lud", datetime.utcnow().date())
        
    
    def _update_net_cash_and_equivalents_based_on_companyfacts(self, connection: Connection, id: int, companyfacts):
        try:
            net_cash = get_cash_and_equivalents(companyfacts)
            if net_cash is None:
                logger.critical(("couldnt get netcash extracted", id))
                return None
        except ValueError as e:
            logger.critical((e, id))
            raise e
        except KeyError as e:
            logger.critical((e, id))
            raise e
        logger.debug(f"net_cash: {net_cash}")
        for fact in net_cash:
            self.db.create_net_cash_and_equivalents(
                connection,
                id, fact["end"], fact["val"])
        # get the cash flow, partially tested
        try:
            cash_financing = get_cash_financing(companyfacts)
            for fact in cash_financing:
                self.db.create_cash_financing(
                    connection,
                    id, fact["start"], fact["end"], fact["val"]
                )
            cash_operating = get_cash_operating(companyfacts)
            for fact in cash_operating:
                self.db.create_cash_operating(
                    connection,
                    id, fact["start"], fact["end"], fact["val"]
                )
            cash_investing = get_cash_investing(companyfacts)
            for fact in cash_investing:
                self.db.create_cash_investing(
                    connection,
                    id, fact["start"], fact["end"], fact["val"]
                )
        except ValueError as e:
            logger.critical((e, id))
            logger.debug((e, id))
            raise e 

    
    def _update_outstanding_shares_based_on_companyfacts(self, c: Connection, id: int, companyfacts):
        try:
            outstanding_shares = get_outstanding_shares(companyfacts)
            if outstanding_shares is None:
                logger.critical(("couldnt get outstanding_shares extracted", id))
                return None
        except ValueError as e:
            logger.critical((e, id))
            return None
        except KeyError as e:
            logger.critical((e, id))
            return None
        logger.debug(f"outstanding_shares: {outstanding_shares}")
        for fact in outstanding_shares:
            self.db.create_outstanding_shares(
                c,
                id,
                fact["end"],
                fact["val"])
        

    
    def force_cashBurn_update(self, ticker: str):
        '''clear cash burn rate and summary data and recalculate for one company
        based on the available data from the database. Doesnt pull new data.'''
        company = db.read_company_by_symbol(ticker)[0]
        id = company["id"]
        try:
            with self.db.conn() as connection:
                try:
                    connection.execute("DELETE FROM cash_burn_rate * WHERE company_id = %s", [id])
                    connection.execute("DELETE FROM cash_burn_summary * WHERE company_id = %s", [id])
                    self.db._init_cash_burn_rate(connection, id)
                    self.db._init_cash_burn_summary(connection, id)
                except Exception as e:
                    connection.rollback()
                    logger.critical((f"couldnt force cashBurn update for ticker: {company}", e))
                    print(e)
                    pass
                else:
                    connection.commit()
        except KeyError as e:
            logger.info((e, company))

    def force_cashBurn_update_all(self):
        '''clear cash burn rate and summary data and recalculate for all companies
        based on the available data from the database. Doesnt pull new data.'''
        companies = self.db.read_all_companies()
        for company in companies:
            id = company["id"]
            try:
                with self.db.conn() as connection:
                    try:
                        connection.execute("DELETE FROM cash_burn_rate * WHERE company_id = %s", [id])
                        connection.execute("DELETE FROM cash_burn_summary * WHERE company_id = %s", [id])
                        self.db._init_cash_burn_rate(connection, id)
                        self.db._init_cash_burn_summary(connection, id)
                    except Exception as e:
                        connection.rollback()
                        logger.critical((f"couldnt force cashBurn update for ticker: {company}", e))
                        pass
                    else:
                        connection.commit()
            except KeyError as e:
                logger.info((e, company))
    

class DilutionDBUtil:
    def __init__(self, db):
        self.db = db
        self.logging_file = cnf.DEFAULT_LOGGING_FILE
        self.logger = logging.getLogger("DilutionDBUtil")
        self.logger_handler = logging.FileHandler(self.logging_file)
        self.logger_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.logger_handler)
    
    def reset_database(self):
        '''delete all data and recreate the tables'''
        self.db._delete_all_tables()
        self.db._create_tables()
        self.db.create_sics()
        self.db.create_form_types()

    def format_submissions_json_for_db(self, cik: str, sub: json, base_url: str=EDGAR_BASE_ARCHIVE_URL):
        '''
        converts submissions of a company file from submission.zip into a 
        list[dict] that is easier to use to create a filings_link entry in the DilutionDB.
        
        cik: 10 digit CIK
        sub: company file from submission.zip

        Returns:
            a list[dict] with following keys:
                "accessionNumber",
                "filingDate",
                "form",
                "fileNumber",
                "primaryDocument",
                "primaryDocDescription",
                "primaryDocument",
                "filing_html"
        '''
        filings = sub["filings"]["recent"]
        wanted_fields = ["accessionNumber", "filingDate", "form", "fileNumber", "primaryDocument", "primaryDocDescription", "primaryDocument"]
        cleaned = []
        for r in range(0, len(filings["accessionNumber"]), 1):
            entry = {}
            for field in wanted_fields:
                entry[field] = filings[field][r]
            entry["filing_html"] = self.build_submission_link(base_url, cik, entry["accessionNumber"].replace("-", ""), entry["primaryDocument"])
            cleaned.append(entry)
        return cleaned
    

    def build_submission_link(self, base_url, cik, accession_number, primary_document):
        return urljoin(urljoin(urljoin(base_url, cik), accession_number), primary_document)

    
    def inital_setup(
            self,
            dl_root_path: str,
            polygon_overview_files_path: str,
            polygon_api_key: str,
            forms: list[str],
            tickers: list[str],
            after: str=None):
        '''inital download of filings and data for population and inital population of the DilutionDB.
        
        Args:
            dl_root_path: root path for the downloaded data
            polygon_overview_files_path: path for the overview files
            polygon_api_key: api key for polygon
            forms: list of forms to download
            tickers: list of symbols to add
            after: after what date filings should be downloaded
            
        Returns:
            Nothing
        '''
        dl = Downloader(root_path=dl_root_path, user_agent=cnf.SEC_USER_AGENT)
        polygon_client = PolygonClient(polygon_api_key)
        if not Path(polygon_overview_files_path).exists():
            Path(polygon_overview_files_path).mkdir(parents=True)
            logger.debug(
                f"created overview_files_path and parent folders: {polygon_overview_files_path}")
        for ticker in tqdm(tickers, mininterval=0.5):
            self.get_filing_set(dl, ticker, forms, after=after, number_of_filings=9999)
            id = self._inital_population(self.db, dl, polygon_client, polygon_overview_files_path, ticker)
    
    def _get_companyfacts_file(self, cik10: str):
        cp_path = Path(cnf.DOWNLOADER_ROOT_PATH) / "companyfacts" / ("CIK"+cik10+".json")
        with open(cp_path, "r") as f:
            cp = json.load(f)
            return cp
    
    def _get_submissions_file(self, cik10: str):
        sub_path = Path(cnf.DOWNLOADER_ROOT_PATH) / "submissions" / ("CIK"+cik10+".json")
        with open(sub_path, "r") as f:
            sub = json.load(f)
            return sub
    
    def _inital_population(self, db: DilutionDB,  dl: Downloader, polygon_client: PolygonClient, polygon_overview_files_path: str, ticker: str):
        '''
        download none filing data and populate base information for a company.
        
        Returns:
            None when failing at some stage
            company.id of DilutionDB if successfull
        '''
        logger.info(f"currently working on: {ticker}")
        # get basic info and create company
        try:
            ov = polygon_client.get_overview_single_ticker(ticker)
        except HTTPError as e:
            logger.critical((e, ticker, "couldnt get overview file"))
            logger.info("couldnt get overview file")
            return None
        with open(Path(polygon_overview_files_path) / (ov["cik"] + ".json"), "w+") as f:
            json.dump(ov, f)
        logger.debug(f"overview_data: {ov}")
        # check that we have a sic otherwise assign  9999 --> nonclassifable
        try:
            ov["sic_code"]
        except KeyError:
            ov["sic_code"] = "9999"
            ov["sic_description"] = "Nonclassifiable"
        
        # load the xbrl facts 
        recent_submissions_file_path = dl.root_path / "submissions" / ("CIK"+ov["cik"]+".json")
        try:
            companyfacts = self._get_companyfacts_file(ov["cik"])
            
            # query for wanted xbrl facts and write to db
            # start db transaction to ensure only complete companies get added
            id = None
            with db.conn() as connection:
                try:
                    id = db.create_company(
                        connection,
                        ov["cik"],
                        ov["sic_code"],
                        ticker, ov["name"],
                        ov["description"],
                        ov["sic_description"])
                    if not id:
                        raise ValueError("couldnt get the company id from create_company")
                except Exception as e:
                    logger.critical(("Phase1.0", e, ticker), exc_info=True)
                    connection.rollback()
                    return None
                else:
                    connection.commit()
            with db.conn() as connection:   
                try:
                    db._init_outstanding_shares(connection, id, companyfacts)
                    db._init_net_cash_and_equivalents(connection , id, companyfacts)
                except Exception as e:
                    logger.critical(("Phase1.1", e, ticker), exc_info=True)
                    connection.rollback()
                    with db.conn() as del_connection:
                        del_connection.execute("DELETE FROM companies * WHERE id = %s", [id])
                    return None
                else:
                    connection.commit()
                
                
            with db.conn() as connection:
                try:
                    db._init_cash_burn_rate(connection, id)
                except KeyError as e:
                    logger.critical(("Phase2.1", e, ticker), exc_info=True)
                    connection.rollback()
                    raise e
                else:
                    connection.commit()
                try:
                    db._init_cash_burn_summary(connection, id)
                except KeyError as e:
                    logger.critical(("Phase2.2", e, ticker), exc_info=True)
                    connection.rollback()
                    raise e
                else:
                    connection.commit()
                
            with db.conn() as connection1:      
                # populate filing_links table from submissions.zip
                try:
                    submissions_file = self._get_submissions_file(ov["cik"])
                    submissions = db.util.format_submissions_json_for_db(
                        ov["cik"],
                        submissions_file)
                    for s in submissions:
                        with db.conn() as connection2:
                            try:
                                db.create_filing_link(
                                    connection2,
                                    id,
                                    s["filing_html"],
                                    s["form"],
                                    s["filingDate"],
                                    s["primaryDocDescription"],
                                    s["fileNumber"])
                            except Exception as e:
                                logger.debug((e, s))
                                connection2.rollback()
                            else:
                                connection2.commit()
                except Exception as e:
                    logger.critical(("Phase3", e, ticker), exc_info=True)
                    connection1.rollback()
                    raise e
                else:
                    connection1.commit()
        except FileNotFoundError as e:
            logger.critical((e,"This is mostlikely a fund or trust and not a company.", ticker))
            return None
        else:
            return id
    
    def get_filing_set(self, downloader: Downloader, ticker: str, forms: list, after: str, number_of_filings: int = 250):
        # download the last 2 years of relevant filings
        if after is None:
            after = str((datetime.now() - timedelta(weeks=104)).date())
        for form in forms:
        #     # add check for existing file in pysec_donwloader so i dont download file twice
            try:
                downloader.get_filings(ticker, form, after, number_of_filings=number_of_filings)
            except Exception as e:
                self.logger.info((ticker, form, e), exc_info=True)
                pass

    def _get_overview_files(self, dl_root_path: str, polygon_overview_files_path: str, polygon_api_key: str, tickers: list):
        # get the polygon overview files
        polygon_client = PolygonClient(polygon_api_key)
        dl = Downloader(dl_root_path)
        if not Path(polygon_overview_files_path).exists():
            Path(polygon_overview_files_path).mkdir(parents=True)
            logger.debug(
                f"created overview_files_path and parent folders: {polygon_overview_files_path}")
        for ticker in tqdm(tickers):
            logger.info(f"currently working on: {ticker}")
            # get basic info and create company
            try:
                ov = polygon_client.get_overview_single_ticker(ticker)
            except HTTPError as e:
                logger.critical((e, ticker, "couldnt get overview file"), exc_info=True)
                logger.info("couldnt get overview file")
                continue
            try:
                ov["cik"]
            except KeyError as e:
                print(e)
                continue
            with open(Path(polygon_overview_files_path) / (ov["cik"] + ".json"), "w+") as f:
                dump(ov, f)

    # def create_tracked_companies(self):
    #     base_path = config["polygon"]["overview_files_path"]
    #     for ticker in self.tracked_tickers:
    #         company_overview_file_path = path.join(base_path, f"{ticker}.json")
    #         with open(company_overview_file_path, "r") as company:
    #             corp = load(company)
    #             try:
    #                 self.create_company(corp["cik"], corp["sic_code"], ticker, corp["name"], corp["description"])
    #             except UniqueViolation:
    #                 logger.debug("couldnt create {}:company, already present in db.", ticker)
    #                 pass
    #             except Exception as e:
    #                 if "fk_sic" in str(e):
    #                     self.create_sic(corp["sic_code"], "unclassfied", corp["sic_description"], "unclassified")
    #                     self.create_company(corp["cik"], corp["sic_code"], ticker, corp["name"], corp["description"])
    #                 else:
    #                     raise e


if __name__ == "__main__":

    db = DilutionDB(cnf.DILUTION_DB_CONNECTION_STRING)
    with open("./resources/company_tickers.json", "r") as f:
        tickers = list(load(f).keys())
        db.util._get_overview_files(cnf.DOWNLOADER_ROOT_PATH, cnf.POLYGON_OVERVIEW_FILES_PATH, cnf.POLYGON_API_KEY, tickers)
    
    # fake_args1 = [1, 0, 0, 0, 0, "2010-04-01", "2011-02-27"]
    # fake_args2 = [1, 0, 0, 0, 0, "2010-04-01", "2011-04-27"]
    # db.init_cash_burn_summary(1)
    # print(db.read("SELECT * FROM cash_burn_summary", []))
    # db.init_cash_burn_rate(1)
    

    # db.create_cash_burn_rate(*fake_args2)
    # with db.conn() as c:
    #     res = c.execute("SELECT * FROM companies JOIN sics ON companies.sic = sics.sic")
    #     for r in res:
    #         print(r)
    # db._delete_all_tables()
    # db._create_tables()
    # db.create_sics()
    # db.create_tracked_companies()
