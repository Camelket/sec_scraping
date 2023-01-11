from datetime import datetime, timedelta
from email.parser import Parser
import platform
from typing import Callable, Optional
from numpy import number
from psycopg.rows import dict_row
from psycopg.errors import ProgrammingError, UniqueViolation, ForeignKeyViolation
from psycopg_pool import ConnectionPool
from psycopg import Connection, sql
from json import load, dump
from functools import reduce
from os import PathLike, access, path
import pandas as pd
import logging
from posixpath import join as urljoin
from pathlib import Path
from requests.exceptions import HTTPError
from scipy.fftpack import idct
from tqdm import tqdm
import json
from datetime import datetime
from sqlalchemy import inspect


from pysec_downloader.downloader import Downloader
from main.data_aggregation.polygon_basic import PolygonClient
from main.data_aggregation.fact_extractor import get_cash_and_equivalents, get_outstanding_shares, get_cash_financing, get_cash_investing, get_cash_operating
from main.parser.filings_base import Filing
import main.parser.extractors as extractors
import main.parser.parsers as parsers
from main.configs import cnf, GlobalConfig
from _constants import FORM_TYPES_INFO, EDGAR_BASE_ARCHIVE_URL
from main.services.messagebus import MessageBus
from main.domain import commands, model

# from main.adapters.repository import AbstractRepository
# from main.adapters.orm import start_mappers
from main.services.unit_of_work import AbstractUnitOfWork
from main.domain import model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


'''
THIS IS OUTDATED AS OF 10.11.22 (TODO: NEEDS REWRITE)
To add a new .htm Form to the parsing pipeline:
    in parsers.py:
    1) Add a new entry to the filing_factory_default
       of form: (form_type: str, ".htm", HTMFiling)
    in extractors.py:
    2) write a new Extractor for the form, subclassing AbstractFilingExtractor
       and BaseExtractor (for creation of FilingValue) or write your
       own implementation.
    3) add the newly written extractor to the pipeline by adding an entry to the
       extractor_factory_default of form: (form_type: str, ".htm", newly written extractor)
    
    if any new field_names were added:
       in dilution_db.py:
       1) write a new function that takes as args:
          filing_value: FilingValue
          connection: Connection
          and contains the logic what to do with that FilingValue
       2) register the function into the DBUploaderFactory: either add it in
          init_uploaders to the defaults list or call db.uploader.register_uploader
          after instantiation of the DilutionDB

To add a new file type to the parsing pipeline:
    in parsers.py:
    1) create a new filing type by subclassing Filing
    then follow .htm example above, replacing ".htm" with the new extension and
    "HTMFiling" with the newly created Filing class


quick overview of the main classes involved in the parsing pipeline:
    Filing:
        Classes that represents a filing. The idea: we transform one file type
        regardless of form type into a common starting point.
    FilingSection:
        Used to break a filing into smaller logical parts
    Parsers:
        Used to transform a raw filing into a Filing
    Extractors:
        extract values from a Filing and create FilingValues 
    Factories:
        used so we can more easily organise and avoid if/else jungle.
        There might be a better solution.


'''

if cnf.ENV_STATE != "prod":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


class DilutionDB:
    def __init__(self, config: GlobalConfig, uow: AbstractUnitOfWork, message_bus: MessageBus, tracked_tickers: Optional[list[str]]=None, tracked_forms: Optional[list[str]]=None):
        self.uow = uow
        self.bus = message_bus
        self.config = config
        self.pool = ConnectionPool(
            config.DILUTION_DB_CONNECTION_STRING, kwargs={"row_factory": dict_row}
        )
        self.conn = self.pool.connection
        self.tracked_tickers = self._get_tracked_tickers_from_config() if tracked_tickers is None else tracked_tickers
        self.tracked_forms = self._get_tracked_forms_from_config() if tracked_forms is None else tracked_forms
        self.util = DilutionDBUtil(self)
        self.updater = DilutionDBUpdater(self)
    
    def _is_security_present(self, symbol: str, security_name: str):
        '''return bool if a security with given security_name is present for company with given symbol.'''
        with self.uow as uow:
            company: model.Company = uow.company.get(symbol=symbol, lazy=True)
            if not company:
                raise model.CompanyNotFound(f"{symbol} couldnt be found in the database. Make sure to add the company before querying against it to avoid this exception.")
            security = company.get_security_by_name(name=security_name)
            if security:
                return True
            else:
                return False
    
    def _ensure_common_stock_security_is_added(self, cik: str, symbol: str) -> None:
        try:
            is_present = self._is_security_present(symbol, "common stock")
        except model.CompanyNotFound as e:
            logger.warning(f"common stock not added to {symbol}, because of Exception: {e}")
            return
        if is_present:
            return
        else:
            cmd = commands.AddSecurities(
                cik,
                symbol, 
                securities=[model.Security(
                                model.CommonShare(),
                                name="common stock",
                                underlying=None)]
                )
            self.bus.handle(cmd)
            return

    def _get_tracked_tickers_from_config(self):
        return self.config.APP_CONFIG.TRACKED_TICKERS
    
    def _get_tracked_forms_from_config(self):
        return self.config.APP_CONFIG.TRACKED_FORMS

    def execute_sql_file(self, path: str | PathLike):
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
    
    def inital_setup(self, reset_database=False, after="2010-01-01", before=None):
        if reset_database is True:
            self.util.inital_table_setup()
        self.updater.update_bulk_files()
        self.util.inital_company_setup(
            cnf.DOWNLOADER_ROOT_PATH,
            cnf.POLYGON_OVERVIEW_FILES_PATH,
            cnf.POLYGON_API_KEY,
            cnf.APP_CONFIG.TRACKED_FORMS,
            cnf.APP_CONFIG.TRACKED_TICKERS,
            after=after,
            before=before
        )
        for ticker in cnf.APP_CONFIG.TRACKED_TICKERS:
            self.updater.update_ticker(ticker)
    
    def read_all_companies(self):
        return self.read("SELECT id, cik, symbol, name_ as name FROM companies", [])

    def read_company_by_symbol(self, symbol: str):
        return self.read("SELECT * FROM companies WHERE symbol = %s", [symbol.upper()])
    
    def read_company_id_by_symbol(self, symbol: str):
        id = self.read("SELECT id FROM companies WHERE symbol = %s", [symbol.upper()])
        print(id)
        if id:
            try:
                id = id[0]["id"]
            except KeyError:
                return None
            else:
                return id
        else:
            return None
    
    def read_company_last_update(self, id: int):
        return self.read("SELECT * FROM company_last_update WHERE company_id = %s", [id])
    
    def read_parsed_filings(self, id: int):
        return self.read("SELECT * FROM filing_parse_history WHERE company_id = %s", [id])

    def create_form_types(self):
        with self.conn() as c:
            for name, values in FORM_TYPES_INFO.items():
                category = values["category"]
                c.execute(
                    "INSERT INTO form_types(form_type, category) VALUES(%s, %s)",
                    [name, category]
                )
                # print(category)

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
            c.commit()

    def create_company(self, c: Connection, cik, sic, symbol, name, description, sic_description=None):
        logger.debug("create company executing..")
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
            logger.debug(f"excepted e: {e}", exc_info=True)
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

    def create_outstanding_shares(self, connection: Connection, company_id, instant, amount):
        try:
            connection.execute(
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
        self, company_id, filing_html, accn, form_type, filing_date, description, file_number
    ):
        with self.conn() as co:
            try:
                co.execute(
                    "INSERT INTO filing_links(company_id, filing_html, accn, form_type, filing_date, description_, file_number) VALUES(%s, %s, %s, %s, %s, %s, %s)",
                    [
                        company_id,
                        filing_html,
                        accn,
                        form_type,
                        filing_date,
                        description,
                        file_number,
                    ],
                )
            except ForeignKeyViolation as e:
                if "fk_form_type" in str(e):
                    co.rollback()
                    logger.debug(f"fk_form_type violaton is trying to be resolved for {company_id, filing_html, accn, form_type, filing_date, description, file_number}")
                    self.create_form_type(form_type, "unspecified")
                    self.create_filing_link(company_id, filing_html, accn, form_type, filing_date, description, file_number)      
                else:
                    raise e
            else:
                co.commit()

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
        logger.debug(f"create_cash_burn_summary called...")
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
        else:
            logger.debug("create_cash_burn_summary done..")
            return

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
    
    def _init_outstanding_shares(self, connection: Connection, id: int, cik: str, symbol: str, companyfacts):
        '''inital population of outstanding shares based on companyfacts'''
        self.updater._update_outstanding_common_shares_based_on_companyfacts(cik=cik, symbol=symbol, companyfacts=companyfacts)
        self._update_company_lud(connection, id, "outstanding_shares_lud", datetime.utcnow())
    
    def _init_net_cash_and_equivalents(self, connection: Connection, id: int, companyfacts):
        '''inital population of net cash and equivalents based on companyfacts'''
        self.updater._update_net_cash_and_equivalents_based_on_companyfacts(connection, id, companyfacts)
        self._update_company_lud(connection, id, "net_cash_and_equivalents_lud", datetime.utcnow())
    
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
        logger.debug(f"burn_rate: {burn_rate}")
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
        return       
                

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
        '''makes  sure the files_last_update table has a row otherwise create it and inserts NUlls'''
        luds = self.read("SELECT * FROM files_last_update", [])
        if luds == []:
            with self.conn() as conn:
                conn.execute("INSERT INTO files_last_update(submissions_zip_lud, companyfacts_zip_lud) VALUES(NULL, NULL)", [])
    
    def _update_files_lud(self, connection: Connection, col_name: str, lud: datetime):
        '''
        updates the table files_last_update with a new last_update_date
        
        Args:
            connection: connection from the database connection pool
            col_name: column name to be updated (submissions_zip_lud, companyfacts_zip_lud..) 
            lud: last update date
        '''
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
            connection.execute(sql.SQL("UPDATE files_last_update SET {} = %s").format(sql.Identifier(col_name)), [lud])
        else:
            connection.execute(sql.SQL("UPDATE files_last_update SET {} = %s WHERE {} = %s").format(sql.Identifier(col_name), sql.Identifier(col_name)), [lud,  old_lud])

    def _update_company_lud(self, c: Connection, id: int, col_name: str, lud: datetime):
        '''
        updates the table company_last_update with a new last_update_date
        
        Args:
            connection: connection from the database connection pool
            col_name: column name to be updated (filings_download_lud, outstanding_shares_lud, ...) 
            lud: last update date
        '''
        res = c.execute(sql.SQL("UPDATE company_last_update SET {} = %s WHERE company_id = %s RETURNING *").format(sql.Identifier(col_name)), [lud, id])
        try:
            res.fetchall()
        except ProgrammingError as e:
            logger.debug(f"except ProgrammingError e: {e} and raise ValueError instead", exc_info=True)
            raise ValueError(f"No entry found for col_name and id. Make sure the entry for the id({id}) was created, and the col_name({col_name}) is spelled correctly.")

    def _update_filing_parse_history(self, connection: Connection, id: int, accession_number: str, date_parsed: datetime):
        '''
        upsert the table filing_parse_history with a new date_parsed or creates a new entry
        
        Args:
            connection: connection from the database connection pool
            id: company_id
            accession_number: no dash  accession number of the filing ("000147793221001077")
            date_parsed: when the parse of filing and therefor addition of new filing_values happend
        '''
        res = connection.execute((
            "INSERT INTO filing_parse_history "
            "(company_id, accession_number, date_parsed) "
            "VALUES (%(c_id)s, %(accn)s, %(d_parsed)s) "
            "ON CONFLICT ON CONSTRAINT unique_co_accn "
            "DO UPDATE SET "
            "date_parsed = %(d_parsed)s "
            "WHERE filing_parse_history.company_id = %(c_id)s "
            "AND filing_parse_history.accession_number = %(accn)s "
            "RETURNING *"),
            {"c_id": id,
            "accn": accession_number,
            "d_parsed": date_parsed}
            )
    
    def _create_filing_values(self, connection: Connection, id: int, accession_number: str, date_parsed: datetime, field_name: str, field_values: dict):
        '''
        upsert the table filing_values.
        
        Args:
            connection: connection from the database connection pool
            id: company_id
            accesssion_number: no dash accesssion number of the filing from which
                               which the values were extracted ("000147793221001077")
            date_parsed: on which date values where extracted
            field_name: what field/group was extracted ("outstanding_shares")
            field_values: json of the values 
        '''
        res = connection.execute((
            "INSERT INTO filing_values "
            "(company_id, accession_number, date_parsed, field_name, field_values) "
            "VALUES (%(c_id)s, %(accn)s, %(d_parsed)s, %(f_name)s, %(f_values)s) "
            "ON CONFLICT ON CONSTRAINT unique_co_accn_field_name "
            "DO UPDATE SET "
            "date_parsed = %(d_parsed)s "
            "field_values = %(f_values)s "
            "WHERE company_id = %(c_id)s "
            "AND accession_number = %(accn)s " 
            "RETURNING *"),
            {"c_id": id,
            "accn": accession_number,
            "d_parsed": date_parsed,
            "f_name": field_name,
            "f_values": field_values}
            )
    
        

class DilutionDBUpdater:
    def __init__(self, db: DilutionDB):
        self.db = db
        self.dl = Downloader(root_path=db.config.DOWNLOADER_ROOT_PATH)
    
    def update_ticker(self, symbol: str):
        '''
        main entry point to update a ticker.

        Args:
            symbol: symbol of a company in the database 
        '''
        self.update_bulk_files()
        self.update_outstanding_common_shares(symbol=symbol)
        with self.db.conn() as conn:
            self.update_net_cash_and_equivalents(connection=conn, ticker=symbol)
        self.force_cashBurn_update(ticker=symbol)
        with self.db.conn() as conn:
            self.update_local_filings(connection=conn, symbol=symbol)
            self.update_filing_values(connection=conn, symbol=symbol)
    
    def update_outstanding_common_shares(self, symbol: str):
        '''update the outstanding shares information table.
        
        Args:
            symbol: symbol of company
        '''
        company = self.db.read_company_by_symbol(symbol=symbol)
        if company != []:
            company = company[0]
            cik = company["cik"]
            id = company["id"]
            self._update_outstanding_common_shares_based_on_companyfacts(
                cik=cik,
                symbol=symbol,
                companyfacts=self.db.util._get_companyfacts_file(cik)
                )

            # update outstanding shares by other means
                # no other ways yet
            # if all successfull write new lud
            with self.db.conn() as connection:
                #TODO: add a command and handler for update_company_lud
                self.db._update_company_lud(connection, id, "outstanding_shares_lud", datetime.utcnow())
        
    def _update_outstanding_common_shares_based_on_companyfacts(self, cik: str, symbol: str, companyfacts):
        try:
            outstanding_shares = get_outstanding_shares(companyfacts)
            if outstanding_shares is None:
                logger.warning(("couldnt get outstanding_shares extracted from companyfacts for: ", symbol))
                return None
        except ValueError as e:
            logger.critical((e, symbol))
            return None
        except KeyError as e:
            logger.critical((e, symbol))
            return None
        logger.debug(f"outstanding_shares: {outstanding_shares}")
        outstanding = []
        for fact in outstanding_shares:
            outstanding.append(model.SecurityOutstanding(fact["val"], fact["end"]))
        cmd = commands.AddOutstandingSecurityFact(
            cik=cik,
            symbol=symbol,
            attributes={"name": "common stock"},
            outstanding=outstanding)
        self.db.bus.handle(cmd) 

    def update_filing_values(self, connection: Connection, symbol: str):
       self.db.util.parse_filings(connection, symbol)
                

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
            else:
                raise ValueError("company wasnt added to company_last_update table, make sure it was when creating the company!")
            if filings_lud is None:
                return True, None
            now = datetime.utcnow()
            if (now - filings_lud) > timedelta(hours=max_age):
                return True, filings_lud
            else:
                return False, None
        else:
            return None, None

    
    def update_local_filings(self, connection: Connection, symbol: str):
        '''download new filings if available'''
        logger.debug(f"updating local filings for: {symbol}")
        needs_update, filings_lud = self._filings_needs_update(symbol)
        if filings_lud is None:
            filings_lud = datetime(2000, 1, 1).date()
        logger.debug(f"local filings---")
        logger.debug(f"     needs_update: {needs_update}")
        logger.debug(f"     filings_lud: {filings_lud}")
        if needs_update is None:
            raise AttributeError("ticker not found")
        if needs_update is True:
            company = self.db.read_company_by_symbol(symbol)
            if company != []:
                company = company[0]
                cik = company["cik"]
                id = company["id"]
                possible_newer_filings = self.db.updater.dl.index_handler.get_newer_filings_meta(
                    cik,
                    str(filings_lud),
                    set(self.db.config.APP_CONFIG.TRACKED_FORMS))
                try:
                    newer_filings = possible_newer_filings[cik]
                    # logger.debug(f"possible_newer_filings: {possible_newer_filings}")
                    
                except KeyError:
                    return None
                else:
                    if len(newer_filings) > 0:
                        for filing in newer_filings:
                            form_type = filing[0]
                            accession_number = filing[1]
                            file_name = filing[2]
                            filing_date = filing[3]
                            file_nums = filing[4]
                            
                            self.db.updater.dl.get_filing_by_accession_number(cik=cik, form_type=form_type, accession_number=accession_number, save_name=file_name, filing_date=filing_date, file_nums=file_nums)
                        self.db._update_company_lud(connection, id, "filings_download_lud", datetime.utcnow())
                    else:
                        return None
    
    def _file_needs_update_lud_filesystem(self, file_path: Path, max_age: int):
        if not file_path.exists():
            return True
        if platform.system() == "Windows":
            folder_mtime = get_folder_mtime(file_path)
            logger.info(f"folder_mtime: {folder_mtime}, path: {file_path}")
            return is_outdated(folder_mtime, max_age=timedelta(hours=max_age), now=datetime.now())
        else:
            raise OSError(f"this function only works properly on Windows systems for now.") 
    

    
    def _file_needs_update_lud_database(self, col_name: str, max_age: int):
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
        _lud = lud[0]["_lud"]
        if _lud is None:
            logger.debug(f"no last update for '{col_name}' found.")
            return True
        logger.debug(f"last update of: {col_name} was on {_lud}")
        now = datetime.utcnow()
        if (now - pd.to_datetime(_lud)) >= timedelta(hours=max_age):
            return True
        else:
            return False
    
    def update_bulk_files(self):
        '''update submissions and companyfacts bulk files'''
        with self.db.conn() as conn:
            if self._file_needs_update_lud_filesystem(Path(cnf.DOWNLOADER_ROOT_PATH) / "submissions", max_age=288):
            # if self._file_needs_update_lud_database("submissions_zip_lud", max_age=24):
                logger.debug("updating submissions.zip...")
                self.dl.get_bulk_submissions()
                self.db._update_files_lud(conn, "submissions_zip_lud", datetime.utcnow())
                logger.info("successfully updated submissions.zip")
            if self._file_needs_update_lud_filesystem(Path(cnf.DOWNLOADER_ROOT_PATH) / "companyfacts", max_age=288):
            # if self._file_needs_update_lud_database("companyfacts_zip_lud", max_age=24):
                logger.debug("updating companyfacts.zip...")
                self.dl.get_bulk_companyfacts()
                self.db._update_files_lud(conn, "companyfacts_zip_lud", datetime.utcnow())
                logger.info("successfully updated companyfacts.zip")
            
    
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
            self.db._update_company_lud(connection, id, "net_cash_and_equivalents_lud", datetime.utcnow())       
    
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

    def force_cashBurn_update(self, ticker: str):
        '''clear cash burn rate and summary data and recalculate for one company
        based on the available data from the database. Doesnt pull new data.'''
        company = self.db.read_company_by_symbol(ticker)[0]
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
    def __init__(self, db: DilutionDB):
        self.db = db
        self.logging_file = db.config.DEFAULT_LOGGING_FILE
        self._init_logging_file()
        self.logger = logging.getLogger("DilutionDBUtil")
        self.logger_handler = logging.FileHandler(self.logging_file)
        self.logger_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.logger_handler)
    
    def _init_logging_file(self):
        if Path(self.logging_file).exists():
            pass
        else:
            Path(self.logging_file).parent.mkdir(parents=True)
            with open(self.logging_file, "w") as f:
                pass

    
    def reset_database(self):
        '''delete all data and recreate the tables'''
        self.db._delete_all_tables()
        self.inital_table_setup()
    
    def inital_table_setup(self):
        self.db._delete_all_tables()
        self.db._create_tables()
        self.db.create_sics()
        self.db.create_form_types()

    def parse_filings(self, connection: Connection, ticker: str, forms: Optional[list[str]]=None):
        
        '''
        parses the unparsed local filings of a ticker.
        
        Args:
            connection: connection from the database connection pool
            ticker: symbol associated with a company
            forms: what form types to parse, defaults to None (uses the tracked_forms variable of the supplied DilutionDB list)
                   Can be supplied with 'all' to try and parse all filings regardless of form_type
        '''
        if forms is None:
            forms = set(self.db.tracked_forms)
        company = self.db.read_company_by_symbol(ticker)
        if company != []:
            company = company[0]
            cik = company["cik"]
            id = company["id"]
        else:
            raise ValueError(f"no company with that symbol was found. ticker: {ticker}")
        with self.db.uow as uow:
            company = uow.company.get(ticker)
            if not company:
                raise ValueError(f"no company with that symbol({ticker}) was found.")
            logger.info(f"parsing filings for: {company}")
            cik = company.cik
            id = company.id
            logger.info(f"company has id({id}) and cik({cik})")

        # get unparsed_filings
        unparsed_filings = self.get_unparsed_filings(id, cik)
        logger.info(f"found {len(unparsed_filings)} unparsed filings.")
        logger.info(f"only allowing forms: {forms}")
        for idx, unparsed in enumerate(unparsed_filings):
            logger.debug(f"currently on unparsed_filing number {idx}")
            form_type, file_number, file_path, filing_date, accession_number = unparsed.values()
            if (form_type in forms) or (forms == "all"):
                logger.debug(f"values passed to _create_filing: {form_type, accession_number, file_path, filing_date, cik, file_number}")
                try:
                    filings = self._create_filing(form_type, accession_number, file_path, filing_date, cik, file_number)
                except ValueError as e:
                    logger.error(f"_create_filing ran into a ValueError: {e}", exc_info=True)
                else:
                    logger.debug(f"_create_filing created: {len(filings)} filings out of one file.")
                    for filing in filings:
                        with self.db.uow as uow:
                            company = uow.company.get(ticker)
                            uow.session.expunge(company)
                        company = self._parse_filing(filing, company)


    def _create_filing(self,
        form_type: str,
        accession_number: str,
        path: str,
        filing_date: str = None,
        cik: str = None,
        file_number: str = None,
        ) -> list[Filing]:
        extension = Path(path).suffix
        filings = parsers.filing_factory.create_filing(
            extension=extension,
            path=path,
            filing_date=filing_date,
            accession_number=accession_number,
            cik=cik,
            file_number=file_number,
            form_type=form_type)
        if isinstance(filings, list):
            return filings
        else:
            return [filings]

    def _parse_filing(self, 
        filing: Filing,
        company: model.Company
        ) -> list[list[commands.Command]]:
        try:
            extractor: extractors.AbstractFilingExtractor = extractors.extractor_factory.get_extractor(filing.form_type, filing.extension)
        except ValueError as e:
            logger.info(f"excepted ValueError in _parse_filing: {e}", exc_info=True)
            return []
        try:
            extractor_return = extractor.extract_form_values(filing, company, self.db.bus)
        except Exception as e:
            logger.error(f"encountered error during extraction: {e}", exc_info=True)
        else:
            with self.db.conn() as c:
                self.db._update_filing_parse_history(c, company.id, filing.accession_number, datetime.now().date())
            return extractor_return

    def reparse_local_filings(self, symbol: str, form: str):
        with self.db.uow as uow:
            company = uow.company.get(symbol=symbol, lazy=False)
            cik = company.cik
            uow.session.expunge(company)
        all_filings = self.db.updater.dl.index_handler.get_local_filings_by_cik(cik)
        for filing_entry in all_filings:
            form_type, file_number, file_path, filing_date, accession_number = filing_entry.values()
            if form == form_type:
                try:
                    filings = self._create_filing(form_type, accession_number, file_path, filing_date, cik, file_number)
                except ValueError as e:
                    logger.error(f"_create_filing ran into a ValueError: {e}", exc_info=True)
                else:
                    logger.debug(f"_create_filing created: {len(filings)} filings out of one file.")
                    for filing in filings:
                        company = self._parse_filing(filing, company)
    
    def get_unparsed_filings(self, id: int, cik: str):
        # get all the filings and their accession number that are present locally
        all_filings = self.db.updater.dl.index_handler.get_local_filings_by_cik(cik)
        parsed_filings = self.db.read_parsed_filings(id)
        unparsed = []
        if parsed_filings != []:
            parsed_accn = set([p["accesssion_number"] for p in parsed_filings])
            for filing in all_filings:
                if filing["accn"] in parsed_accn:
                    pass
                else:
                    unparsed.append(filing)
            unparsed = sorted(unparsed, key=lambda x: x["filing_date"])
            return unparsed
        return all_filings

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
            entry["accessionNumber"] = _ensure_no_dash_accn(entry["accessionNumber"])
            cleaned.append(entry)
        return cleaned
    

    def build_submission_link(self, base_url, cik, accession_number, primary_document):
        return urljoin(urljoin(urljoin(base_url, cik), accession_number), primary_document)

    
    def inital_company_setup(
            self,
            dl_root_path: str,
            polygon_overview_files_path: str,
            polygon_api_key: str,
            forms: list[str],
            tickers: list[str],
            after: str=None,
            before: str=None):
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
        dl = Downloader(root_path=dl_root_path, user_agent=self.db.config.SEC_USER_AGENT)
        polygon_client = PolygonClient(polygon_api_key)
        if not Path(polygon_overview_files_path).exists():
            Path(polygon_overview_files_path).mkdir(parents=True)
            logger.debug(
                f"created overview_files_path and parent folders: {polygon_overview_files_path}")
        for ticker in tqdm(tickers, mininterval=0.5):
            # self.get_filing_set(dl, ticker, forms, after=after, before=before, number_of_filings=9999)
            id = self._inital_population(self.db, dl, polygon_client, polygon_overview_files_path, ticker)
    
    def _get_companyfacts_file(self, cik10: str):
        cp_path = Path(self.db.config.DOWNLOADER_ROOT_PATH) / "companyfacts" / ("CIK"+cik10+".json")
        with open(cp_path, "r") as f:
            cp = json.load(f)
            return cp
    
    def _get_submissions_file(self, cik10: str):
        sub_path = Path(self.db.config.DOWNLOADER_ROOT_PATH) / "submissions" / ("CIK"+cik10+".json")
        with open(sub_path, "r") as f:
            sub = json.load(f)
            return sub
    
    # def _inital_population_with_domain_model(self, db: DilutionDB, dl: Downloader, polygon_client: PolygonClient, polygon_overview_files_path: str, ticker: str):
    #     logger.info(f"starting inital_population_with_domain_model for ticker: {ticker}")
    #     try:
    #         ov = polygon_client.get_overview_single_ticker(ticker)
    #     except HTTPError as e:
    #         logger.critical((e, ticker, "couldnt get overview file"))
    #         logger.info("couldnt get overview file")
    #         return None
    #     with open(Path(polygon_overview_files_path) / (ov["cik"] + ".json"), "w+") as f:
    #         json.dump(ov, f)
    #     logger.debug(f"overview_data: {ov}")
    #     # check that we have a sic otherwise assign  9999 --> nonclassifable
    #     try:
    #         ov["sic_code"]
    #     except KeyError:
    #         ov["sic_code"] = "9999"
    #         ov["sic_description"] = "Nonclassifiable"
    #     db.bus.handle(
    #         commands.AddCompany(
    #             model.Company(
    #                 name=ov["name"],
    #                 cik=ov["cik"],
    #                 sic=ov["sic_code"],
    #                 description_=ov["description"],
    #             )
    #     ))
    
    
    def _inital_population(self, db: DilutionDB,  dl: Downloader, polygon_client: PolygonClient, polygon_overview_files_path: str, ticker: str):
        # TODO[epic=important]: rework to use model and uow instead of direct calls where applicable
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
                        ticker,
                        ov["name"],
                        ov["description"],
                        ov["sic_description"])
                    if not id:
                        raise ValueError("couldnt get the company id from create_company")
                except Exception as e:
                    logger.critical(("Phase1.0", e, ticker), exc_info=True)
                    connection.rollback()
                    return None
                else:
                    logger.info(f"successfully added company: {ov['cik']}")
                    connection.commit()
            with db.conn() as connection:   
                try:
                    db._init_outstanding_shares(connection, id, ov["cik"], ticker, companyfacts)
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
                    logger.debug("commiting Phase2")
                    connection.commit()
                
            with db.conn() as connection1: 
                logger.debug("starting Phase3")     
                # populate filing_links table from submissions.zip
                try:
                    submissions_file = self._get_submissions_file(ov["cik"])
                    submissions = db.util.format_submissions_json_for_db(
                        ov["cik"],
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
                            logger.debug((e, s))
                                
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
    
    def get_filing_set(self, downloader: Downloader, ticker: str, forms: list, after: str, before: str=None, number_of_filings: int = 250):
        # download the last 2 years of relevant filings
        if after is None:
            after = str((datetime.now() - timedelta(weeks=104)).date())
        for form in forms:
        #     # add check for existing file in pysec_donwloader so i dont download file twice
            try:
                if before is None:
                    downloader.get_filings(ticker, form, after_date=after, number_of_filings=number_of_filings)
                else:
                    downloader.get_filings(ticker, form, after_date=after, before_date=before, number_of_filings=number_of_filings)
            except Exception as e:
                self.logger.info((ticker, form, e), exc_info=True)
                pass

    def _get_overview_files(self, polygon_overview_files_path: str, polygon_api_key: str, tickers: list):
        # get the polygon overview files
        polygon_client = PolygonClient(polygon_api_key)
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
                logger.error(f"Error during access of 'cik': {e}", exc_info=True)
                continue
            with open(Path(polygon_overview_files_path) / (ov["cik"] + ".json"), "w+") as f:
                dump(ov, f)

def get_folder_mtime(file_path: Path):
    if not file_path.exists():
        raise OSError(f"folder doesnt exist: {file_path}")
    if file_path.is_file():
        raise OSError(f"file_path isnt a folder but a file: {file_path}")
    return pd.to_datetime(file_path.stat().st_mtime, unit="s")

def is_outdated(comparison_time: datetime, max_age: timedelta, now: datetime=datetime.now()):
    if (comparison_time + max_age) < now:
        return True
    else:
        return False 

def _ensure_no_dash_accn(accn: str):
    if len(accn) == 18:
        return accn
    else:
        return accn.replace("-", "")

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
    # with open("./resources/company_tickers.json", "r") as f:
    #     tickers = list(load(f).keys())
    #     db.util._get_overview_files(cnf.DOWNLOADER_ROOT_PATH, cnf.POLYGON_OVERVIEW_FILES_PATH, cnf.POLYGON_API_KEY, tickers)
    
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
