import inspect
from multiprocessing.sharedctypes import Value
import pytest
from pytest_postgresql import factories
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, clear_mappers
from sqlalchemy.exc import IntegrityError
from psycopg.rows import dict_row
from psycopg.errors import ProgrammingError
from psycopg_pool import ConnectionPool
from psycopg import Connection
import datetime
from dilution_db import DilutionDB


from main.configs import FactoryConfig, GlobalConfig
from main.domain import model
from boot import bootstrap_dilution_db
from main.domain import commands


cnf = FactoryConfig(GlobalConfig(ENV_STATE="test").ENV_STATE)()
dilution_db_schema = str(Path(__file__).parent.parent / "main" / "sql" / "dilution_db_schema.sql")
delete_tables = str(Path(__file__).parent.parent / "main" / "sql" / "db_delete_all_tables.sql")

fake_filing_link = {
    "company_id": 1,
    'filing_html': 'https://www.sec.gov/Archives/edgar/data/0001309082/000158069520000391/cei-s4a_101220.htm',
    "accn": "123456789123456789",
    'form_type': 'S-4/A',
    'filing_date': '2020-10-14',
    'description': 'AMENDMENT TO FORM S-4',
    'file_number': '333-238927',
}

fake_company = {
    "name":"random company",
    "cik":"0000000001",
    "sic":9000,
    "symbol":"RANC",
    "description_":"random description"
}

fake_common_stock = {
    "name": "common stock",
    "secu_attributes": model.CommonShare(name="common stock")
}

fake_shelf_registration = {
    "accn":"000143774918017591",
    "file_number": "1",
    "form_type": "S-3",
    "capacity": 10000
}


fake_resale_registration = {
    "accn": "000143774918017592",
    "file_number": 2,
    "form_type": "S-3",
    "filing_date": datetime.date(2018, 9, 29),
    "effect_date": None,
    "last_update": None,
    "expiry": None,
    "total_amount_raised": None
}

fake_shelf_offering = {
    "offering_type": "ATM",
    "accn": "000143774918017592",
    "anticipated_offering_amount": 5000,
    "commencment_date": datetime.date(2018, 10, 1),
    "end_date": datetime.date(2021, 10, 1)
}

company_data = {
        "sics": {
            "sic": 9000,
            "sector": "test_sector",
            "industry": "test_industry",
            "division": "test_division"
        },
        "form_types": {
            "form_type": "S-3",
            "category": "prospectus"
        },
        "companies": {
            "id": 1,
            "cik": "0000000001",
            "sic": 9000,
            "symbol": "RAND",
            "name_": "Rand Inc.",
            "description_": "Solely a test company meant for usage with pytest"
        },
        "shelf_registrations": {
            "id": 1,
            "company_id": 1,
            "accn": "0000123456789",
            "file_number": "222",
            "form_type": "S-3",
            "capacity": 100000,
            "total_amount_raised": 0,
            "effect_date": datetime.date(2022, 1, 1),
            "last_update": datetime.datetime.now().date(),
            "expiry": datetime.date(2022, 1, 1) + datetime.timedelta(days=1125),
            "filing_date": datetime.date(2022, 1, 1),
            "is_active": True
        },
        "securities": {
            "id": 1,
            "company_id": 1,
            "security_name": "common stock",
            "security_type": "CommonShare",
            "underlying_security_id": None,
            "security_attributes": model.CommonShare(name="common stock").json()
        }
    }

company_data_expected = {
        "sics": {
            "sic": 9000,
            "sector": "test_sector",
            "industry": "test_industry",
            "division": "test_division"
        },
        "form_types": {
            "form_type": "S-3",
            "category": "prospectus"
        },
        "companies": {
            "id": 1,
            "cik": "0000000001",
            "sic": 9000,
            "symbol": "RAND",
            "name_": "Rand Inc.",
            "description_": "Solely a test company meant for usage with pytest"
        },
        "shelf_registrations": {
            "id": 1,
            "company_id": 1,
            "accn": "0000123456789",
            "file_number": "222",
            "form_type": "S-3",
            "capacity": 100000,
            "total_amount_raised": 0,
            "effect_date": datetime.date(2022, 1, 1),
            "last_update": datetime.datetime.now().date(),
            "expiry": datetime.date(2022, 1, 1) + datetime.timedelta(days=1125),
            "filing_date": datetime.date(2022, 1, 1),
            "is_active": True
        },
        "securities": {
            "id": 1,
            "company_id": 1,
            "security_name": "common stock",
            "security_type": "CommonShare",
            "underlying_security_id": None,
            "security_attributes": model.CommonShare(name="common stock").dict()
        }
    }



def load_schema(user, password, host, port, dbname):
    connectionstring = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    pool = ConnectionPool(
            connectionstring, kwargs={"row_factory": dict_row}
        )
    with pool.connection() as c:
        with open(delete_tables, "r") as sql:
            c.execute(sql.read())
    with pool.connection() as c:
        with open(dilution_db_schema, "r") as sql:
            c.execute(sql.read())
         

postgresql_my_proc = factories.postgresql_noproc(
    host=cnf.DILUTION_DB_HOST,
    port=cnf.DILUTION_DB_PORT,
    user=cnf.DILUTION_DB_USER,
    password=cnf.DILUTION_DB_PASSWORD,
    dbname=cnf.DILUTION_DB_DATABASE_NAME,
    load=[load_schema]
    )
postgresql_np = factories.postgresql("postgresql_my_proc", dbname=cnf.DILUTION_DB_DATABASE_NAME)

@pytest.fixture
def get_bootstrapped_dilution_db(postgresql_np):
    # hacky fix for a bug i dont understand: when running pytest over plugin creating mappers works fine, over cmd pytest creating mappers gives error of mappers already defined, despite logging only showing invocation once.. calling clear_mappers ensures this doesnt happen.
    clear_mappers()
    dilution_db = bootstrap_dilution_db(
        start_orm=True,
        config=cnf
    )
    yield dilution_db
    del dilution_db
    clear_mappers()




@pytest.fixture
def populate_database(get_session):
    session = get_session
    for table, v in company_data.items():
        columns = ", ".join(list(v.keys()))
        values = ", ".join(["'" + str(value) + "'" if value is not None else "NULL" for value in v.values()])
        try:
            t = text(f"INSERT INTO {table}({columns}) VALUES({values})")
            session.execute(t)
        except IntegrityError as e:
            session.rollback()
            print(f"!encountered error during population of database: {e}")
        else:
            session.commit()

def add_example_company(db: DilutionDB):
    sic = model.Sic(9000, "random sector", "random industry", "random divison")
    uow = db.uow
    with uow as u:
        u.session.add(sic)
        u.session.commit()
    company = model.Company(**fake_company)
    with uow as u:
        u.company.add(company)
        u.commit()

def add_example_common_stock(db: DilutionDB):
    with db.uow as u:
        company: model.Company = u.company.get(symbol=fake_company["symbol"])
        assert len(company.securities) == 0
        company.add_security(
            model.Security(**fake_common_stock))
        u.commit()
    
def test_connect(get_session):
    session = get_session
    tables = session.execute(text("Select * from information_schema.tables")).fetchall()
    assert len(tables) > 200    

def test_inserts(get_session, populate_database):
    session = get_session
    values = []
    for k, _ in company_data.items():
        v = session.execute(text(f"SELECT * FROM {k}"))
        values.append(v.fetchall())
    for expected, received in zip(company_data_expected.values(), values):
        print(f"expected: {tuple(expected.values())}")
        print(f"received: {received}")
        for v1, v2 in zip(tuple(expected.values()), received[0]):
            assert v1 == v2

def test_addition_of_filing_link_with_unknown_form_type(get_bootstrapped_dilution_db, populate_database, get_session):
    db: DilutionDB = get_bootstrapped_dilution_db
    session = get_session
    db.create_filing_link(
        **fake_filing_link
    )
    assert ('S-4/A', 'unspecified') in session.execute(text("SELECT * FROM form_types")).fetchall()
    

def test_dilution_db_inital_population(get_bootstrapped_dilution_db, get_uow, get_test_config):
    cnf = get_test_config
    if cnf.ENV_STATE != "test":
        raise ValueError(f"config other than 'test' loaded, aborting test. loaded env: {cnf}")
    test_tickers = ["CEI"]
    test_forms = ["S-3"]
    db: DilutionDB = get_bootstrapped_dilution_db
    db.tracked_forms = test_forms
    db.tracked_tickers = test_tickers
    # setup tables/make sure they exist
    db.util.inital_table_setup()
    # make sure bulk zip files are up to date
    db.updater.update_bulk_files()
    # do donwloads and inital population
    db.util.inital_company_setup(
        cnf.DOWNLOADER_ROOT_PATH,
        cnf.POLYGON_OVERVIEW_FILES_PATH,
        cnf.POLYGON_API_KEY,
        test_forms,
        test_tickers,
        after="2016-01-01",
        before="2017-06-01")
    # do parse of existing filings
    for ticker in test_tickers:
        with db.conn() as conn:
            db.updater.update_ticker(ticker)
    # do download and parse of newer filings
    # for ticker in test_tickers:
    #     db.updater.update_ticker(ticker)
    
    uow = get_uow
    for ticker in test_tickers:
        with uow as u:
            company = u.company.get(ticker)
            resale_file_numbers = [x.file_number for x in company.resales]
            for each in ["333-213713", "333-214085"]:
                assert each in resale_file_numbers
            shelf_file_numbers = [x.file_number for x in company.shelfs]
            assert "333-216231" in shelf_file_numbers
    

def test_live_add_company(get_bootstrapped_dilution_db, get_session):
    sic = model.Sic(9000, "random sector", "random industry", "random divison")
    db = get_bootstrapped_dilution_db
    uow = db.uow
    session = get_session
    with uow as u:
        u.session.add(sic)
        u.session.commit()
    result = session.execute(text("SELECT * FROM sics")).fetchall()
    assert result == [(9000, 'random sector', 'random industry', 'random divison')]

    company = model.Company(**fake_company)
    with uow as u:
        u.company.add(company)
        u.commit()
    with uow as u:
        result:model.Company = u.company.get(symbol=fake_company["symbol"])
        
    with uow as u:
        local_res = u.session.merge(result)
        local_res.add_security(
            model.Security(**{
            "name": "common stock",
            "secu_attributes": model.CommonShare(name="common stock")
        })
        )
        print(local_res, company)
        assert local_res == company

def test_live_add_security(get_bootstrapped_dilution_db):
    db = get_bootstrapped_dilution_db
    uow = db.uow
    add_example_company(db)        
    with uow as u:
        company: model.Company = u.company.get(symbol=fake_company["symbol"])
        assert len(company.securities) == 0
        company.add_security(
            model.Security(**{
            "name": "common stock",
            "secu_attributes": model.CommonShare(name="common stock")
        }))
        u.commit()
    with uow as u:
        company: model.Company = u.company.get(symbol=fake_company["symbol"])
        assert len(company.securities) == 1

def test_live_get_security_from_company_by_attributes(get_bootstrapped_dilution_db):
    db = get_bootstrapped_dilution_db
    uow = db.uow
    add_example_company(db)
    with uow as u:
        company: model.Company = u.company.get(symbol=fake_company["symbol"])
        assert len(company.securities) == 0
        company.add_security(
            model.Security(**{
            "name": "common stock",
            "secu_attributes": model.CommonShare(name="common stock")
        }))
        u.commit()
    with uow as u:
        company = u.company.get(symbol=fake_company["symbol"])
        security = company.get_security_by_attributes("common stock", {"name": "common stock"})
        assert security.name == "common stock"

def test_live_add_securityaccnoccurence(get_bootstrapped_dilution_db):
    db = get_bootstrapped_dilution_db
    uow = db.uow
    add_example_company(db)
    with uow as u:
        company: model.Company = u.company.get(symbol=fake_company["symbol"])
        assert len(company.securities) == 0
        company.add_security(
            model.Security(**{
            "name": "common stock",
            "secu_attributes": model.CommonShare(name="common stock")
        }))
        u.commit()
    db.create_filing_link(
        **fake_filing_link
    )
    with uow as u:
        company = u.company.get(symbol=fake_company["symbol"])
        security = company.get_security_by_attributes("common stock", {"name": "common stock"})
        occurence = model.SecurityAccnOccurence(security.id, fake_filing_link["accn"])
        u.session.add(occurence)
        u.commit()
    with uow as u:
        assert u.session.query(model.SecurityAccnOccurence).all()[0].accn == fake_filing_link["accn"]


def test_transient_model_object_requeried_in_subtransaction(get_bootstrapped_dilution_db):
    db = get_bootstrapped_dilution_db
    uow = db.uow
    sic = model.Sic(9000, "random sector", "random industry", "random divison")
    ft = model.FormType("S-3", "whatever")
    with uow as u:
        u.session.add(sic)
        u.session.add(ft)
        u.session.commit()
    
    company = model.Company(**fake_company)
    with uow as u:
        u.company.add(company)
        u.commit()
    
    
    try:
        with uow as uow1:
            shelf = model.ShelfRegistration(
                accn='000143774918017591',
                file_number='1',
                form_type='S-3',
                capacity=75000000.0,
                filing_date=datetime.date(2018, 9, 28),
                effect_date=None,
                last_update=None,
                expiry=None,
                total_amount_raised=None)
            company1 = uow1.company.get(fake_company["symbol"])
            uow1.session.expunge(company1)
            company1.add_shelf(shelf)
            
            # uow1.session.commit()
            with uow as uow2:
                from sqlalchemy import inspect
                insp = inspect(shelf)
                print([x.value for x  in insp.attrs])
                print(insp.detached, " detached")
                print(insp.persistent, " persistent")
                print(insp.identity, " identity")
                print(insp.transient, " transient")
                print(insp.pending, " pending")
                print(insp.dict, " dict")
                print(insp.mapper, " mapper")
                print(insp.object, " object")
                company = uow2.company.get(fake_company["symbol"])
                shelf = uow2.session.merge(shelf)
                company.add_shelf(shelf)
                uow2.company.add(company)
                uow2.commit()
    except Exception as e:
        raise e
    else:
        assert 1 == 1


class TestHandlers():
    def test_add_sic_(self,  get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        sic = model.Sic(9999, "unclassifiable_sector", "unclassifiable_industry", "unclassifiable_division")
        db.bus.handle(commands.AddSic(sic))
        sic = model.Sic(9999, "unclassifiable_sector", "unclassifiable_industry", "unclassifiable_division")
        db.bus.handle(commands.AddSic(sic))
        with db.uow as uow:
            result = uow.session.query(model.Sic).all()
            assert sic in result

    def test_add_company(self, get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        sic = model.Sic(9000, "random sector", "random industry", "random divison")
        uow = db.uow
        with uow as u:
            u.session.add(sic)
            u.session.commit()
        company = model.Company(**fake_company)
        cmd = commands.AddCompany(company)
        db.bus.handle(cmd)
        with db.uow as uow:
            received = uow.company.get(symbol=company.symbol)
            assert received.name == company.name
            assert received.cik == company.cik
            assert received.sic == company.sic
            assert received.symbol == company.symbol
    
    def test_add_securities(self, get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        add_example_company(db)
        security = model.Security(**{
            "name": "common stock",
            "secu_attributes": model.CommonShare(name="common stock")
        })
        cmd = commands.AddSecurities(
            fake_company["cik"],
            fake_company["symbol"],
            securities=[security])
        db.bus.handle(cmd)
        with db.uow as uow:
            company = uow.company.get(fake_company["symbol"])
            securities = company.securities
            assert len(securities) == 1
            for received in securities:
                assert received.name == security.name
    
    def test_add_shelf_registration(self, get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        add_example_company(db)
        with db.uow as uow:
            uow.session.add(model.FormType("S-3", "whatever"))
            uow.session.commit()
        shelf = model.ShelfRegistration(
                accn='000143774918017591',
                file_number='1',
                form_type='S-3',
                capacity=75000000.0,
                filing_date=datetime.date(2018, 9, 28),
                effect_date=None,
                last_update=None,
                expiry=None,
                total_amount_raised=None
            )
        db.bus.handle(commands.AddShelfRegistration(
            cik="0000000001",
            symbol="RANC",
            shelf_registration=shelf
        ))
        with db.uow as uow:
            company = uow.company.get(symbol="RANC")
            assert shelf in company.shelfs
    
    def test_add_resale_registration(self, get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        add_example_company(db)
        with db.uow as uow:
            uow.session.add(model.FormType("S-3", "whatever"))
            uow.session.commit()
        registration = model.ResaleRegistration(**fake_resale_registration)
        db.bus.handle(commands.AddResaleRegistration(
            cik=fake_company["cik"],
            symbol=fake_company["symbol"],
            resale_registration=registration
        ))
        with db.uow as uow:
            received = uow.session.query(model.ResaleRegistration).first()
            assert received.file_number == registration.file_number
            assert received.accn == registration.accn
    
    def test_add_shelf_offering(self, get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        add_example_company(db)
        with db.uow as uow:
            shelf_offering = model.ShelfOffering(**fake_shelf_offering)
        # create fake shelf offering dict
        # create command
        # handle through bus
        # assert we added offering


    def test_add_shelf_security_registration(self, get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        add_example_company(db)
        with db.uow as uow:
            shelf_registration = model.ShelfSecurityRegistration

    def test_add_form_type(self, get_bootstrapped_dilution_db):
        pass
    
    def test_effect_registration(self, get_bootstrapped_dilution_db):
        pass

    def test_add_outstanding_security_fact(self, get_bootstrapped_dilution_db):
        pass
    
    def test_add_security_accn_occurence(self, get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        uow = db.uow
        add_example_company(db)
        add_example_common_stock(db)
        db.create_filing_link(
            **fake_filing_link
        )
        cmd = commands.AddSecurityAccnOccurence(
            cik=fake_company["cik"],
            symbol=fake_company["symbol"],
            security_attributes={"name": fake_common_stock["name"]}
        )
        db.bus.handle(cmd)
        with uow as u:
            received = u.session.query(model.SecurityAccnOccurence).first()
            assert received.security_id == 1
            assert received.filing_accn == fake_filing_link["accn"]

    def test_add_filing_link_with_missing_form_type(self, get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        add_example_company(db)
        filing_link = model.FilingLink("https://anyrandomurl.com", "123456789123456789", "S-5", datetime.date(2022, 1, 1), "no descrption", "333-123123")
        filing_link2 = model.FilingLink("https://anyrandomurl2.com", "123456789123456780","S-6", datetime.date(2022, 1, 1), "no descrption", "333-123123")
        db.bus.handle(commands.AddFilingLinks("0000000001", "RANC", [filing_link, filing_link2]))
        with db.uow as uow:
            company = uow.company.get(symbol="RANC")
            filing_htmls = [x.filing_html for x in company.filing_links]
            for each in [filing_link, filing_link2]:
                assert each.filing_html in filing_htmls

    def test_add_effect_registration(self, get_bootstrapped_dilution_db):
        db = get_bootstrapped_dilution_db
        add_example_company(db)
        with db.uow as uow:
            uow.session.add(model.FormType("S-3", "whatever"))
            uow.session.commit()
        shelf = model.ShelfRegistration(
                accn='000143774918017591',
                file_number='1',
                form_type='S-3',
                capacity=75000000.0,
                filing_date=datetime.date(2018, 9, 28),
                effect_date=None,
                last_update=None,
                expiry=None,
                total_amount_raised=None
            )
        db.bus.handle(commands.AddShelfRegistration(
            cik="0000000001",
            symbol="RANC",
            shelf_registration=shelf
        ))
        effect = model.EffectRegistration(
            accn="123456789123456789",
            file_number="1",
            form_type="S-3",
            effective_date=datetime.date(2022, 1, 1)
        )
        db.bus.handle(commands.AddEffectRegistration(
            cik="0000000001",
            symbol="RANC",
            effect_registration=effect
        ))
        with db.uow as uow:
            company = uow.company.get(symbol="RANC")
            assert shelf in company.shelfs
            assert effect in company.effects
            modified_shelf = company.get_shelf(file_number="1")
            assert modified_shelf.effect_date == datetime.date(2022, 1, 1)


    









