***
# ABANDONED

## sec_scraping
The aim of this repository is to be able to easily parse SEC-filings and create a model that can be used with a database. Created for dilutionscout.com.

### how to setup dilution_db:
0. Fork/clone repository
1. Create an empty database in postgresql
2. make a copy of the public.env file or add it to gitignore
3. Fill the .env file with the database info
4. Add paths for files to the .env
5. Add the polygon API key to the .env. Get a free polygon api key if you havnt got one. [polygon pricing page](https://polygon.io/pricing)
6. specify what companies and what sec forms to track in main/configs.py (class AppConfig)
7. do (in a file above the main directory):

if you want to parse the filings
```python
    from dilution_db import DilutionDB
    from main.configs import FactoryConfig, GlobalConfig
    from boot import bootstrap_dilution_db
    
    config = FactoryConfig(GlobalConfig(ENV_STATE="prod").ENV_STATE)()

    db = bootstrap_dilution_db(
        start_orm=True,
        config=config
    )
    db.inital_setup()
    # this will download at least 20GB of bulk data, the base info for the
    # company and all filings of the specified forms,
    # and will try to parse the filings and write to the database 
```

if you only want the base and xbrl companyfacts data (cashburn, outstanding common shares)
```python
    from dilution_db import DilutionDB
    from main.configs import FactoryConfig, GlobalConfig
    from boot import bootstrap_dilution_db
    
    config = FactoryConfig(GlobalConfig(ENV_STATE="prod").ENV_STATE)()

    db = bootstrap_dilution_db(
        start_orm=True,
        config=config
    )

    db.util.inital_company_setup(
            config.DOWNLOADER_ROOT_PATH,
            config.POLYGON_OVERVIEW_FILES_PATH,
            config.POLYGON_API_KEY,
            config.APP_CONFIG.TRACKED_FORMS,
            config.APP_CONFIG.TRACKED_TICKERS,
            after="2010-1-1",
            before=None
        )

```


#### accessing company data through the model
lets assume we have some data relating to "AAPL" already in the database.
Retrieving a company object containing the data can be done like so:

```python
    from dilution_db import DilutionDB
    from main.configs import FactoryConfig, GlobalConfig
    from boot import bootstrap_dilution_db
    
    config = FactoryConfig(GlobalConfig(ENV_STATE="prod").ENV_STATE)()

    db = bootstrap_dilution_db(
        start_orm=True,
        config=config
    )
    
    with db.uow as uow:
        company = uow.company.get(symbol="AAPL")
```
### Parsers
using parser directly:
```python
from main.parser import parsers

filing_path = r"absolut_path/to/filing.htm"
parser = parsers.Parser8K()
filing_sections = parser.parse(filing_path)
```


using parser through FilingFactory
```python

from main.parser.parsers import Parsers8K, ParserEFFECT, BaseHTMFiling, BaseFiling, FilingFactory, ParserFactory

# register parser for form type and file extension
parser_factory_default = [
    (".htm", "8-K", Parser8K),
    (".xml", "EFFECT", ParserEFFECT)

]
parser_factory = ParserFactory(defaults=parser_factory_default)

# register object which handles parsing
filing_factory_defaults = [
    ("8-K", ".htm", BaseHTMFiling),
    ("EFFECT", ".xml", BaseFiling)
]
filing_factory = FilingFactory(defaults=filing_factory_defaults)

filing_path_8k = r"absolut_path/to/filing_8k.htm"

parsed_filing = filing_factory.create_filing("8-K", ".htm", path=filing_path_8k)

```

### Extractors
To use an extractor independent of a database:
```python

from main.domain.model import Company
from main.services.messagebus import MessageBus
from main.services.unit_of_work import FakeCompanyUnitOfWork
from main.parser import extractors
from main.parser import parsers

from pathlib import Path

# create fake messagebus
mbus = MessageBus(FakeCompanyUnitOfWork, dict())

# create empty company to hold extracted values
company = Company(
    name="some_company_name",
    cik="0000000000",
    sic=9999,
    symbol="ANY"
    )

filing_path = r"path/cik/form_type/accession_number/filing_s3.htm"
filing = parsers.filing_factory(
        path= path,
        filing_date= datetime.date(2018, 9, 28),
        accession_number= Path(path).parents[0].name,
        cik= Path(path).parents[2].name,
        file_number= None,
        form_type= "S-3",
        extension= ".htm"
)
extractor = extractors.HTMS3Extractor()
extractor.extract_form_values(filing, company, mbus)
# this way of accessing what happened during extraction 
# will likely change in the future and be made available 
# as some kind of result object 
commands_issued_during_extraction = mbus.collect_command_history()
```

### filing_nlp
Used to get the financial instruments and their related attributes in unstructured text with spaCy.
can be used independently:
```python
from main.parser.filing_nlp import SpacyFilingTextSearch

search = SpacyFilingTextSearch

some_str = "This is a text about 500000 shares of common stock (The 'Shares')."

doc = search.nlp(some_str)

# this will mark:
#   'common stock' as SECU entity
#   '500000" as SECUQUANTITY enitity
#   Will determine 'Shares' as an alias and 
#   therefor a SECU entity aswell, with a relation 
#   to the 'common stock' SECU entity. 
#   This relation will be saved in doc._.single_secu_alias_tuples
```
further examples WIP

### XBRL Fact search
if you want to extract XBRL facts from sec companyfacts files check the data_aggregation folder
    1. use pysec-downloader package to retrieve companyfacts or run bulk_files.py to retrieve
        companyfacts and submissions files (around 25GB of files)
    2. use the _get_fact_data function from fact_extractor.py to get a specific fact/
        or facts matching a regex pattern


---


***


    




