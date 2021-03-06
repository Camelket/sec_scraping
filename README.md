***

## sec_scraping
aim is to download various filings and parse them to get past dilutive behavior and possible future dilution,
and create a database for dilutionscout.com in the process (dilution_db).

if you just want to parse filings check the parser folder (very incomplete)
if you want to extract facts from sec companyfacts files check the data_aggregation folder
    1. specify DOWNLOADER_ROOT_PATH in the configs.py
    2. use pysec-downloader package to retrieve companyfacts or run bulk_files.py to retrieve
        companyfacts and submissions files (around 25GB of files)
    3. use the _get_fact_data function from fact_extractor.py to get a specific fact/
        or facts matching a regex pattern

---

## how to setup dilution_db:
0. fork and clone repository
1. create dilution_db in postgres
2. add info (database info (like host, password, ect.), 
    add the folders where you want the data to be stored) 
    to main/configuration/public.env just fill the .env file basically.
    get a free polygon api key if you havnt got one. [polygon pricing page](https://polygon.io/pricing)
3. specify what companies to track and what sec forms to track in main/configs.py (class AppConfig)
4. do (in a file above the main directory):

```python
    from dilution_db import DilutionDB
    from main.configs import cnf
    from db_updater import inital_population

    db = DilutionDB
    db._create_tables()
    db.create_sics()
    db.create_form_types()
    inital_population(
        db, cnf.DOWNLOADER_ROOT_PATH,
        cnf.POLYGON_OVERVIEW_FILES_PATH,
        cnf.POLYGON_API_KEY,
        cnf.APP_CONFIG.TRACKED_TICKERS
        )
    # you can end up with partially populated companies.


    # if you want all the sec filings after a certain date do something like:
        # *replacing the date and number of filings you want, this will take a while ...
        # could take multiple days if you have a lot of filing types and tickers set,
        # you will need to implement restarting and continuing from failed ticker
    for ticker in cnf.APP_CONFIG.TRACKED_TICKERS:
        get_filing_set(Downloader(dl_root_path), ticker, cnf.APP_CONFIG.TRACKED_FORMS, "2018-01-01", number_of_filings=200)
```

***


    




