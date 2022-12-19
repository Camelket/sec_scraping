import main.parser.extractors as extractors
import pytest
import spacy
import pandas as pd
from pathlib import Path
import datetime
import re
from main.parser.parsers import filing_factory
from main.domain import model, commands
from main.services import unit_of_work, messagebus

s3_shelf = r"test_resources\filings\0001035976\S-3\000143774918017591\fncb20180927_s3.htm"
s3_resale = r"test_resources\filings\0000908311\S-3\000110465919045626\a19-16974_2s3.htm"
s3_ATM = r"test_resources\filings\0000831547\S-3\000083154720000018\cleans-3.htm"
s3_resale_with_warrants = r"test_resources\filings\0001453593\S-3\000149315221008120\forms-3.htm"

effect_filing_path = r"test_resources\filings\0001309082\EFFECT\999999999522002596\primary_doc.xml"

def _get_absolute_path(rel_path):
    return str(Path(__file__).parent.parent / rel_path)

@pytest.fixture
def get_fake_messagebus():
    mb = messagebus.MessageBus(unit_of_work.FakeCompanyUnitOfWork, dict())
    yield mb
    del mb


@pytest.fixture
def get_effect_extractor():
    extractor = extractors.XMLEFFECTExtractor()
    yield extractor
    del extractor

@pytest.fixture
def get_base_extractor():
    base_extractor = extractors.BaseHTMExtractor()
    yield base_extractor
    del base_extractor

@pytest.fixture
def get_s3_extractor():
    base_extractor = extractors.HTMS3Extractor()
    yield base_extractor
    del base_extractor

def get_fake_company():
    return model.Company(
        **{   
            "cik": "0000000001",
            "sic": "9000",
            "symbol": "RAND",
            "name": "Rand Inc.",
            "description_": "Solely a test company meant for usage with pytest"
        }
    )

@pytest.fixture
def get_effect_filing():
    path = _get_absolute_path(effect_filing_path)
    info = {
            "path": path,
            "filing_date": datetime.date(2018, 9, 28),
            "accession_number": Path(path).parents[0].name,
            "cik": "0000000001",
            "file_number": "1",
            "form_type": "EFFECT",
            "extension": ".xml"
            }
    filing = filing_factory.create_filing(**info)
    return filing


@pytest.fixture
def get_filing_s3_shelf():
    path = _get_absolute_path(s3_shelf)
    info = {
            "path": path,
            "filing_date": datetime.date(2018, 9, 28),
            "accession_number": Path(path).parents[0].name,
            "cik": Path(path).parents[2].name,
            "file_number": "1",
            "form_type": "S-3",
            "extension": ".htm"
            }
    filing = filing_factory.create_filing(**info)
    return filing

@pytest.fixture
def get_filing_s3_resale_with_warrants():
    path = _get_absolute_path(s3_resale_with_warrants)
    info = {
            "path": path,
            "filing_date": datetime.date(2018, 9, 28),
            "accession_number": Path(path).parents[0].name,
            "cik": Path(path).parents[2].name,
            "file_number": "1",
            "form_type": "S-3",
            "extension": ".htm"
            }
    filing = filing_factory.create_filing(**info)
    return filing

@pytest.fixture
def get_filing_s3_resale():
    path = _get_absolute_path(s3_resale)
    info = {
            "path": path,
            "filing_date": datetime.date(2018, 9, 28),
            "accession_number": Path(path).parents[0].name,
            "cik": Path(path).parents[2].name,
            "file_number": "2",
            "form_type": "S-3",
            "extension": ".htm"
            }
    filing = filing_factory.create_filing(**info)
    return filing

@pytest.fixture
def get_filing_s3_ATM():
    path = _get_absolute_path(s3_ATM)
    info = {
            "path": path,
            "filing_date": datetime.date(2018, 9, 28),
            "accession_number": Path(path).parents[0].name,
            "cik": Path(path).parents[2].name,
            "file_number": "1",
            "form_type": "S-3",
            "extension": ".htm"
            }
    filing = filing_factory.create_filing(**info)[1]
    return filing

# def test_s3_ATM_creation(get_s3_extractor, get_filing_s3_ATM):
#     extractor = get_s3_extractor
#     filing = get_filing_s3_ATM
#     print([s.title for s in filing.sections])
#     cover_page_doc = extractor.doc_from_section(filing.get_section(re.compile("cover page")))
#     offering_amount = extractor.extract_aggregate_offering_amount(cover_page_doc)
#     print(offering_amount, type(offering_amount))
#     assert 1 == 2

def test_s3_classification(get_s3_extractor, get_filing_s3_shelf, get_filing_s3_resale, get_filing_s3_ATM):
    extractor: extractors.HTMS3Extractor = get_s3_extractor
    shelf_filing = get_filing_s3_shelf
    assert "shelf" in extractor.classify_s3(shelf_filing)["classifications"]
    resale_filing = get_filing_s3_resale
    assert "resale" in extractor.classify_s3(resale_filing)["classifications"]
    atm_filing = get_filing_s3_ATM
    assert "ATM" in extractor.classify_s3(atm_filing)["classifications"]


def test_security_extraction_s3_shelf(get_s3_extractor, get_fake_messagebus, get_filing_s3_shelf):
    extractor: extractors.HTMS3Extractor = get_s3_extractor
    bus = get_fake_messagebus
    filing = get_filing_s3_shelf
    company = get_fake_company()
    cover_page = filing.get_section(re.compile("cover page"))
    cover_page_doc = extractor.doc_from_section(cover_page)
    # add return to default pipeline
    securities = extractor.extract_securities(filing, company, bus, cover_page_doc)
    expected_securities = [
        model.Security(model.CommonShare(name="common stock", par_value=0.001)),
        model.Security(model.PreferredShare(name="preferred stock", par_value=0.001, convertible_feature=None)),
        ]
    assert securities == expected_securities

def test_security_extraction_s3_warrant(get_s3_extractor, get_fake_messagebus, get_filing_s3_resale_with_warrants):
    extractor: extractors.HTMS3Extractor = get_s3_extractor
    bus = get_fake_messagebus
    filing = get_filing_s3_resale_with_warrants
    company = get_fake_company()
    filing_text = filing.get_text_only()
    filing_text_doc = extractor.spacy_text_search.nlp(filing_text)
    securities = extractor.extract_securities(filing, company, bus, filing_text_doc)
    print("securities extracted: ", securities)
    assert all([isinstance(secu, model.Security) for secu in securities])
    expected_security_names = [
        "preferred stock",
        "investor warrant",
        "placement agent warrant",
        "warrant",
        "common stock",
        ]
    received_security_names = [secu.name for secu in securities]
    assert len(expected_security_names) == len(received_security_names)
    for each in expected_security_names:
        assert each in received_security_names
    
def test_security_extraction_s3_ATM(get_s3_extractor, get_fake_messagebus, get_filing_s3_ATM):
    extractor: extractors.HTMS3Extractor = get_s3_extractor
    bus = get_fake_messagebus
    filing = get_filing_s3_ATM
    company = get_fake_company()
    filing_text = filing.get_text_only()
    filing_text_doc = extractor.spacy_text_search.nlp(filing_text)
    securities = extractor.extract_securities(filing, company, bus, filing_text_doc)
    print("securities extracted: ", securities)
    assert all([isinstance(secu, model.Security) for secu in securities])
    received_security_names = [secu.name for secu in securities]
    expected_security_names = [
        "common stock",
        "preferred stock",
        ]
    received_security_names = [secu.name for secu in securities]
    assert len(expected_security_names) == len(received_security_names)
    for each in expected_security_names:
        assert each in received_security_names


def test_extract_shelf_s3(get_s3_extractor, get_fake_messagebus, get_filing_s3_shelf):
    extractor: extractors.HTMS3Extractor = get_s3_extractor
    bus: messagebus.MessageBus = get_fake_messagebus
    filing = get_filing_s3_shelf
    company = get_fake_company()
    company = extractor.extract_form_values(filing, company, bus)
    command_history = bus.collect_command_history()
    print("command_history:", command_history)
    expected_shelf = model.ShelfRegistration("000143774918017591", "1", "S-3", 75000000, datetime.date(2018, 9, 28))
    received_add_shelf_commands = [cmd.shelf_registration if isinstance(cmd, commands.AddShelfRegistration) else None for cmd in command_history]
    assert expected_shelf in received_add_shelf_commands

def test_extract_resale_s3(get_s3_extractor, get_fake_messagebus, get_filing_s3_resale):
    extractor: extractors.HTMS3Extractor = get_s3_extractor
    bus = get_fake_messagebus
    filing = get_filing_s3_resale
    company = get_fake_company()
    company = extractor.extract_form_values(filing, company, bus)
    command_history = bus.collect_command_history()
    print("command_history:", command_history)
    received_resale_registrations_from_commands = [cmd.resale_registration if isinstance(cmd, commands.AddResaleRegistration) else None for cmd in command_history]
    expected_resale = model.ResaleRegistration(**{
        "accn": filing.accession_number,
        "form_type": filing.form_type,
        "file_number": filing.file_number,
        "filing_date": filing.filing_date
    })
    assert expected_resale in received_resale_registrations_from_commands

def test_extract_ATM_s3(get_s3_extractor, get_fake_messagebus, get_filing_s3_ATM):
    extractor: extractors.HTMS3Extractor = get_s3_extractor
    bus = get_fake_messagebus
    company = get_fake_company()
    # TODO: import fake data from a common place across all tests
    company.add_shelf(model.ShelfRegistration("000143774918017591", "1", "S-3", 75000000, datetime.date(2018, 9, 28)))
    atm_filing = get_filing_s3_ATM
    company = extractor.extract_form_values(atm_filing, company, bus)
    command_history = bus.collect_command_history()
    received_shelf_offering_from_commands = [cmd.shelf_offering if isinstance(cmd, commands.AddShelfOffering) else None for cmd in command_history]
    expected_offering = model.ShelfOffering(**{
        "offering_type": "ATM",
        "accn": atm_filing.accession_number,
        "anticipated_offering_amount": 75000000,
        "commencment_date": datetime.date(2018, 9, 28),
        "end_date": datetime.date(2021, 9, 27)
    })
    assert expected_offering in received_shelf_offering_from_commands
    cover_page_doc = extractor.doc_from_section(atm_filing.get_section(re.compile("cover page")))
    aggregate_offering = extractor.extract_aggregate_offering_amount(cover_page_doc)
    expected_aggregate_offering = {'SECU': ['common stock'], 'amount': 75000000}
    assert aggregate_offering == expected_aggregate_offering
    

def test_extract_form_values_effect(get_fake_messagebus, get_effect_extractor, get_effect_filing):
    extractor = get_effect_extractor
    filing = get_effect_filing
    bus = get_fake_messagebus
    company = get_fake_company()
    extractor.extract_form_values(filing, company, bus)
    command_history = bus.collect_command_history()
    received_effect_registrations_from_commands = [cmd.effect_registration if isinstance(cmd, commands.AddEffectRegistration) else None for cmd in command_history]
    expected_effect_registration = model.EffectRegistration(accn='999999999522002596', file_number='333-265715', form_type="S-1", effective_date='2022-09-06')
    assert expected_effect_registration in received_effect_registrations_from_commands

        


    
    

    