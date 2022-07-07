import pytest
from dilution_db import DilutionDB
from test_database import get_session
import datetime

@pytest.fixture
def company_population_data():
    data = {
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
            "cik": "0000000001",
            "sic": "9000",
            "symbol": "RAND",
            "name_": "Rand Inc.",
            "description_": "Solely a test company meant for usage with pytest"
        },
        "shelf_registration": {
            "company_id": 1,
            "accn": "0000123456789",
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
            "company_id": 1,
            "security_name": "common stock",
            "security_type": "CommonShares",
            "underlying_security_id": None,
            "security_attributes": 
        }
    }

