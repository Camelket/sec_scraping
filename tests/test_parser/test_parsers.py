from pathlib import Path
import os
import pytest
from main.filings_base import Filing
from main.parser.filing_parsers import BaseFiling, filing_factory, HTMFilingParser, XMLFilingParser, ParserEFFECT
import datetime

from xml.etree import ElementTree

s3_rel_path = r"test_resources\filings\0001325879\S-3\000119312518218817\d439397ds3.htm"
s3_shelf = r"test_resources\filings\0001035976\S-3\000143774918017591\fncb20180927_s3.htm"
effect_xml = r"test_resources\filings\0001309082\EFFECT\999999999522002596\primary_doc.xml"
'''
needs test for:
    * each table classification
    *
'''

def _get_absolute_path(rel_path):
    return str(Path(__file__).parent.parent / rel_path)

def test_XMLFilingParser_get_doc():
    file_path = _get_absolute_path(effect_xml)
    parser = XMLFilingParser()
    assert isinstance(parser.get_doc(file_path), ElementTree.ElementTree) is True

def test_ParserEFFECT():
    path = _get_absolute_path(effect_xml)
    filing_info = {
        "path": path,
        "filing_date": None,
        "accession_number": Path(path).parents[0].name,
        "cik": Path(path).parents[2].name,
        "file_number": "333-147568",
        "form_type": "EFFECT",
        "extension": ".xml"
    }
    filing = filing_factory.create_filing(**filing_info)
    assert isinstance(filing, BaseFiling)
    assert filing.sections[0].content_dict == {
        'for_form': 'S-1',
        'effective_date': '2022-09-06',
        'file_number': '333-265715',
        'cik': '0001309082'
    }


# @pytest.mark.parser_splitting
def test_s3_splitting_by_toc_hrefs():
    s3_path = _get_absolute_path(s3_rel_path)
    parser = HTMFilingParser()
    doc = parser.get_doc(s3_path)
    sections = parser._split_by_table_of_contents_based_on_hrefs(parser.make_soup(doc))
    print(sections)
    #TODO: implement test for toc_href splitting
    assert 1 == 2


def test_s3_splitting_by_toc_headers():
    s3_path = _get_absolute_path(s3_rel_path)
    parser = HTMFilingParser()
    doc = parser.get_doc(s3_path)
    sections = parser._split_by_table_of_content_based_on_headers(parser.make_soup(doc))
    #TODO: implement test for toc_header splitting

def test_s3_filing_creation2():
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
    sections = [s.title for s in filing.sections]
    front_page = filing.get_section("front page")
    assert front_page is not None
    for each in ['front page', 'cover page 0', 'toc 0', 'about this prospectus', 'cautionary note regarding forward-looking statements', 'the company', 'the offering', 'use of proceeds', 'risk factors', 'risk factors', 'ratio of earnings to fixed charges', 'ratio of earnings to fixed charges (1) :', 'use of proceeds', 'description of securities', 'description of capital stock', 'description of senior and subordinated debt securities', 'description of depositary shares', 'description of depositary shares', 'description of purchase contracts', 'description of units', 'description of warrants', 'description of rights', 'plan of distribution', 'legal matters', 'experts']:
        assert each in sections
    reg = front_page.get_tables("registration_table", "extracted")
    assert reg is not None


def test_s3_filing_creation():
    s3_path = _get_absolute_path(s3_rel_path)
    s3_path_Path = Path(s3_path)
    info = {
            "path": s3_path,
            "filing_date": None,
            "accession_number": s3_path_Path.parents[0].name,
            "cik": s3_path_Path.parents[2].name,
            "file_number": None,
            "form_type": "S-3",
            "extension": ".htm"
            }
    required_sections = [
        "prospectus summary",
        "risk factors", 
        "cautionary note regarding forward-looking information",
        "use of proceeds",
        "price range of common stock",
        "dividend policy",
        "determination of offering price",
        "dilution",
        "plan of distribution",
        "description of securities",
        "material u.s. federal tax considerations for non-u.s. holders of common stock",
        "legal matters",
        "where you can find more information",
        "incorporation of certain documents by reference"
        ]
    filing = filing_factory.create_filing(**info)
    filing_titles = set(s.title for s in filing.sections)
    print(filing_titles)
    for rsec in required_sections:
        assert rsec in filing_titles
        assert len(filing.get_section(rsec).content) > 10


# def test_get_for_form_xml():
#     parser = ParserEFFECT()
    
    # need to run tests on the splitting by toc table first to detect why we cant find close_to_toc

    
        