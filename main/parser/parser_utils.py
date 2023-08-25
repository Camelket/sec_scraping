from pathlib import Path
from main.parser.filing_parsers import create_htm_filing
import re

def extract_sections_from_filings(root_path: str, rglob_term: str, section_re_term: re.Pattern, save_path: str, section_delimiter: str="+++:::"):
    '''extract all sections matching section_search_term and save to save_path'''
    fps = list(Path(root_path).rglob(rglob_term))
    failed = []
    for idx, fp in enumerate(fps):
        if idx%10 == 0:
            print(f"{idx}|{len(fps)}")
        info = {
            "form_type":"S-3",
            "extension": ".htm",
            "path": fp,
            "filing_date": "-",
            "accession_number": "-",
            "cik": "-",
            "file_number": "-",
        }
        try:
            filings = create_htm_filing(**info)
        except AttributeError as e:
            failed.append((fp, e))
        else:
            if isinstance(filings, list):
                pass
            else:
                filings = [filings]
            for filing in filings:
                sections = filing.get_sections(identifier=section_re_term)
                if sections:
                    for section in sections:
                        with open(save_path, "a", encoding="utf-8") as f:
                            text = section.text_only
                            f.write(section_delimiter)
                            f.write(text)
    for each in failed:
        print(each)