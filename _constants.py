# check here https://help.edgar-online.com/edgar/formtypes.asp
# and here https://www.sec.gov/forms 

EDGAR_BASE_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"
FORM_TYPES_INFO = {
    "1": {"category": "unspecified"},
    "1-A": {"category": "unspecified"},
    "1-A POS": {"category": "unspecified"},
    "1-A-W": {"category": "unspecified"},
    "1-E": {"category": "unspecified"},
    "1-E AD": {"category": "unspecified"},
    "1-K": {"category": "Financials"},
    "1-SA": {"category": "Financials"},
    "1-U": {"category": "unspecified"},
    "1-Z": {"category": "unspecified"},
    "1-Z-W": {"category": "unspecified"},
    "10-12B": {"category": "unspecified"},
    "10-12G": {"category": "unspecified"},
    "10-D": {"category": "unspecified"},
    "10-K": {"category": "Financials", "description": "Annual report"},
    "10-KT": {"category": "unspecified"},
    "10-Q": {"category": "Financials", "description": "Quarterly report"},
    "10-Q/A": {"category": "Financials", "description": "Amendment to Quarterly report"},
    "10-QT": {"category": "unspecified"},
    "11-K": {"category": "unspecified"},
    "11-KT": {"category": "unspecified"},
    "13F-HR": {"category": "Ownership"},
    "13F-NT": {"category": "Ownership"},
    "13FCONP": {"category": "Ownership"},
    "144": {"category": "Others"},
    "15-12B": {"category": "unspecified"},
    "15-12G": {"category": "unspecified"},
    "15-15D": {"category": "unspecified"},
    "15F-12B": {"category": "Prospectus"},
    "15F-12G": {"category": "Prospectus"},
    "15F-15D": {"category": "Prospectus"},
    "18-12B": {"category": "unspecified"},
    "18-K": {"category": "Financials"},
    "19B-4E": {"category": "unspecified"},
    "2-A": {"category": "unspecified"},
    "2-AF": {"category": "unspecified"},
    "2-E": {"category": "unspecified"},
    "20-F": {"category": "unspecified"},
    "20FR12B": {"category": "unspecified"},
    "20FR12G": {"category": "unspecified"},
    "24F-2NT": {"category": "unspecified"},
    "25": {"category": "Others"},
    "25-NSE": {"category": "unspecified"},
    "253G1": {"category": "unspecified"},
    "253G2": {"category": "unspecified"},
    "253G3": {"category": "unspecified"},
    "253G4": {"category": "unspecified"},
    "3": {"category": "Ownership"},
    "3/A": {"category": "Ownership"},
    "305B2": {"category": "unspecified"},
    "34-12H": {"category": "unspecified"},
    "4": {"category": "Ownership"},
    "40-17F1": {"category": "unspecified"},
    "40-17F2": {"category": "unspecified"},
    "40-17G": {"category": "unspecified"},
    "40-17GCS": {"category": "unspecified"},
    "40-202A": {"category": "unspecified"},
    "40-203A": {"category": "unspecified"},
    "40-206A": {"category": "unspecified"},
    "40-24B2": {"category": "unspecified"},
    "40-33": {"category": "unspecified"},
    "40-6B": {"category": "unspecified"},
    "40-8B25": {"category": "unspecified"},
    "40-8F-2": {"category": "unspecified"},
    "40-APP": {"category": "unspecified"},
    "40-F": {"category": "Financials"},
    "40-OIP": {"category": "unspecified"},
    "40FR12B": {"category": "unspecified"},
    "40FR12G": {"category": "unspecified"},
    "424A": {"category": "unspecified"},
    "424B1": {"category": "Prospectus"},
    "424B2": {"category": "Prospectus"},
    "424B3": {"category": "Prospectus"},
    "424B4": {"category": "Prospectus"},
    "424B5": {"category": "Prospectus"},
    "424B7": {"category": "Prospectus"},
    "424B8": {"category": "Prospectus"},
    "424H": {"category": "unspecified"},
    "425": {"category": "unspecified"},
    "485APOS": {"category": "unspecified"},
    "485BPOS": {"category": "unspecified"},
    "485BXT": {"category": "unspecified"},
    "486APOS": {"category": "unspecified"},
    "486BPOS": {"category": "unspecified"},
    "486BXT": {"category": "unspecified"},
    "487": {"category": "unspecified"},
    "497": {"category": "unspecified"},
    "497AD": {"category": "unspecified"},
    "497H2": {"category": "unspecified"},
    "497J": {"category": "unspecified"},
    "497K": {"category": "unspecified"},
    "5": {"category": "Ownership"},
    "6-K": {"category": "Disclosure", "description": "Disclosure of Foreign Issuer"},
    "6B NTC": {"category": "unspecified"},
    "6B ORDR": {"category": "unspecified"},
    "8-A12B": {"category": "unspecified"},
    "8-A12G": {"category": "unspecified"},
    "8-K": {"category": "Disclosure"},
    "8-K/A": {"category": "Disclosure"},
    "8-K12B": {"category": "unspecified"},
    "8-K12G3": {"category": "unspecified"},
    "8-K15D5": {"category": "unspecified"},
    "8-M": {"category": "unspecified"},
    "8F-2 NTC": {"category": "unspecified"},
    "8F-2 ORDR": {"category": "unspecified"},
    "9-M": {"category": "unspecified"},
    "ABS-15G": {"category": "unspecified"},
    "ABS-EE": {"category": "unspecified"},
    "ADN-MTL": {"category": "unspecified"},
    "ADV-E": {"category": "unspecified"},
    "ADV-H-C": {"category": "unspecified"},
    "ADV-H-T": {"category": "unspecified"},
    "ADV-NR": {"category": "unspecified"},
    "ANNLRPT": {"category": "unspecified"},
    "APP NTC": {"category": "unspecified"},
    "APP ORDR": {"category": "unspecified"},
    "APP WD": {"category": "unspecified"},
    "APP WDG": {"category": "unspecified"},
    "ARS": {"category": "unspecified"},
    "ATS-N": {"category": "unspecified"},
    "ATS-N-C": {"category": "unspecified"},
    "ATS-N/UA": {"category": "unspecified"},
    "AW": {"category": "unspecified"},
    "AW WD": {"category": "unspecified"},
    "C": {"category": "unspecified"},
    "C-AR": {"category": "unspecified"},
    "C-AR-W": {"category": "unspecified"},
    "C-TR": {"category": "unspecified"},
    "C-TR-W": {"category": "unspecified"},
    "C-U": {"category": "unspecified"},
    "C-U-W": {"category": "unspecified"},
    "C-W": {"category": "unspecified"},
    "CB": {"category": "unspecified"},
    "CERT": {"category": "unspecified"},
    "CERTARCA": {"category": "unspecified"},
    "CERTBATS": {"category": "unspecified"},
    "CERTCBO": {"category": "unspecified"},
    "CERTNAS": {"category": "unspecified"},
    "CERTNYS": {"category": "unspecified"},
    "CERTPAC": {"category": "unspecified"},
    "CFPORTAL": {"category": "unspecified"},
    "CRS": {"category": "Others", "description": "Customer Relationship Report"},
    "CFPORTAL-W": {"category": "unspecified"},
    "CORRESP": {"category": "Others", "description": "SEC correspondence letter"},
    "CT ORDER": {"category": "unspecified"},
    "D": {"category": "Prospectus", "description": "Private securities offering"},
    "DEF 14A": {"category": "Proxy"},
    "DEFA14A": {"category": "Proxy"},
    "DEFA14C": {"category": "Proxy"},
    "DEFC14A": {"category": "Proxy"},
    "DEFC14C": {"category": "Proxy"},
    "DEFM14A": {"category": "Proxy"},
    "DEF 14C": {"category": "Proxy"},
    "DEFM14C": {"category": "Proxy"},
    "DEFN14A": {"category": "Proxy"},
    "DEFR14A": {"category": "Proxy"},
    "DEFR14C": {"category": "Proxy"},
    "DEL AM": {"category": "unspecified"},
    "DFAN14A": {"category": "unspecified"},
    "DFRN14A": {"category": "unspecified"},
    "DOS": {"category": "unspecified"},
    "DOSLTR": {"category": "unspecified"},
    "DRS": {"category": "unspecified"},
    "DRSLTR": {"category": "unspecified"},
    "DSTRBRPT": {"category": "unspecified"},
    "EFFECT": {"category": "Prospectus"},
    "F-1": {"category": "Prospectus"},
    "F-10": {"category": "Prospectus"},
    "F-10EF": {"category": "unspecified"},
    "F-10POS": {"category": "Prospectus"},
    "F-1MEF": {"category": "Prospectus"},
    "F-3": {"category": "Prospectus"},
    "F-3ASR": {"category": "Prospectus", "description": "Automatic Shelf Registration"},
    "F-3D": {"category": "unspecified"},
    "F-3DPOS": {"category": "unspecified"},
    "F-3MEF": {"category": "unspecified"},
    "F-4": {"category": "Prospectus"},
    "F-4 POS": {"category": "Prospectus"},
    "F-4MEF": {"category": "unspecified"},
    "F-6": {"category": "Prospectus"},
    "F-6 POS": {"category": "Prospectus"},
    "F-6EF": {"category": "Prospectus"},
    "F-7": {"category": "Prospectus"},
    "F-7 POS": {"category": "Prospectus"},
    "F-8": {"category": "Prospectus"},
    "F-8 POS": {"category": "Prospectus"},
    "F-80": {"category": "Prospectus"},
    "F-80POS": {"category": "Prospectus"},
    "F-9": {"category": "unspecified"},
    "F-9 POS": {"category": "unspecified"},
    "F-N": {"category": "unspecified"},
    "F-X": {"category": "unspecified"},
    "FOCUSN": {"category": "unspecified"},
    "FWP": {"category": "unspecified"},
    "G-405": {"category": "unspecified"},
    "G-405N": {"category": "unspecified"},
    "G-FIN": {"category": "unspecified"},
    "G-FINW": {"category": "unspecified"},
    "IRANNOTICE": {"category": "unspecified"},
    "MA": {"category": "unspecified"},
    "MA-A": {"category": "unspecified"},
    "MA-I": {"category": "unspecified"},
    "MA-W": {"category": "unspecified"},
    "MSD": {"category": "unspecified"},
    "MSDCO": {"category": "unspecified"},
    "MSDW": {"category": "unspecified"},
    "N-1": {"category": "unspecified"},
    "N-14": {"category": "Prospectus"},
    "N-14 8C": {"category": "Prospectus"},
    "N-14MEF": {"category": "Prospectus"},
    "N-18F1": {"category": "unspecified"},
    "N-1A": {"category": "unspecified"},
    "N-2": {"category": "unspecified"},
    "N-23C-2": {"category": "unspecified"},
    "N-23C3A": {"category": "unspecified"},
    "N-23C3B": {"category": "unspecified"},
    "N-23C3C": {"category": "unspecified"},
    "N-2MEF": {"category": "unspecified"},
    "N-30B-2": {"category": "unspecified"},
    "N-30D": {"category": "unspecified"},
    "N-4": {"category": "unspecified"},
    "N-5": {"category": "unspecified"},
    "N-54A": {"category": "unspecified"},
    "N-54C": {"category": "unspecified"},
    "N-6": {"category": "unspecified"},
    "N-6F": {"category": "unspecified"},
    "N-8A": {"category": "unspecified"},
    "N-8B-2": {"category": "unspecified"},
    "N-8F": {"category": "unspecified"},
    "N-8F NTC": {"category": "unspecified"},
    "N-8F ORDR": {"category": "unspecified"},
    "N-CEN": {"category": "unspecified"},
    "N-CR": {"category": "unspecified"},
    "N-CSR": {"category": "unspecified"},
    "N-CSRS": {"category": "unspecified"},
    "N-MFP": {"category": "unspecified"},
    "N-MFP1": {"category": "unspecified"},
    "N-MFP2": {"category": "unspecified"},
    "N-PX": {"category": "unspecified"},
    "N-Q": {"category": "unspecified"},
    "NO ACT": {"category": "unspecified"},
    "NPORT-EX": {"category": "unspecified"},
    "NPORT-NP": {"category": "unspecified"},
    "NPORT-P": {"category": "unspecified"},
    "NRSRO-CE": {"category": "unspecified"},
    "NRSRO-UPD": {"category": "unspecified"},
    "NSAR-A": {"category": "unspecified"},
    "NSAR-AT": {"category": "unspecified"},
    "NSAR-B": {"category": "unspecified"},
    "NSAR-BT": {"category": "unspecified"},
    "NSAR-U": {"category": "unspecified"},
    "NT 10-D": {"category": "unspecified"},
    "NT 10-K": {"category": "Financials", "description": "Cant keep filing Deadline for 10-K"},
    "NT 10-Q": {"category": "Financials", "description": "Cant keep filing Deadline for 10-Q"},
    "NT 11-K": {"category": "unspecified"},
    "NT 20-F": {"category": "Financials", "description": "Cant keep filing Deadline for 20-F"},
    "NT N-CEN": {"category": "unspecified"},
    "NT N-MFP": {"category": "unspecified"},
    "NT N-MFP1": {"category": "unspecified"},
    "NT N-MFP2": {"category": "unspecified"},
    "NT NPORT-EX": {"category": "unspecified"},
    "NT NPORT-P": {"category": "unspecified"},
    "NT-NCEN": {"category": "unspecified"},
    "NT-NCSR": {"category": "unspecified"},
    "NT-NSAR": {"category": "unspecified"},
    "NTFNCEN": {"category": "unspecified"},
    "NTFNCSR": {"category": "unspecified"},
    "NTFNSAR": {"category": "unspecified"},
    "NTN 10D": {"category": "unspecified"},
    "NTN 10K": {"category": "unspecified"},
    "NTN 10Q": {"category": "unspecified"},
    "NTN 20F": {"category": "unspecified"},
    "OIP NTC": {"category": "unspecified"},
    "OIP ORDR": {"category": "unspecified"},
    "POS 8C": {"category": "unspecified"},
    "POS AM": {"category": "Prospectus", "description": "Post EFFECT amendment to registration form"},
    "POS AMI": {"category": "unspecified"},
    "POS EX": {"category": "unspecified"},
    "POS462B": {"category": "unspecified"},
    "POS462C": {"category": "unspecified"},
    "POSASR": {"category": "unspecified"},
    "PRE 14A": {"category": "Proxy"},
    "PRE 14C": {"category": "unspecified"},
    "PREC14A": {"category": "Proxy"},
    "PREC14C": {"category": "unspecified"},
    "PREM14A": {"category": "unspecified"},
    "PREM14C": {"category": "unspecified"},
    "PREN14A": {"category": "unspecified"},
    "PRER14A": {"category": "unspecified"},
    "PRER14C": {"category": "unspecified"},
    "PRRN14A": {"category": "unspecified"},
    "PX14A6G": {"category": "unspecified"},
    "PX14A6N": {"category": "unspecified"},
    "QRTLYRPT": {"category": "unspecified"},
    "QUALIF": {"category": "unspecified"},
    "REG-NR": {"category": "unspecified"},
    "REVOKED": {"category": "unspecified"},
    "RW": {"category": "Prospectus", "description": "Withdrawl of a prospectus with EFFECT"},
    "RW WD": {"category": "unspecified"},
    "S-1": {"category": "Prospectus"},
    "S-1/A": {"category": "Prospectus"},
    "S-11": {"category": "unspecified"},
    "S-11MEF": {"category": "unspecified"},
    "S-1MEF": {"category": "unspecified"},
    "S-20": {"category": "unspecified"},
    "S-3": {"category": "Prospectus"},
    "S-3ASR": {"category": "Prospectus"},
    "S-3D": {"category": "unspecified"},
    "S-3DPOS": {"category": "unspecified"},
    "S-3MEF": {"category": "unspecified"},
    "S-4": {"category": "Prospectus"},
    "S-4 POS": {"category": "Prospectus"},
    "S-4EF": {"category": "unspecified"},
    "S-4MEF": {"category": "unspecified"},
    "S-6": {"category": "Prospectus"},
    "S-8": {"category": "Prospectus"},
    "S-8 POS": {"category": "Prospectus"},
    "S-B": {"category": "unspecified"},
    "S-BMEF": {"category": "unspecified"},
    "SC 13D": {"category": "Ownership"},
    "SC 13D/A": {"category": "Ownership"},
    "SC 13E1": {"category": "unspecified"},
    "SC 13E3": {"category": "unspecified"},
    "SC 13G": {"category": "Ownership"},
    "SC 13G/A": {"category": "Ownership"},
    "SC 14D9": {"category": "unspecified"},
    "SC 14F1": {"category": "unspecified"},
    "SC 14N": {"category": "unspecified"},
    "SC TO-C": {"category": "unspecified"},
    "SC TO-I": {"category": "unspecified"},
    "SC TO-T": {"category": "unspecified"},
    "SC13E4F": {"category": "unspecified"},
    "SC14D1F": {"category": "unspecified"},
    "SC14D9C": {"category": "unspecified"},
    "SC14D9F": {"category": "unspecified"},
    "SD": {"category": "unspecified"},
    "SDR": {"category": "unspecified"},
    "SE": {"category": "unspecified"},
    "SEC ACTION": {"category": "unspecified"},
    "SEC STAFF ACTION": {"category": "unspecified"},
    "SEC STAFF LETTER": {"category": "unspecified"},
    "SF-1": {"category": "unspecified"},
    "SF-3": {"category": "unspecified"},
    "SL": {"category": "unspecified"},
    "SP 15D2": {"category": "unspecified"},
    "STOP ORDER": {"category": "unspecified"},
    "SUPPL": {"category": "unspecified"},
    "T-3": {"category": "unspecified"},
    "TA-1": {"category": "unspecified"},
    "TA-2": {"category": "unspecified"},
    "TA-W": {"category": "unspecified"},
    "TACO": {"category": "unspecified"},
    "TH": {"category": "unspecified"},
    "TTW": {"category": "unspecified"},
    "UNDER": {"category": "unspecified"},
    "UPLOAD": {"category": "unspecified"},
    "WDL-REQ": {"category": "unspecified"},
    "X-17A-5": {"category": "unspecified"}
}