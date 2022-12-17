from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Integer,
    Column,
    String,
    Table,
    ForeignKey,
    join,
)
from sqlalchemy.orm import (
    relationship,
    registry,
    remote,
    foreign
)
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy import types
from main.domain.model import (
    Underwriter,
    OfferingStatus,
    SecurityCusip,
    FormType,
    Security,
    SecurityAccnOccurence,
    SecurityConversion,
    SecurityOutstanding,
    SecurityAuthorized,
    ShelfOffering,
    ShelfSecurityRegistration,
    ShelfSecurityComplete,
    ShelfRegistration,
    ResaleSecurityRegistration,
    ResaleSecurityComplete,
    ResaleRegistration,
    EffectRegistration,
    Sic,
    FilingParseHistoryEntry,
    FilingLink,
    CashOperating,
    CashFinancing,
    CashInvesting,
    NetCashAndEquivalents,
    Company
)

import logging
logger = logging.getLogger(__name__)

class SecurityTypes(types.TypeDecorator):
    impl = types.String
    cache_ok = True


reg = registry()

underwriters = Table(
    "underwriters",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("name_", String)
)

underwriters_shelf_offerings = Table(
    "underwriters_shelf_offerings",
    reg.metadata,
    Column("offerings_id", ForeignKey("shelf_offerings.id")),
    Column("underwriter_id", ForeignKey("underwriters.id"))
)

offering_status = Table(
    "offering_status",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("name_", String)
)

securities_cusip = Table(
    "securities_cusip",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("cusip_number", String),
    Column("security_id", ForeignKey("securities.id"))
)

form_types = Table(
    "form_types",
    reg.metadata,
    Column("form_type", String, primary_key=True),
    Column("category", String)
)

securities = Table(
    "securities",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("security_name", String),
    Column("security_type", SecurityTypes),
    Column("underlying_security_id", ForeignKey("securities.id")),
    Column("security_attributes", JSON)
)

securities_accn_occurence = Table(
    "securities_accn_occurence",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("security_id", ForeignKey("securities.id")),
    Column("accn", ForeignKey("filing_links.accn"))
)

securities_conversion = Table(
    "securities_conversion",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("from_security_id", ForeignKey("securities.id")),
    Column("to_security_id", ForeignKey("securities.id")),
    Column("conversion_attributes", JSON)
)

securities_outstanding = Table(
    "securities_outstanding",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("security_id", ForeignKey("securities.id")),
    Column("amount", BigInteger),
    Column("instant", Date)
)

securities_authorized = Table(
    "securities_authorized",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("security_type",  SecurityTypes),
    Column("amount", BigInteger),
    Column("instant", Date)
)

shelf_offerings =  Table(
    "shelf_offerings",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("shelf_registrations_id", ForeignKey("shelf_registrations.id")),
    Column("accn", String),
    Column("filing_date", Date),
    Column("offering_type", String),
    Column("final_offering_amount", BigInteger),
    Column("anticipated_offering_amount", BigInteger),
    Column("offering_status_id", ForeignKey("offering_status.id")),
    Column("commencment_date", Date),
    Column("end_date", Date)
)

securities_shelf_offerings_registered = Table(
    "securities_shelf_offerings_registered",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("security_id", ForeignKey("securities.id")),
    Column("shelf_offerings_id", ForeignKey("shelf_offerings.id")),
    Column("source_security_id", ForeignKey("securities.id")),
    Column("amount_dollar", BigInteger, nullable=True),
    Column("amount_security", BigInteger, nullable=True)
)

securities_shelf_offerings_completed = Table(
    "securities_shelf_offerings_completed",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("security_id", ForeignKey("securities.id")),
    Column("shelf_offerings_id", ForeignKey("shelf_offerings.id")),
    Column("source_security_id", ForeignKey("securities.id")),
    Column("amount_dollar", BigInteger, nullable=True),
    Column("amount_security", BigInteger, nullable=True)
)

securities_resale_registered = Table(
    "securities_resale_registered",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("security_id", ForeignKey("securities.id")),
    Column("resale_registrations_id", ForeignKey("resale_registrations.id")),
    Column("source_security_id", ForeignKey("securities.id")),
    Column("amount_dollar", BigInteger, nullable=True),
    Column("amount_security", BigInteger, nullable=True)
)

securities_resale_completed = Table(
    "securities_resale_completed",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("security_id", ForeignKey("securities.id")),
    Column("resale_registrations_id", ForeignKey("resale_registrations.id")),
    Column("source_security_id", ForeignKey("securities.id")),
    Column("amount_dollar", BigInteger, nullable=True),
    Column("amount_security", BigInteger, nullable=True)
)

shelf_registrations = Table(
    "shelf_registrations",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("accn", String),
    Column("file_number", String),
    Column("form_type", ForeignKey("form_types.form_type")),
    Column("capacity", BigInteger),
    Column("total_amount_raised", BigInteger),
    Column("effect_date", Date),
    Column("last_update", Date),
    Column("expiry", Date),
    Column("filing_date", Date),
    Column("is_active", Boolean),
)

resale_registrations = Table(
    "resale_registrations",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("accn", String),
    Column("file_number", String),
    Column("form_type", ForeignKey("form_types.form_type")),
    Column("effect_date", Date),
    Column("last_update", Date),
    Column("expiry", Date),
    Column("filing_date", Date),
    Column("is_active", Boolean),
)

effect_registrations = Table(
    "effect_registrations",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("accn", String),
    Column("form_type", ForeignKey("form_types.form_type")),
    Column("file_number", String),
    Column("effective_date", Date)
)

sics = Table(
    "sics",
    reg.metadata,
    Column("sic", Integer, primary_key=True),
    Column("sector", String),
    Column("industry", String),
    Column("division", String)
)

filing_parse_history = Table(
    "filing_parse_history",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("accession_number", String),
    Column("date_parsed", Date)
)

filing_links = Table(
    "filing_links",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("filing_html", String, unique=True),
    Column("accn", String),
    Column("form_type", ForeignKey("form_types.form_type")),
    Column("filing_date", Date),
    Column("description_", String),
    Column("file_number", String)
)

cash_operating = Table(
    "cash_operating",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("from_date", Date),
    Column("to_date", Date),
    Column("amount", BigInteger)
)

cash_financing = Table(
    "cash_financing",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("from_date", Date),
    Column("to_date", Date),
    Column("amount", BigInteger)
)

cash_investing = Table(
    "cash_investing",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("from_date", Date),
    Column("to_date", Date),
    Column("amount", BigInteger)
)

net_cash_and_equivalents = Table(
    "net_cash_and_equivalents",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("company_id", ForeignKey("companies.id")),
    Column("instant", Date),
    Column("amount", BigInteger)
)

# company_last_update = Table(
#     Column("company_id", ForeignKey("companies.id"), primary_key=True),
#     Column("filings_download_lud", DateTime, nullabel=True),
#     Column("filing_links_lud", DateTime, nullabel=True),
#     Column("outstanding_shares_lud", DateTime, nullabel=True),
#     Column("net_cash_and_equivalents_lud", DateTime, nullabel=True),
#     Column("cash_burn_rate_lud", DateTime, nullabel=True),
# )

companies = Table(
    "companies",
    reg.metadata,
    Column("id", Integer, primary_key=True),
    Column("cik", String, nullable=False),
    Column("sic", ForeignKey("sics.sic")),
    Column("symbol", String, unique=True),
    Column("name_", String),
    Column("description_", String)
)

def start_mappers():
    logger.debug(f"Creating mappers for sqlalchemy model...")
    underwriters_mapper = reg.map_imperatively(
        Underwriter,
        underwriters,
        properties={
            "name": underwriters.c.name_
        }
    )

    offering_status_mapper = reg.map_imperatively(
        OfferingStatus,
        offering_status,
        properties={
            "name": offering_status.c.name_
        }
    )

    securities_cusip_mapper = reg.map_imperatively(
        SecurityCusip,
        securities_cusip
    )

    form_types_mapper = reg.map_imperatively(
        FormType,
        form_types
    )

    securities_outstanding_mapper = reg.map_imperatively(
        SecurityOutstanding,
        securities_outstanding
    )

    securities_authorized_mapper = reg.map_imperatively(
        SecurityAuthorized,
        securities_authorized
    )

    securities_mapper = reg.map_imperatively(
        Security,
        securities,
        properties={
            "name": securities.c.security_name,
            "underlying": relationship(
                Security,
                primaryjoin=securities.c.id==securities.c.underlying_security_id,
                remote_side=securities.c.id,
                uselist=False,
                lazy="joined"
            ),
            "outstanding": relationship(
                securities_outstanding_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "conversion_from_self": relationship(
                SecurityConversion,
                primaryjoin=securities.c.id==securities_conversion.c.from_security_id,
                lazy="joined"
            ),
            "conversion_to_self": relationship(
                SecurityConversion,
                primaryjoin=securities.c.id==securities_conversion.c.to_security_id,
                lazy="joined"
            )
        }
    )

    securities_accn_occurence_mapper = reg.map_imperatively(
        SecurityAccnOccurence,
        securities_accn_occurence
    )

    securities_conversion_mapper = reg.map_imperatively(
        SecurityConversion,
        securities_conversion,
        properties={
            "from_security": relationship(
                securities_mapper,
                primaryjoin=securities_conversion.c.from_security_id == remote(securities.c.id),
                # foreign_keys=[securities_conversion.c.from_security_id],
                back_populates="conversion_from_self",
                collection_class=list,
                lazy="joined"
            ),
            "to_security": relationship(
                securities_mapper,
                primaryjoin=securities_conversion.c.to_security_id == remote(securities.c.id),
                # foreign_keys=[securities_conversion.c.to_security_id],
                back_populates="conversion_to_self",
                collection_class=list,
                lazy="joined"
            )
        }
    )

    securities_shelf_offerings_registered_mapper = reg.map_imperatively(
        ShelfSecurityRegistration,
        securities_shelf_offerings_registered,
        properties={
            "security": relationship(
                securities_mapper,
                primaryjoin=securities_shelf_offerings_registered.c.security_id==securities.c.id,
                foreign_keys=[securities_shelf_offerings_registered.c.security_id],
                uselist=False,
                lazy="joined"),
            "source_security": relationship(
                securities_mapper,
                primaryjoin=securities_shelf_offerings_registered.c.source_security_id==securities.c.id,
                foreign_keys=[securities_shelf_offerings_registered.c.source_security_id],
                uselist=False,
                lazy="joined")
        }
    )

    securities_shelf_offerings_completed_mapper = reg.map_imperatively(
        ShelfSecurityComplete,
        securities_shelf_offerings_completed,
        properties={
            "security": relationship(
                securities_mapper,
                primaryjoin=securities_shelf_offerings_completed.c.security_id==securities.c.id,
                foreign_keys=[securities_shelf_offerings_completed.c.security_id],
                uselist=False,
                lazy="joined"),
            "source_security": relationship(
                securities_mapper,
                primaryjoin=securities_shelf_offerings_completed.c.source_security_id==securities.c.id,
                foreign_keys=[securities_shelf_offerings_completed.c.source_security_id],
                uselist=False,
                lazy="joined")
        }
    )

    securities_resale_registered_mapper = reg.map_imperatively(
        ResaleSecurityRegistration,
        securities_resale_registered,
        properties={
            "security": relationship(
                securities_mapper,
                primaryjoin=securities_resale_registered.c.security_id==securities.c.id,
                foreign_keys=[securities_resale_registered.c.security_id],
                uselist=False,
                lazy="joined"),
            "source_security": relationship(
                securities_mapper,
                primaryjoin=securities_resale_registered.c.source_security_id==securities.c.id,
                foreign_keys=[securities_resale_registered.c.source_security_id],
                uselist=False,
                lazy="joined")
        }
    )

    securities_resale_completed_mapper = reg.map_imperatively(
        ResaleSecurityComplete,
        securities_resale_completed,
        properties={
            "security": relationship(
                securities_mapper,
                primaryjoin=securities_resale_completed.c.security_id==securities.c.id,
                foreign_keys=[securities_resale_completed.c.security_id],
                uselist=False,
                lazy="joined"),
            "source_security": relationship(
                securities_mapper,
                primaryjoin=securities_resale_completed.c.source_security_id==securities.c.id,
                foreign_keys=[securities_resale_completed.c.source_security_id],
                uselist=False,
                lazy="joined")
        }
    )

    shelf_offerings_mapper = reg.map_imperatively(
        ShelfOffering,
        shelf_offerings,
        properties={
            "underwriters": relationship(
                underwriters_mapper,
                secondary=underwriters_shelf_offerings,
                collection_class=set,
                lazy="selectin"
            ),
            "registrations": relationship(
                securities_shelf_offerings_registered_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "completed": relationship(
                securities_shelf_offerings_completed_mapper,
                collection_class=set,
                lazy="selectin"
            )
        }
    )

    shelf_registrations_mapper = reg.map_imperatively(
        ShelfRegistration,
        shelf_registrations,
        properties={
            "offerings": relationship(
                shelf_offerings_mapper,
                collection_class=set,
                lazy="selectin"
            )
        }
    )

    resale_registrations_mapper = reg.map_imperatively(
        ResaleRegistration,
        resale_registrations,
        properties={
            "registrations": relationship(
                securities_resale_registered_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "completed": relationship(
                securities_resale_completed_mapper,
                collection_class=set,
                lazy="selectin"
            )
        }
    )

    effect_registrations_mapper = reg.map_imperatively(
        EffectRegistration,
        effect_registrations
    )

    sics_mapper = reg.map_imperatively(
        Sic,
        sics
    )

    filing_parse_history_mapper = reg.map_imperatively(
        FilingParseHistoryEntry,
        filing_parse_history
    )

    filing_links_mapper = reg.map_imperatively(
        FilingLink,
        filing_links
    )

    cash_operating_mapper = reg.map_imperatively(
        CashOperating,
        cash_operating
    )

    cash_financing_mapper = reg.map_imperatively(
        CashFinancing,
        cash_financing
    )

    cash_investing_mapper = reg.map_imperatively(
        CashInvesting,
        cash_investing
    )

    net_cash_and_equivalents_mapper = reg.map_imperatively(
        NetCashAndEquivalents,
        net_cash_and_equivalents
    )

    companies_mapper = reg.map_imperatively(
        Company,
        companies,
        properties={
            "name": companies.c.name_,
    
            "securities": relationship(
                securities_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "security_conversion": relationship(
                securities_conversion_mapper,
                secondary=join(companies, securities, companies.c.id==securities.c.company_id).join(securities_conversion, ((securities.c.id==securities_conversion.c.from_security_id)|(securities.c.id==securities_conversion.c.to_security_id))),
                primaryjoin=companies.c.id==remote(securities.c.company_id),
                secondaryjoin=(securities_conversion.c.to_security_id==remote(securities.c.id))|(securities_conversion.c.from_security_id==remote(securities.c.id)),
                lazy="joined",
                collection_class=list,
                viewonly=True
            ),
            "securities_authorized": relationship(
                securities_authorized_mapper,
                lazy="selectin",
                collection_class=set
            ),
            "shelfs": relationship(
                shelf_registrations_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "resales": relationship(
                resale_registrations_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "effects": relationship(
                effect_registrations_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "filing_parse_history": relationship(
                filing_parse_history_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "filing_links": relationship(
                filing_links_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "cash_operating": relationship(
                cash_operating_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "cash_financing": relationship(
                cash_financing_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "cash_investing": relationship(
                cash_investing_mapper,
                collection_class=set,
                lazy="selectin"
            ),
            "net_cash_and_equivalents": relationship(
                net_cash_and_equivalents_mapper,
                collection_class=set,
                lazy="selectin"
            )

        }
    )








