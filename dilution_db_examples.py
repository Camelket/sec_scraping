from boot import bootstrap_dilution_db
from main.configs import FactoryConfig

def get_dilution_db_dev_instance():
    cnf = FactoryConfig("dev")
    db = bootstrap_dilution_db(start_orm=True, config=cnf)