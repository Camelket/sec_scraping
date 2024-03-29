import inspect
from dilution_db import DilutionDB
from main.services.messagebus import Message, MessageBus
from main.services import handlers, unit_of_work
from main.adapters import orm
from main.configs import GlobalConfig, cnf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def bootstrap_dilution_db(
    start_orm=True,
    config: GlobalConfig = cnf,
    uow: unit_of_work.AbstractUnitOfWork = None,
) -> DilutionDB:
    if uow is None:
        session_factory = sessionmaker(
            bind=create_engine(config.DILUTION_DB_CONNECTION_STRING),
            expire_on_commit=False
        )
        uow = unit_of_work.SqlAlchemyCompanyUnitOfWork(
            session_factory=session_factory
        )
    if start_orm is True:
        orm.start_mappers()
    dependencies = {"uow": uow}
    injected_command_handlers = {
        command_type: inject_dependencies(handler, dependencies)
        for command_type, handler in handlers.COMMAND_HANDLERS.items()
    }
    message_bus = MessageBus(
        uow=uow,
        command_handlers=injected_command_handlers)
    dilution_db = DilutionDB(
        config=config,
        uow=uow,
        message_bus=message_bus
    )
    return dilution_db
    

def inject_dependencies(handler, dependencies):
    params = inspect.signature(handler).parameters
    deps = {
        name: dependency
        for name, dependency in dependencies.items()
        if name in params
    }
    return lambda message: handler(message, **deps)