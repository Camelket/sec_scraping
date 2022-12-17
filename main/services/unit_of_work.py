from typing_extensions import Self
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from abc import abstractmethod, ABC
from main.configs import cnf
from main.adapters import repository


class AbstractUnitOfWork(ABC):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args):
        self.rollback()

    def commit(self):
        self._commit()
    
    def rollback(self):
        self._rollback()

    # def collect_new_events(self):
    #     for product in self.products.seen:
    #         while product.events:
    #             yield product.events.pop(0)

    @abstractmethod
    def _commit(self):
        raise NotImplementedError

    @abstractmethod
    def _rollback(self):
        raise NotImplementedError

print(cnf.DILUTION_DB_CONNECTION_STRING)
DEFAULT_SESSION_FACTORY = sessionmaker(
    bind=create_engine(
        url=cnf.DILUTION_DB_CONNECTION_STRING,
        # isolation_level="REPEATABLE READ"  
    ),
    expire_on_commit=False
)

# print(cnf.DILUTION_DB_CONNECTION_STRING)

class SqlAlchemyCompanyUnitOfWork(AbstractUnitOfWork):
    def __init__(self, session_factory=DEFAULT_SESSION_FACTORY):
        self.session_factory = session_factory

    def __enter__(self):
        self.session = self.session_factory()  # type: Session
        self.company = repository.SqlAlchemyCompanyRepository(self.session)
        return super().__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)
        self.session.close()

    def _commit(self):
        self.session.commit()

    def _rollback(self):
        self.session.rollback()
    
class FakeCompanyUnitOfWork(AbstractUnitOfWork):
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        self.company = repository.FakeCompanyRepository()
    
    def __exit__(self):
        pass

    def _commit(self):
        pass
    
    def _rollback(self):
        pass

    
