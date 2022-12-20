from spacy.tokens import Token, Span
from pandas import Timestamp
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)


class DatefulRelation:
    def _format_context_as_lemmas(
        self, context: dict[str, list[Token]]
    ) -> dict[str, set[str]]:
        if context is None:
            return {}
        lemma_dict = defaultdict(set)
        for key, tokens in context.items():
            if tokens:
                for token in tokens:
                    lemma_dict[key].add(token.lemma_)
        return lemma_dict


class DatetimeRelation(DatefulRelation):
    def __init__(
        self, spacy_date: Span, timestamp: Timestamp, context: dict[str, list[Token]]
    ):
        self.spacy_date = spacy_date
        self.timestamp = timestamp
        self._context = context
        self.lemmas = self._format_context_as_lemmas(
            context.get("formatted", None) if context is not None else None
        )
    
    @property
    def context(self):
        return self._context if self._context else {}
    
    @context.setter
    def context(self, value):
        if isinstance(value, dict):
            if ["original", "formatted"] in value.keys():
                self._context = value
                return
            else:
                logger.warning(f"context needs to be of type dict and have at least keys: 'original' and 'formatted'. received: {type(value), value}")
        else:
            logger.warning(f"expecting a dict for the context instance variable: context; got: {type(value)}")
        logger.warning(f"Failed to set context attribute of {self}")
    

    def __repr__(self):
        return f"{self.spacy_date} - {self.context.get('formatted', None)}"

    def __eq__(self, other):
        if isinstance(other, DatetimeRelation):
            if (
                (self.spacy_date == other.spacy_date) and
                (self.timestamp == other.timestamp) and
                (self.lemmas == other.lemmas) and
                (self.context == other.context)
            ):
                return True
        return False