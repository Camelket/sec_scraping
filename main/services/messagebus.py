import logging
from typing import Callable, Dict, List, Union, Type, TYPE_CHECKING
from main.domain import commands
from main.services import unit_of_work


logger = logging.getLogger(__name__)

# Message = Union[commands.Command, events.Event]
Message = commands.Command


class MessageBus:
    def __init__(
        self,
        uow: unit_of_work.AbstractUnitOfWork,
        # event_handlers: Dict[Type[events.Event], List[Callable]],
        command_handlers: Dict[Type[commands.Command], Callable],
    ):
        self.uow = uow
        # self.event_handlers = event_handlers
        self.command_handlers = command_handlers
        self.command_history = []

    def handle(self, message: Message):
        self.queue = [message]
        while self.queue:
            message = self.queue.pop(0)
            # if isinstance(message, events.Event):
            #     self.handle_event(message)
            if isinstance(message, commands.Command):
                self.command_history.append(message)
                self.handle_command(message)
            else:
                raise Exception(f"{message} was not an Event or Command")
    
    # def handle_event(self, event: events.Event):
    #     for handler in self.event_handlers[type(event)]:
    #         try:
    #             logger.debug("handling event %s with handler %s", event, handler)
    #             handler(event)
    #             self.queue.extend(self.uow.collect_new_events())
    #         except Exception:
    #             logger.exception("Exception handling event %s", event)
    #             continue

    def handle_command(self, command: commands.Command):
        logger.debug("handling command %s", command)
        try:
            handler = self.command_handlers[type(command)]
            handler(command)
            # self.queue.extend(self.uow.collect_new_events())
        except KeyError:
            logger.warning("Exception handling command %s. No Handler for Command registered", command)
        except Exception:
            logger.exception("Exception handling command %s", command)
    
    def collect_command_history(self):
        command_history = self.command_history
        self.command_history = []
        return command_history