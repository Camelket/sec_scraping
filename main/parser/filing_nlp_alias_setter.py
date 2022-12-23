from collections import defaultdict
import re
from spacy.tokens import Token, Span, Doc
import logging
logger = logging.getLogger(__name__)

def find_overlapping_intervals(array: list[tuple[int, int]]):
    '''find overlapping intervals using sweep line algorithm. O(nlog(n))'''
    points = []
    overlapping: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, x in enumerate(array):
        points.append(tuple(idx, "S", x[0]))
        points.append(tuple(idx, "E", x[1]))
    half_sorted = points.sorted(key=lambda x: x[2])
    sorted_points = half_sorted.sorted(key=lambda x: x[1], reverse=True)
    currently_open_intervals = set()
    for i, point in enumerate(points):
        idx, kind, x = point[0], point[1], point[2]
        if len(currently_open_intervals) == 0:
            if kind != "S":
                raise ValueError(f"we shouldnt have no interval being tracked and encountering an end of an interval at the same time")
            else:
                currently_open_intervals.add(points[idx])
        else:
            

class AliasCache:
    '''
    This is to keep track of alias assignment across AliasSetter/AliasMatcher Components.
    Custom extensions added:
        Doc extensions:
            ??
    Usage:
        Create instance of this class and pass to AliasSetter/AliasMatcher components through config on creation.
        OR I could create an instance on the doc as a custom extension if it isnt set yet?
        In the AliasSetter Component:
            1) add a possible alias through add_alias
            2) determine if a possible alias has an origin
            3) add the alias, origin and similarity_score through assign_alias

    restrictions:
        No overlapping aliases.
        An alias can only be assigned to one origin at a time.
    '''
    def __init__(self):
        self._idx_alias_map: dict[int, tuple[int, int]] = dict()
        self._sent_alias_map: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        self._alias_sent_map: dict[tuple[int, int], tuple[int, int]] = dict()
        self._already_assigned_aliases: set[tuple[int, int]] = set()
        self._alias_assignment: dict[tuple[int, int], tuple[int, int]] = dict()
        self._origin_alias_map: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        self._alias_assignment_score: dict[tuple[int, int], float] = dict()
        self._unassigned_aliases: set[tuple[int, int]] = set()
    
    def get_aliases_by_origin(self, origin: Token|Span) -> list[tuple[int, int]]|None:
        '''get all assigned aliases to this origin.'''
        origin_tuple = self._get_start_end_tuple(origin)
        return self._origin_alias_map.get(origin_tuple, None)
    
    def add_alias(self, alias: Token|Span):
        '''Add an alias to the map of aliases. No assignment happens here.'''
        alias_tuple = self._get_start_end_tuple(alias)
        for x in range(alias_tuple[0], alias_tuple[1]+1):
            self._idx_alias_map[x] = alias_tuple
        sent_start_end = self._get_sent_start_end_tuple(alias)
        self._sent_alias_map[sent_start_end].append(alias_tuple)
        self._alias_sent_map[alias_tuple] = sent_start_end
    
    def assign_alias(self, alias: Token|Span, origin: Token|Span, score: float):
        '''Assign an alias to an origin and set the similarity_score of the alias to the origin.'''
        alias_tuple = self._get_start_end_tuple(alias)
        origin_tuple = self._get_sent_start_end_tuple(origin)
        self._assign_alias(alias_tuple, origin_tuple, score)
    
    def unassign_alias(self, alias: Token|Span):
        '''Remove the link between alias and origin and put the alias into an unassigned state'''
        alias_tuple = self._get_start_end_tuple(alias)
        self._unassign_alias(alias_tuple)

    def remove_alias(self, alias: Token|Span):
        '''Remove the alias completely (needs to be added again, alias loses assignment to origin)'''
        alias_tuple = self._get_start_end_tuple(alias)
        self._remove_alias(alias_tuple)
    
    def _unassign_alias(self, alias_tuple: tuple[int, int]):
        if alias_tuple in self._already_assigned_aliases:
            # first remove assignments in orign to alias map
            self._remove_alias_tuple_from_origin_alias_map(alias_tuple)
            self._alias_assignment.pop(alias_tuple)
            self._alias_assignment_score.pop(alias_tuple)
            self._unassigned_aliases.add(alias_tuple)

    def _remove_alias_tuple_from_origin_alias_map(self, alias_tuple):
        origin_tuple = self._alias_assignment[alias_tuple]
        idx_to_remove = None
        for idx, each in self._origin_alias_map[origin_tuple]:
            if each == alias_tuple:
                idx_to_remove = idx
                break
        if idx_to_remove is not None:
            self._origin_alias_map[origin_tuple].pop(idx_to_remove)
    
    def _remove_alias(self, alias_tuple: tuple[int, int]):
        self._unassign_alias(alias_tuple)
        for x in range(alias_tuple[0], alias_tuple[1]+1):
            self._idx_alias_map.pop(x)
        self._remove_alias_tuple_from_sent_alias_map(alias_tuple)
        self._alias_sent_map.pop(alias_tuple)

    def _remove_alias_tuple_from_sent_alias_map(self, alias_tuple):
        idx_to_remove = None
        for idx, each in self._sent_alias_map[self._alias_sent_map[alias_tuple]]:
            if each == alias_tuple:
                idx_to_remove = idx
                break
        if idx_to_remove is not None:
            self._sent_alias_map[self._alias_sent_map[alias_tuple]].pop(idx_to_remove)

    def _assign_alias(self, alias_tuple: tuple[int, int], origin_tuple: tuple[int, int], score: float):
        if alias_tuple not in self._unassigned_aliases:
            if alias_tuple in self._already_assigned_aliases:
                logger.warning(f"This alias_tuple:{alias_tuple} already is assigned to an origin:{self._alias_assignment[alias_tuple]}")
            else:
                logger.warning(f"This alias_tuple:{alias_tuple} wasn't added yet. Use add_alias to add the alias before assigning the alias_tuple.")
        else:
            self._unassigned_aliases.remove(alias_tuple)
            self._alias_assignment[alias_tuple] = origin_tuple
            self._alias_assignment_score = score
            self._origin_alias_map[origin_tuple].append(alias_tuple)
    
    def _get_start_end_tuple(self, alias: Token|Span):
        if isinstance(alias, Token):
            start_idx = alias.i
            end_idx = alias.i
        if isinstance(alias, Span):
            start_idx = alias.start
            end_idx = alias.end
        return tuple(start_idx, end_idx)
    
    def _get_sent_start_end_tuple(self, alias: Token|Span):
        if isinstance(alias, Token):
            sent_start = alias.sent[0].i
            sent_end = alias.sent[-1].i
        if isinstance(alias, Span):
            sent_start = alias[0].sent[0].i
            sent_end = alias[0].sent[-1].i
        return tuple(sent_start, sent_end)


def ensure_alias_cache_extension(doc: Doc):
    if not Doc.has_extension("alias_cache"):
        Doc.set_extension("alias_cache", default=None)
    if doc._.alias_cache is None:
        alias_cache = AliasCache()
        doc._.alias_cache = alias_cache
    
class AliasMatcher:
    def __init__(self, vocab):
        self.vocab = vocab
        self.chars_to_tokens_map: dict = None

    def __call__(self, doc: Doc):
        ensure_alias_cache_extension(doc)
        if self.chars_to_tokens_map is None:
            self.chars_to_tokens_map = self._get_chars_to_tokens_map(doc)
        pass

    def _get_chars_to_tokens_map(self, doc: Doc):
        chars_to_tokens = {}
        for token in doc:
            for i in range(token.idx, token.idx + len(token.text)):
                chars_to_tokens[i] = token.i
        return chars_to_tokens

    def get_base_alias_spans(self, doc: Doc):
        # secu_alias_exclusions = set(["we", "us", "our"])
        spans = []
        parenthesis_pattern = re.compile(r"\([^(]*\)")
        possible_alias_pattern = re.compile(r"(?:\"|“)[a-zA-Z\s-]*(?:\"|”)")
        for match in re.finditer(parenthesis_pattern, doc.text):
            if match:
                start_idx = match.start()
                for possible_alias in re.finditer(
                    possible_alias_pattern, match.group()
                ):
                    if possible_alias:
                        start_token = self.chars_to_tokens_map.get(
                            start_idx + possible_alias.start()
                        )
                        end_token = self.chars_to_tokens_map.get(
                            start_idx + possible_alias.end() - 1
                        )
                        if (start_token is not None) and (end_token is not None):
                            alias_span = doc[start_token + 1 : end_token]
                            # if alias_span.text in secu_alias_exclusions:
                            #     logger.debug(
                            #         f"Was in secu_alias_exclusions -> discarded alias span: {alias_span}"
                            #     )
                            #     continue
                            spans.append(alias_span)
                        else:
                            logger.debug(
                                f"couldnt find start/end token for alias: {possible_alias}; start/end token: {start_token}/{end_token}"
                            )
        return spans
    
    def get_extended_alias_spans(self, doc: Doc, base_spans: list[Span]):
        # TODO: how can i handle the plural of the base_spans?
        # TODO: how can I handle aliases like ["Investor", "Investor Warrant"] since they overlap if i just regex it
        # Do i just take the longest match if they overlap ?



