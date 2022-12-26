from collections import defaultdict
import re
from spacy.tokens import Token, Span, Doc
import logging
from operator import itemgetter
from .filing_nlp_utils import get_dep_distance_between_spans, get_dep_distance_between, get_span_distance
logger = logging.getLogger(__name__)

def get_all_overlapping_intervals(array: list[tuple[int, int]]) -> list[tuple[int, int]]:
    '''find overlapping intervals using sweep line algorithm. O(n*log(n))'''
    points = []
    for idx, x in enumerate(array):
        points.append((idx, "L", x[0]))
        points.append((idx, "R", x[1]))
    sorted_points = sorted(points, key=itemgetter(2, 1))
    logger.debug(f"sorted_points {sorted_points}")
    current_open = None
    overlap = set()
    overlapping_groups = []
    all_currently_open = set()
    counter = -1
    for i, point in enumerate(sorted_points):
        idx, kind, x = point[0], point[1], point[2]
        logger.debug(("currently on item: ", idx, kind, x))
        if kind == "R":
            all_currently_open.remove(idx)
            if current_open is None:
                pass
                # raise ValueError(f"we shouldnt have no interval being tracked and encountering an end of an interval at the same time")
            else:
                if current_open == idx:
                    logger.debug("closing current open")
                    current_open = None
                else:
                    pass
        if kind == "L":
            all_currently_open.add(idx)
            if current_open is not None:
                if len(overlapping_groups[counter]) == 0:
                    overlapping_groups[counter].add(current_open)
                overlapping_groups[counter].add(idx)
                if current_open != idx:
                    overlap.add(current_open)
                    overlap.add(idx)
                    if array[current_open][1] <= array[idx][1]:
                        current_open = idx
            if current_open is None:
                counter += 1
                overlapping_groups.append(set())
                current_open = idx
            logger.debug(f"current_open item: {current_open if current_open is not None else None}")
        logger.debug(f"overlap at end of cycle: {overlap}")
    overlapping_tuples: list[tuple[int, int]] = []
    for over in overlap:
        overlapping_tuples.append(array[over])
    _overlapping_groups = []
    for idx, items in enumerate(overlapping_groups):
        if (len(items) == 0 or len(items) == 1):
           pass
        else:
            tuples = []
            for i in items:
                tuples.append(array[i])
            _overlapping_groups.append(tuples)
    return overlapping_tuples, _overlapping_groups

def get_longest_from_overlapping_groups(overlapping_tuples: list[list[tuple[int, int]]]):
    longest = []
    for group in overlapping_tuples:
        item_lengths = [t[1]-t[0] for t in group]
        longest.append(group[item_lengths.index(max(item_lengths))])
    return longest
      

class AliasCache:
    '''
    This is to keep track of alias assignment across AliasSetter/AliasMatcher Components.
    Custom extensions added:
        Doc extensions:
            ??
    Usage:
        Get created in the AliasMatcher Component and made availabel as a custom Doc extension ._.alias_cache
        in the AliasMatcher Component:
            1) add a possible alias through add_alias
            2) add the references to the base alias assigned in 1)
        In the AliasSetter Component:
            3) given a list of Spans, determine if the spans have aliases with high
               enough similarity in the sent to the right of them
            2) connect the alias, origin and similarity_score through assign_alias if we found one

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
        self._base_alias_references: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        
    def get_base_aliases_by_origin(self, origin: Token|Span) -> list[tuple[int, int]]|None:
        '''get all assigned aliases to this origin.'''
        origin_tuple = self._get_start_end_tuple(origin)
        return self._origin_alias_map.get(origin_tuple, None)
    
    def get_all_aliases_by_origin(self, origin: Token|Span) -> dict[tuple[int, int], list[tuple[int, int]]]:
        '''
        returns a dict of base_alias and references to the alias(excluding origin).'''
        result = dict()

        base_aliases = self.get_base_aliases_by_origin(origin)
        if base_aliases:
            for base_tuple in base_aliases:
                result[base_tuple] = []
                references = self._base_alias_references[base_tuple]
                if references:
                    result[base_tuple] = references
        return result

    def add_reference_to_alias(self, alias: Token|Span, reference: Token|Span):
        alias_tuple = self._get_start_end_tuple(alias)
        reference_tuple = self._get_start_end_tuple(reference)
        self._add_reference_to_alias(alias_tuple, reference_tuple)
    
    def _add_reference_to_alias(self, alias_tuple: tuple[int, int], reference_tuple: tuple[int, int]):
        if (alias_tuple in self._already_assigned_aliases) or (alias_tuple in self._unassigned_aliases):
            self._base_alias_references[alias_tuple].append(reference_tuple)
        else:
            logger.warning(f"Add the base Alias({alias_tuple}) first, before registering references to the base Alias.")  
        
    def add_alias(self, alias: Token|Span, references: list[Span]|None=None):
        '''Add an alias to the map of aliases. No assignment happens here.'''
        alias_tuple = self._get_start_end_tuple(alias)
        for x in range(alias_tuple[0], alias_tuple[1]+1):
            self._idx_alias_map[x] = alias_tuple
        sent_start_end = self.get_sent_start_end_tuple(alias)
        self._sent_alias_map[sent_start_end].append(alias_tuple)
        self._alias_sent_map[alias_tuple] = sent_start_end
        self._unassigned_aliases.add(alias_tuple)
        if references:
            for ref in references:
                self.add_reference_to_alias(alias, ref)
    
    def assign_alias(self, alias: Token|Span, origin: Token|Span, score: float):
        '''Assign an alias to an origin and set the similarity_score of the alias to the origin.'''
        alias_tuple = self._get_start_end_tuple(alias)
        origin_tuple = self.get_sent_start_end_tuple(origin)
        self._assign_alias(alias_tuple, origin_tuple, score)
    
    def unassign_alias(self, alias: Token|Span):
        '''Remove the link between alias and origin and put the alias into an unassigned state'''
        alias_tuple = self._get_start_end_tuple(alias)
        self._unassign_alias(alias_tuple)

    def remove_alias(self, alias: Token|Span):
        '''Remove the alias completely (needs to be added again, alias loses assignment to origin)'''
        alias_tuple = self._get_start_end_tuple(alias)
        self._remove_alias(alias_tuple)
    
    def _remove_references(self, alias_tuple: tuple[int, int]):
        if self._base_alias_references.get(alias_tuple, None):
            self._base_alias_references.pop(alias_tuple)
    
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
        self._remove_references(alias_tuple)
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
        return (start_idx, end_idx)
    
    def get_sent_start_end_tuple(self, alias: Token|Span):
        if isinstance(alias, Token):
            sent_start = alias.sent[0].i
            sent_end = alias.sent[-1].i
        if isinstance(alias, Span):
            sent_start = alias[0].sent[0].i
            sent_end = alias[0].sent[-1].i
        return (sent_start, sent_end)


def ensure_alias_cache_created(doc: Doc):
    if doc._.alias_cache is None:
        alias_cache = AliasCache()
        doc._.alias_cache = alias_cache
    
class AliasMatcher:
    def __init__(self, vocab):
        self._set_needed_extensions()
        self.vocab = vocab
        self.chars_to_tokens_map: dict = None

    def __call__(self, doc: Doc):
        ensure_alias_cache_created(doc)
        if self.chars_to_tokens_map is None:
            self.chars_to_tokens_map = self._get_chars_to_tokens_map(doc)
        base_aliases = self.get_base_alias_spans(doc)
        base_to_references = self.get_references_to_base_alias_spans(doc, base_aliases)
        self.add_aliases(doc, base_aliases, base_to_references)
        return doc
    
    def _set_needed_extensions(self):
        if not Doc.has_extension("alias_cache"):
            Doc.set_extension("alias_cache", default=None)

    def _get_chars_to_tokens_map(self, doc: Doc):
        chars_to_tokens = {}
        for token in doc:
            for i in range(token.idx, token.idx + len(token.text)):
                chars_to_tokens[i] = token.i
        return chars_to_tokens

    def get_base_alias_spans(self, doc: Doc) -> list[Span]:
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
                            spans.append(alias_span)
                        else:
                            logger.debug(
                                f"couldnt find start/end token for alias: {possible_alias}; start/end token: {start_token}/{end_token}"
                            )
        return spans
    
    def get_references_to_base_alias_spans(self, doc: Doc, base_spans: list[Span]) -> dict[Span, list[Span]]:
        # TODO: how can i handle the plural of the base_spans?
        all_start_end_tuples = []
        start_end_to_base_map = {}
        for base in base_spans:
            print(f"current_base: {base.text}")
            pattern = re.compile(f"(?:[^a-zA-z])({base.text})(?:[^a-zA-z])")
            text_to_search = doc[base.end:].text
            for m in re.finditer(pattern, text_to_search):
                if m:
                    # FIXME: what is the correct offest? why does private placement not get added to extended spans ?
                    # where is the offset wrong, the -2 hack doesnt work
                    match = re.search(base.text, text_to_search[m.span()[0]:m.span()[1]])
                    if match:
                        search_start_offset = (len(doc.text) - len(text_to_search))
                        start_char = search_start_offset + m.start() + match.start()
                        end_char = search_start_offset + m.start() + match.end() - 1
                    start_token = self.chars_to_tokens_map.get(start_char)
                    end_token =  self.chars_to_tokens_map.get(end_char) + 1
                    if (start_token and end_token):
                        start_end = (start_token, end_token)
                        if start_end_to_base_map.get(start_end, None):
                            old_base = start_end_to_base_map[start_end]
                            new_base = base
                            if len(old_base) > len(new_base):
                                pass
                            else:
                                start_end_to_base_map[start_end] = new_base
                        else:
                            start_end_to_base_map[start_end] = base
                        all_start_end_tuples.append(start_end)
                    else:
                        logger.debug(f"couldnt find start/end token for match: {match}; start/end token: {start_token}/{end_token}")
        overlapping, overlapping_groups = get_all_overlapping_intervals(all_start_end_tuples)
        longest_from_groups = get_longest_from_overlapping_groups(overlapping_groups)
        base_span_to_reference_span = defaultdict(list)
        for each in longest_from_groups:
            base_span_to_reference_span[start_end_to_base_map[each]].append(doc[each[0]:each[1]])
        none_overlapping = list(set(all_start_end_tuples) - set(overlapping))
        for each in none_overlapping:
            base_span_to_reference_span[start_end_to_base_map[each]].append(doc[each[0]:each[1]])
        return base_span_to_reference_span

    def add_aliases(self, doc: Doc, base_aliases: list[Span], base_to_references: dict[Span, list[Span]]):
        for alias in base_aliases:
            doc._.alias_cache.add_alias(alias)
        for base, references in base_to_references.items():
            for ref in references:
                doc._.alias_cache.add_reference_to_alias(base, ref)

class AliasSetter:
    # TODO: change __init__ params to set config
    def __init__(self):
        self.similarity_score_threshold: float = 0.75 #FIXME: add a sensible value and make the similarity score something between [0, 1]
        self.very_similar_threshold: float = 0.65
        self.dep_distance_weight = 0.7
        self.span_distance_weight: float = 0.3
    
    def __call__(self, doc: Doc, origins: list[Span]):
        self._ensure_alias_cache_present(doc)
        '''
        1) receive doc and a list of spans (list of spans could be through a ent_type set with other config for component or through a separat function where we pass spans explicitly => would mean this component isnt just assigned to the pipeline but would need explicit calls)
        2) get all possible aliases for each span
        3) calculate similarity score
        4) max similarity_scores gotten
        5) see if we are above threshold for origin -> alias assignment
        6) if 5) assign alias to origin
        '''
        self.assign_aliases_to_origins(doc, origins)
        
    
    def assign_aliases_to_origins(self, doc: Doc, origins: list[Span]):
        assignment_history = []
        for origin in origins:
            base_aliases = self._get_right_base_aliases_in_same_sent(doc, origin)
            if base_aliases:
                best_alias, best_score = self._determine_best_alias(origin, base_aliases)
                if best_alias:
                    if self._similarity_score_above_threshold(best_score):
                        doc._.alias_cache.assign_alias(best_alias, origin, best_score)
                        assignment_history.append((best_alias, origin, best_score))
        logger.debug(f"{self} assigned following aliases (alias, origin, similarity_score): {assignment_history}")

    def _similarity_score_above_threshold(self, similarity_score: float):
        if similarity_score >= self.similarity_score_threshold:
            return True
        return False

    def _determine_best_alias(self, origin: Span, aliases: list[Span]) -> tuple[Span, float]:
        print(f"_determine_best_alias given: origin({origin}) aliases({aliases})")
        best_score = 0
        best_alias = None
        for base_alias in aliases:
            similarity_score = get_span_to_span_similarity_score(origin, base_alias)
            if similarity_score > best_score:
                best_score = similarity_score
                best_alias = base_alias
        return best_alias, best_score
    
    def _get_right_base_aliases_in_same_sent(self, doc: Doc, origin: Span) -> list[Span]|None:
        # TODO: ensure somewhere that we pass aliases as Spans (create a Span if alias would be a single Token)
        '''get all base aliases in the same sentence with origin[-1].i < base_alias[0].i or None if none present'''
        result = []
        sent_start_end_tuple = doc._.alias_cache.get_sent_start_end_tuple(origin)
        base_alias_in_sent = doc._.alias_cache._sent_alias_map[sent_start_end_tuple]
        if base_alias_in_sent:
            for base_alias in base_alias_in_sent:
                if base_alias[0] > origin.start:
                    result.append(doc[base_alias[0]:base_alias[1]])
            return result
        else:
            return None
    
    def _ensure_alias_cache_present(self, doc: Doc):
        if not Doc.has_extension("alias_cache"):
            raise AttributeError(f"The alias_cache extension wasnt set, make sure the AliasMatcher Component is placed in the pipeline and called before the AliasSetter is used.")
        if not doc._.alias_cache:
            raise AttributeError("The alias_cache wasnt correctly set on the doc, make sure AliasMatcher component is working as intended and correctly added to the pipeline.")
    
def _get_origin_to_target_similarity_map(origin: list[Token]|Span, target: list[Token]|Span) -> dict[tuple[Token, Token], float]:
    similarity_map = {}
    for origin_token in origin:
        for target_token in target:
            similarity = origin_token.similarity(target_token)
            similarity_map[(origin_token, target_token)] = similarity
    return similarity_map

def _calculate_similarity_score(
    target: list[Token]|Span,
    similarity_map: dict[tuple[Token, Token], float],
    dep_distance: int,
    span_distance: int,
    very_similar_threshold: float,
    dep_distance_weight: float,
    span_distance_weight: float,
) -> float:
    very_similar = sum([v > very_similar_threshold for v in similarity_map.values()])
    very_similar_score = very_similar / len(target) if very_similar != 0 else 0
    dep_distance_score = dep_distance_weight * (1 / dep_distance)
    span_distance_score = span_distance_weight * (10 / span_distance)
    total_score = dep_distance_score + span_distance_score + very_similar_score
    return total_score
    
def get_span_to_span_similarity_score(
    span1: list[Token]|Span,
    span2: list[Token]|Span,
    dep_distance_weight: float = 0.7,
    span_distance_weight: float = 0.3,
    very_similar_threshold: float = 0.65,
) -> float:
    span1_tokens = (
        span1._.premerge_tokens
        if (span1.has_extension("premerge_tokens") and (span1._.was_merged if span1.has_extension("was_merged") else False)) 
        else span1
    )
    span2_tokens = (
        span2._.premerge_tokens
        if (span2.has_extension("premerge_tokens") and (span2._.was_merged if span2.has_extension("was_merged") else False))
        else span2
    )
    similarity_map = _get_origin_to_target_similarity_map(span1_tokens, span2_tokens)
    dep_distance = get_dep_distance_between_spans(span1, span2)
    span_distance = get_span_distance(span1, span2)
    if dep_distance and span_distance and similarity_map:
        score = _calculate_similarity_score(
            span2,
            similarity_map,
            dep_distance,
            span_distance,
            very_similar_threshold=very_similar_threshold,
            dep_distance_weight=dep_distance_weight,
            span_distance_weight=span_distance_weight,
        )
        return score
    return 0
    # start with aliasSetter component which handles base alias to origin mapping based on a given list of Spans
    # using similarity score from filing_nlp.py
    






