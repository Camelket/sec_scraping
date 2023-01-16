from collections import defaultdict
import re
from spacy.tokens import Token, Span, Doc
from spacy.vocab import Vocab
from spacy.matcher import Matcher
import logging
from operator import itemgetter
from typing import Generator
from .filing_nlp_utils import _set_extension, get_dep_distance_between_spans, get_span_distance, change_capitalization_after_symbol
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
        An alias cache should only be used with a single Doc object
        No overlapping aliases.
        An alias can only have one origin assigned with a similarity score at a time.
    '''

    def __init__(self):
        self._idx_alias_map: dict[int, tuple[int, int]] = dict()
        self._sent_alias_map: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        self._alias_sent_map: dict[tuple[int, int], tuple[int, int]] = dict()
        self._already_assigned_aliases: set[tuple[int, int]] = set()
        self._alias_origin_map: dict[tuple[int, int], tuple[int, int]] = dict()
        self._origin_base_alias_map: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        self._alias_assignment_score: dict[tuple[int, int], float] = dict()
        self._unassigned_aliases: set[tuple[int, int]] = set()
        self._base_alias_references: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        self._base_alias_covered_tokens: set[int] = set()
        self._reference_alias_set: set[tuple[int, int]] = set()
        self._base_alias_set: set[tuple[int, int]] = set()

        self._components_to_alias: dict[tuple[tuple[int, int]], tuple[int, int]] = dict() #mapping multiple components to their base_alias
        self._alias_to_components: dict[tuple[int, int], tuple[tuple[int, int]]] = dict() #component makeup of a multi component base alias
        self._alias_to_ultimate_origin: dict[tuple[int, int], list[tuple[int, int]]] = dict() # map the alias to its ultimate single or multi origin
        self._parent_map: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list) # map an item to its immediate parent
        self._children_map: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list) # map an item to all its immediate children
        self._tuple_to_type_map: dict[tuple[int, int], str] = dict() # map a tuple to its associacted type

    def _add_multi_component_alias(self, alias: tuple[int, int], components: list[tuple[int, int]]):
        # ensure components arent origins
        # for component in components:
        #     if self._tuple_to_type_map[component] == "origin":
        #         raise ValueError(f"A component of a multi_component_alias making up a new origin cant be itself an origin!")
        if len(components) <= 1:
            raise ValueError(f"expecting more than a single component to create the relationships for a multi_base_alias, got components: {components} for multi_base_alias: {alias}")
        # check if the components are registered as one of: base_alias or reference_alias
        # assign final origins of alias
        for component in components:
            self._add_parent(component, alias)
        self._alias_to_ultimate_origin[alias] = self._bfs_ultimate_origins(components)
        self._alias_to_components[alias] = components
        self._components_to_alias[(a for a in components)] = alias
        # where do i update children and parent map
        # add tuple type at same stage as parents and children
        
    def _bfs_ultimate_origins(self, components: list[tuple[int, int]]):
        #  BFS search for all parents without parents (leaves of the tree)
        ultimate_origin = []
        # print(f"running bfs for components: {components}")
        visited = set()
        for component in components:
            origins = []
            queue = [[component]]
            node = component
            while queue:
                path = queue.pop(0)
                node = path[-1]
                if node == None:
                    break
                visited.add(node)
                parents = self._parent_map.get(node, None)
                if not parents or parents == []:                   
                    origins.append(node)
                    continue
                else:
                    for parent in parents:
                        if parent in visited:
                            continue
                        new_path = list(path)
                        new_path += parents
                        queue.append(new_path)
            ultimate_origin += origins
        # print(f"-> ultimate_origin: {ultimate_origin}")
        return ultimate_origin
    

            # could i improve this if i looked up the node in the already existing ultimate origins

        
    def get_base_aliases_by_origin(self, origin: Token|Span) -> list[tuple[int, int]]|None:
        '''get all assigned aliases to this origin.'''
        origin_tuple = self._get_start_end_tuple(origin)
        return self._origin_base_alias_map.get(origin_tuple, None)
    
    def get_all_aliases_by_origin(self, origin: Token|Span) -> dict[tuple[int, int], list[tuple[int, int]]]:
        '''
        returns a dict of base_alias and references to the alias(excluding origin, but including base alias).'''
        result = dict()

        base_aliases = self.get_base_aliases_by_origin(origin)
        if base_aliases:
            for base_tuple in base_aliases:
                result[base_tuple] = []
                references = self._base_alias_references[base_tuple]
                if references:
                    result[base_tuple] = references
        return result
    
    def get_all_alias_references_by_origin(self, origin: Token|Span) -> list[tuple[int, int]]:
        '''returns a list of all references to the base aliases connected to this origin'''
        result = []
        base_aliases = self.get_base_aliases_by_origin(origin)
        if base_aliases:
            for base_tuple in base_aliases:
                references = self._base_alias_references[base_tuple]
                if references:
                    for ref in references:
                        result.append(ref)
        return result

    def add_reference_to_alias(self, alias: Token|Span, reference: Token|Span):
        alias_tuple = self._get_start_end_tuple(alias)
        reference_tuple = self._get_start_end_tuple(reference)
        self._add_reference_to_alias(alias_tuple, reference_tuple)
    
    def _add_parent(self, parent: tuple[int, int], child: tuple[int,int]):
        if parent not in self._parent_map[child]:
            self._parent_map[child].append(parent)

    def _add_child(self, parent: tuple[int, int], child: tuple[int, int]):
        if child not in self._children_map[parent]:
            self._children_map[parent].append(child)
        
    def _add_child_parent_entry(self, parent: tuple[int, int], child: tuple[int, int]):
        self._add_child(parent, child)
        self._add_parent(parent, child)
    
    def _add_reference_to_alias(self, alias_tuple: tuple[int, int], reference_tuple: tuple[int, int]):
        if (alias_tuple in self._already_assigned_aliases) or (alias_tuple in self._unassigned_aliases):
            self._reference_alias_set.add(reference_tuple)
            for x in range(reference_tuple[0], reference_tuple[1]+1):
                self._idx_alias_map[x] = reference_tuple
            self._base_alias_references[alias_tuple].append(reference_tuple)
            self._add_child_parent_entry(alias_tuple, reference_tuple)
            self._tuple_to_type_map[reference_tuple] = "reference"
        else:
            logger.warning(f"Add the base Alias({alias_tuple}) first, before registering references to the base Alias.")  
        
    def add_alias(self, alias: Token|Span, references: list[Span]|None=None):
        '''Add an alias to the map of aliases. No assignment happens here.'''
        alias_tuple = self._get_start_end_tuple(alias)
        for x in range(alias_tuple[0], alias_tuple[1]+1):
            self._idx_alias_map[x] = alias_tuple
            self._base_alias_covered_tokens.add(x)
        self._base_alias_set.add(alias_tuple)
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
        origin_tuple = self._get_start_end_tuple(origin)
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
            references = self._base_alias_references.pop(alias_tuple)
            for ref_tuple in references:
                self._reference_alias_set.remove(ref_tuple)
                self._tuple_to_type_map.pop(ref_tuple)
    
    def _unassign_alias(self, alias_tuple: tuple[int, int]):
        if alias_tuple in self._already_assigned_aliases:
            # first remove assignments in orign to alias map
            self._remove_alias_tuple_from_origin_alias_map(alias_tuple)
            self._alias_origin_map.pop(alias_tuple)
            self._alias_assignment_score.pop(alias_tuple)
            self._unassigned_aliases.add(alias_tuple)

    def _remove_alias_tuple_from_origin_alias_map(self, alias_tuple):
        origin_tuple = self._alias_origin_map[alias_tuple]
        idx_to_remove = None
        for idx, each in self._origin_base_alias_map[origin_tuple]:
            if each == alias_tuple:
                idx_to_remove = idx
                break
        if idx_to_remove is not None:
            self._origin_base_alias_map[origin_tuple].pop(idx_to_remove)
    
    def _remove_alias(self, alias_tuple: tuple[int, int]):
        self._unassign_alias(alias_tuple)
        self._remove_references(alias_tuple)
        for x in range(alias_tuple[0], alias_tuple[1]+1):
            self._idx_alias_map.pop(x)
        self._base_alias_set.remove(alias_tuple)
        self._remove_alias_tuple_from_sent_alias_map(alias_tuple)
        self._alias_sent_map.pop(alias_tuple)
        self._tuple_to_type_map.pop(alias_tuple)
        for x in range(alias_tuple[0], alias_tuple[1]+1):
            self._base_alias_covered_tokens.remove(x)

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
                logger.warning(f"This alias_tuple:{alias_tuple} already is assigned to an origin:{self._alias_origin_map[alias_tuple]}")
            else:
                logger.warning(f"This alias_tuple:{alias_tuple} wasn't added yet. Use add_alias to add the alias before assigning the alias_tuple.")
        else:
            self._unassigned_aliases.remove(alias_tuple)
            self._alias_origin_map[alias_tuple] = origin_tuple
            self._alias_assignment_score = score
            self._origin_base_alias_map[origin_tuple].append(alias_tuple)
            self._tuple_to_type_map[origin_tuple] = "origin"
    
    def _get_start_end_tuple(self, alias: Token|Span):
        if isinstance(alias, Token):
            start_idx = alias.i
            end_idx = start_idx + 1
        if isinstance(alias, Span):
            start_idx = alias.start
            end_idx = alias.end
        return (start_idx, end_idx)
    
    def get_sent_start_end_tuple(self, alias: Token|Span):
        if isinstance(alias, Token):
            sent_start = alias.sent.start
            sent_end = alias.sent.end
        elif isinstance(alias, Span):
            sent_start = alias[0].sent.start
            sent_end = alias[0].sent.end
        else:
            raise TypeError(f"expecting spacy.tokens.Token or spacy.tokens.Span, got: {type(alias)}")
        return (sent_start, sent_end)
        

def ensure_alias_cache_created(doc: Doc):
    if doc._.alias_cache is None:
        alias_cache = AliasCache()
        doc._.alias_cache = alias_cache
    
class AliasMatcher:
    '''
    A Component which finds ("alias") in SEC filings and tries to find the references to
    them further to the right/down in the text.

    Custom Extensions:
        Doc Extensions:
            ._.alias_cache 
            (see AliasCache class for details, but in short: maps to keep track of aliases)
            ._.parentheses_base_alias_map (dict[tuple[int, int], list[tuple[int, int]]])
            ._.sent_parentheses_map (dict[tuple[int, int], list[tuple[int, int]]])
            
        Span Extensions:
            ._.is_alias (getter -> boolean)
            ._.is_base_alias (getter -> boolean)
            ._.is_reference_alias (getter -> boolean)

        Token Extensions:
            ._.is_part_of_alias (getter -> boolean)
            ._.containing_alias_span (getter -> tuple[int, int])
    '''
    def __init__(self):
        self._set_needed_extensions()
        self.chars_to_tokens_map: dict = None
        self.parentheses_base_alias_map: dict[tuple[int, int], list[tuple[int, int]]] = None
        self.aliases_are_set = False

    def __call__(self, doc: Doc):
        ensure_alias_cache_created(doc)
        self.chars_to_tokens_map = self._get_chars_to_tokens_map(doc)
        doc._.parentheses_base_alias_map = self._get_parentheses_base_alias_map(doc, self.chars_to_tokens_map)
        doc._.sent_parentheses_map = self._get_sent_parentheses_map(doc, doc._.parentheses_base_alias_map)
        self.handle_base_aliases(doc, doc._.parentheses_base_alias_map)
        self.aliases_are_set = True
        return doc

    def handle_base_aliases(self, doc: Doc, parentheses_base_alias_map: dict[tuple[int, int], list[tuple[int, int]]]):
        base_aliases = []
        for _, values in parentheses_base_alias_map.items():
            if len(values) > 0:
                for value in values:
                    base_aliases.append(doc[value[0]:value[1]])
        logger.warning(f"base_aliases found: {base_aliases}")
        if len(base_aliases) > 0:
            self.add_base_aliases(doc, base_aliases, get_references=True)
    
    def reinitalize_extensions(self, doc: Doc):
        doc._.alias_cache = None
        ensure_alias_cache_created(doc)
        doc._.parentheses_base_alias_map = defaultdict(list)
        doc._.sent_parentheses_map = defaultdict(list)
        self.aliases_are_set = False
    
    def _set_needed_extensions(self):
        if not Doc.has_extension("alias_cache"):
            Doc.set_extension("alias_cache", default=None)
        
        def is_alias(span: Span):
            start_end_tuple = span.doc._.alias_cache._get_start_end_tuple(span)
            if start_end_tuple in span.doc._.alias_cache._base_alias_set:
                return True
            if start_end_tuple in span.doc._.alias_cache._reference_alias_set:
                return True
            return False
        
        def is_reference_alias(span: Span):
            if (span.doc._.alias_cache._get_start_end_tuple(span) 
                in span.doc._.alias_cache._reference_alias_set):
                return True
            return False
        
        def is_base_alias(span: Span):
            if (span.doc._.alias_cache._get_start_end_tuple(span) 
                in span.doc._.alias_cache._base_alias_set):
                return True
            return False
        
        def is_part_of_alias(token: Token):
            if token.doc._.alias_cache._idx_alias_map.get(token.i, None) is not None:
                return True
            return False
        
        def get_containing_alias_span(token: Token):
            return token.doc._.alias_cache._idx_alias_map.get(token.i, None)
        
        doc_extensions = [
            {"name": "parentheses_base_alias_map", "kwargs": {"default": defaultdict(list)}},
            {"name": "sent_parentheses_map", "kwargs": {"default": defaultdict(list)}}
        ]
        span_extensions = [
            {"name": "is_alias", "kwargs": {"getter": is_alias}},
            {"name": "is_base_alias", "kwargs": {"getter": is_base_alias}},
            {"name": "is_reference_alias", "kwargs": {"getter": is_reference_alias}},
        ]
        token_extensions = [
            {"name": "is_part_of_alias", "kwargs": {"getter": is_part_of_alias}},
            {"name": "containing_alias_span", "kwargs": {"getter": get_containing_alias_span}},
        ]
        for each in span_extensions:
            _set_extension(Span, each["name"], each["kwargs"])
        for each in doc_extensions:
            _set_extension(Doc, each["name"], each["kwargs"])
        for each in token_extensions:
            _set_extension(Token, each["name"], each["kwargs"])

    def _get_chars_to_tokens_map(self, doc: Doc) -> dict[int, int]:
        chars_to_tokens = {}
        for token in doc:
            for i in range(token.idx, token.idx + len(token.text) + 1):
                chars_to_tokens[i] = token.i
        return chars_to_tokens
    
    def _get_parentheses_matches(self, text: str) -> Generator[re.Match[str], None, None]:
        parenthesis_pattern = re.compile(r"\([^(]*\)")
        for match in re.finditer(parenthesis_pattern, text):
            if match:
                yield match
            else:
                yield None
    
    def _get_in_double_quote_matches(self, text: str) -> Generator[re.Match[str], None, None]:
        double_quote_pattern = re.compile(r"(?:\"|“)([a-zA-Z]*[a-zA-Z\s-]*[a-zA-Z]*)(?:\"|”)")
        for match in re.finditer(double_quote_pattern, text):
            if match:
                yield match
            else:
                yield None
    
    
    def _get_parentheses_base_alias_map(self, doc: Doc, chars_to_tokens_map: dict[int, int]) -> dict[tuple[int,int], list[tuple[int, int]]]:
        '''gets all base_alias start_end_token tuples and maps them to their parentheses start_end_token tuple, in discovery order.'''
        parentheses_to_base_alias_tuples: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for parentheses_match in self._get_parentheses_matches(doc.text):
            if parentheses_match:
                parentheses_start = chars_to_tokens_map.get(parentheses_match.start(), None)
                parentheses_end = chars_to_tokens_map.get(parentheses_match.end(), None)
                if (parentheses_end is None) or (parentheses_start is None):
                    logger.warning(f"couldnt assign token to start/end char of parentheses, match_text/start_char/end_char/chars_to_tokens_map: {doc.text[parentheses_match.start():parentheses_match.end()]}/{parentheses_match.start()}/{parentheses_match.end()}/{chars_to_tokens_map}")
                parentheses_start_end = (parentheses_start, parentheses_end + 1)
                for quote_match in self._get_in_double_quote_matches(parentheses_match.group()):
                    if quote_match:
                        # +/- 1 to remove the matched double quotes
                        alias_start = chars_to_tokens_map[quote_match.start() + parentheses_match.start() + 1]
                        alias_end = chars_to_tokens_map[quote_match.end() + parentheses_match.start() - 1]
                        parentheses_to_base_alias_tuples[parentheses_start_end].append((alias_start, alias_end))
        return parentheses_to_base_alias_tuples
    
    def _get_sent_parentheses_map(self, doc: Doc, parentheses_base_alias_map: dict[tuple[int,int], list[tuple[int, int]]]):
        parentheses_tuples = sorted(list(parentheses_base_alias_map.keys()), key=lambda x: x[0])
        sent_parentheses_map = defaultdict(list)
        parentheses_idx = 0
        for sent in doc.sents:
            sent_start_end_tuple = (sent.start, sent.end)
            if parentheses_idx >= len(parentheses_tuples):
                break
            while parentheses_tuples[parentheses_idx][0] < sent.end:
                current_parentheses = parentheses_tuples[parentheses_idx]
                if (current_parentheses[0] >= sent.start) and (current_parentheses[1] <= sent.end):
                    sent_parentheses_map[sent_start_end_tuple].append(current_parentheses)
                    parentheses_idx += 1
                else:
                    break 
                if parentheses_idx >= len(parentheses_tuples):
                    break
        return sent_parentheses_map
    
    def _get_regex_reference_patterns(self, span: Span) -> list[tuple[re.Pattern, str]]:
        patterns = [(re.compile(f"(?:[^a-zA-z])({re.escape(span.text)})(?:[^a-zA-z])"), span.text)]
        if "-" in span.text:
            core_str = change_capitalization_after_symbol(span.text, '-')
            recapitalized_pattern = re.compile(
                f"(?:[^a-zA-z])({re.escape(core_str)})(?:[^a-zA-z])")
            patterns.append((recapitalized_pattern, core_str))
        return patterns
        
    def get_longest_references_to_spans(self, doc: Doc, base_spans: list[Span]) -> dict[Span, list[Span]]:
        # TODO[epic="important"]: how can i handle the plural of the base_spans?
        all_start_end_tuples = []
        start_end_to_base_map = {}
        base_spans = sorted(base_spans, key=lambda x: len(x.text), reverse=True)
        logger.debug(f"get_longest_references_to_spans; used base_spans: {base_spans}")
        for base in base_spans:
            # print(f"current_base: {base.text}")
            patterns = self._get_regex_reference_patterns(base)
            logger.debug(f"working with reference core_str: {[p[1] for p in patterns]}")
            text_to_search = doc[base.end:].text
            for (pattern, core_str) in patterns:
                for m in re.finditer(pattern, text_to_search):
                    if m:
                        match = re.search(core_str, text_to_search[m.span()[0]:m.span()[1]+1])
                        if match:
                            search_start_offset = (len(doc.text) - len(text_to_search))
                            start_char = search_start_offset + m.start() + match.start()
                            end_char = search_start_offset + m.start() + match.end() - 1
                            start_token = self.chars_to_tokens_map.get(start_char)
                            end_token =  self.chars_to_tokens_map.get(end_char) + 1
                            if (start_token and end_token):
                                is_invalid = False
                                for x in range(start_token, end_token + 1):
                                    if x in doc._.alias_cache._base_alias_covered_tokens:
                                        logger.debug(f"get_longest_references_to_spans disregarded match {m} because tokens in it are covered by a base_alias")
                                        is_invalid = True
                                        break
                                start_end = (start_token, end_token)
                                if is_invalid is False:
                                    logger.debug(f"found base -> reference: {base} -> {doc[start_token:end_token]}")
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
        logger.debug(f"all_start_end_tuples: {all_start_end_tuples}")
        logger.debug(f"overlapping_groups: {overlapping_groups}")
        logger.debug(f"longest_from_groups: {longest_from_groups}")
        base_span_to_reference_span = defaultdict(list)
        for each in longest_from_groups:
            base_span_to_reference_span[start_end_to_base_map[each]].append(doc[each[0]:each[1]])
        none_overlapping = list(set(all_start_end_tuples) - set(overlapping))
        for each in none_overlapping:
            base_span_to_reference_span[start_end_to_base_map[each]].append(doc[each[0]:each[1]])
        logger.info(f"base_span_to_alias_span: base -> references")
        for base, references in base_span_to_reference_span.items():
            logger.info(f"- {base} {(base.start, base.end)} -> {references} {[(ref.start, ref.end) for ref in references]}")
        return base_span_to_reference_span

    def add_base_aliases(self, doc: Doc, base_aliases: list[Span], get_references: bool=True):
        for alias in base_aliases:
            doc._.alias_cache.add_alias(alias)
        logger.info(f"base_alias_covered_tokens set before getting references: {doc._.alias_cache._base_alias_covered_tokens}")
        if get_references is True:
            base_to_references = self.get_longest_references_to_spans(doc, base_aliases)
            for base, references in base_to_references.items():
                for ref in references:
                    doc._.alias_cache.add_reference_to_alias(base, ref)
        else:
            base_to_references = {k:[] for k in base_aliases}
        logger.info(f"AliasMatcher added following base_alias, references: base_alias -> references")
        for base, references in base_to_references.items():
            logger.info(f" - {base} -> {[(ref, (ref.start, ref.end)) for ref in references]}")
        logger.info("----")

class AliasSetter:
    '''
    Component which sets multi base alias relations and connects single compound 
    base aliases to their origin.
    
    Relies on AliasMatcher being called on the Doc before!
    '''
    def __init__(self, vocab: Vocab):
        self.similarity_score_threshold: float = 1.4 #FIXME: add a sensible value after thinking of a better similarity score function
        self.very_similar_threshold: float = 0.65
        self.dep_distance_weight: float = 0.7
        self.span_distance_weight: float = 0.3
        self.multi_compound_alias_matcher = self._init_multi_component_alias_matcher(vocab)
        self._is_multi_compound_alias_relationship_assignment_done: bool = False

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
        
        self.assign_single_component_base_aliases_to_origins(doc, origins)
        if self._is_multi_compound_alias_relationship_assignment_done is False:
            self._handle_relationships_for_multi_component_aliases(doc)
            self._is_multi_compound_alias_relationship_assignment_done = True
        # logger.debug(f"doc._.parentheses_base_alias_map: parentheses tuple -> base_alias spans")
        # for parentheses_tuple, base_alias_tuples in doc._.parentheses_base_alias_map.items():
        #     logger.debug(f"- {parentheses_tuple} -> {[doc[i[0]:i[1]] for i in base_alias_tuples]}")
        # logger.debug("----")
        return doc
    
    def _init_multi_component_alias_matcher(self, vocab: Vocab):
        matcher = Matcher(vocab)
        multi_compound_alias_patterns = [
            [
                {"_": {"is_part_of_alias": True}, "OP": "+"},
                {"OP": "*"},
                {"POS": "CCONJ"},
                {"OP": "*"},
                {"_": {"is_part_of_alias": True}, "OP": "+"}
            ]
        ]
        matcher.add("multi_compound_alias", multi_compound_alias_patterns)
        return matcher
    
    def _eliminate_alias_references_from_origins(self, origins: list[Span]):
        valid_origins = []
        for origin in origins:
            if origin._.is_reference_alias is False:
                valid_origins.append(origin)
        return valid_origins
    
    def _eliminate_base_aliases_from_origins(self, origins: list[Span]):
        valid_origins = []
        for origin in origins:
            if origin._.is_base_alias is False:
                valid_origins.append(origin)
        return valid_origins
    
    def _handle_relationships_for_multi_component_aliases(self, doc: Doc):
        multi_component_base_alias_map = self._get_multi_component_base_alias_map(doc)
        logger.info(f"AliasSetter registered following multi_base_aliases: (components) -> multi_base_alias")
        for multi_base, components in multi_component_base_alias_map.items():
            doc._.alias_cache._add_multi_component_alias(multi_base, components)
            logger.info(f" - ({[doc[i[0]:i[1]] for i in components]}) -> {doc[multi_base[0]:multi_base[1]]}")
        logger.info("----")
        # call aliasCache add methode for each multi component alias
    
    def _has_expression_in_parentheses_before_alias_span(self, doc: Doc, parentheses: Span, alias_span: Span, expressions: list[str]):
        if (parentheses.start > alias_span.start) or (parentheses.end < alias_span.end):
            raise ValueError(f"alias_span is outside of the given parentheses span; expecting a alias_span within the parentheses span") 
        text_before = doc[parentheses.start:alias_span.start].text
        matches = []
        for expression in expressions:
            expression_matches = re.findall(expression, text_before)
            if expression_matches:
                matches += expression_matches
        if len(matches) > 0:
            return True
        return False
    
    def get_first_single_component_base_alias_spans(self, doc: Doc) -> list[Span]:
        first_single_component_base_alias_spans = []
        for p, aliases in doc._.parentheses_base_alias_map.items():
            counter = 0
            for alias in aliases:
                if counter > 0:
                    break
                if self._has_expression_in_parentheses_before_alias_span(
                    doc,
                    p, 
                    alias,
                    expressions=["together", "and with", "collectively"]
                    ) is True:
                    counter += 1
                    continue
                else:
                    first_single_component_base_alias_spans.append(doc[alias[0]:alias[1]])
        return first_single_component_base_alias_spans
    
    def _get_multi_component_base_alias_map(self, doc: Doc) -> dict[tuple[int, int], list[tuple[int, int]]]:
        multi_component_alias_map = defaultdict(list)
        for parentheses_tuple, aliases in doc._.parentheses_base_alias_map.items():
            if len(aliases) > 1:
                parentheses_span = doc[parentheses_tuple[0]:parentheses_tuple[1]]
                matches = self.multi_compound_alias_matcher(parentheses_span)
                if matches:
                    longest_match = matches[matches.index(max(matches, key=lambda x: x[2]-x[1]))]
                    aliases: list[tuple[int, int]] = []
                    for token_idx in range(longest_match[1], longest_match[2] + 1):
                        containing_alias_span = parentheses_span[token_idx]._.containing_alias_span
                        if (containing_alias_span) and (containing_alias_span not in aliases):
                            aliases.append(containing_alias_span)
                    if len(aliases) > 0:
                        multi_base = aliases[-1]
                        components = aliases[:-1]
                        multi_component_alias_map[multi_base] = components 
        return multi_component_alias_map
                   
            

    def assign_single_component_base_aliases_to_origins(self, doc: Doc, origins: list[Span]):
        # TODO[epic="important"]: change this to try and match single compound base aliases only
        assignment_history = []
        assigned_origin_tuples = set()
        origins = self._eliminate_alias_references_from_origins(origins)
        origins = self._eliminate_base_aliases_from_origins(origins)
        all_origin_tuples = set([(origin.start, origin.end) for origin in origins])
        for origin in origins:
            base_aliases = self._get_right_single_component_base_aliases_in_same_sent(doc, origin)
            if base_aliases:
                best_alias, best_score = self._determine_best_alias(origin, base_aliases)
                if best_alias:
                    if self._similarity_score_above_threshold(best_score):
                        doc._.alias_cache.assign_alias(best_alias, origin, best_score)
                        assignment_history.append((best_alias, origin, best_score))
                        assigned_origin_tuples.add((origin.start, origin.end))
        logger.info(f"AliasSetter assigned following aliases (alias, origin, similarity_score):")
        for entry in assignment_history:
            logger.info(f" - {entry}")
        logger.info("----")
        logger.debug(f"origins which didnt get a base_alias assigned: {[doc[j[0]:j[1]] for j in (all_origin_tuples - assigned_origin_tuples)]}")

    def _similarity_score_above_threshold(self, similarity_score: float):
        if similarity_score >= self.similarity_score_threshold:
            return True
        return False

    def _determine_best_alias(self, origin: Span, aliases: list[Span]) -> tuple[Span, float]:
        best_score = 0
        best_alias = None
        for base_alias in aliases:
            logger.debug(f"_determine_best_alias: similarity score for [{origin}:{base_alias}]")
            similarity_score = get_span_to_span_similarity_score(origin, base_alias)
            if similarity_score > best_score:
                best_score = similarity_score
                best_alias = base_alias
        return best_alias, best_score
    
    def _get_right_base_aliases_in_same_sent(self, doc: Doc, origin: Span) -> list[Span]|None:
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
    
    def _get_right_single_component_base_aliases_in_same_sent(self, doc: Doc, origin: Span) -> list[Span]|None:
        '''
        Get all base aliases in the same sentence with origin[-1].i < base_alias[0].i 
        which are found inside of parentheses and arent excluded by 
        preceding text inside of the parentheses.
        '''
        result = []
        sent_start_end_tuple = doc._.alias_cache.get_sent_start_end_tuple(origin)
        base_alias_in_sent = doc._.alias_cache._sent_alias_map[sent_start_end_tuple]
        parentheses_in_sent = doc._.sent_parentheses_map[sent_start_end_tuple]
        # find out why base_alias arent in sent?
        if base_alias_in_sent:
            for base_alias in base_alias_in_sent:
                if base_alias[0] > origin.start:
                    base_alias_found_in_parentheses = False
                    for parentheses in parentheses_in_sent:
                        # TODO[epic="maybe"] add a max number of parentheses with other base_aliases between origin and result base_alias
                        if base_alias in doc._.parentheses_base_alias_map[parentheses]:
                            base_alias_found_in_parentheses = True
                            if self._has_expression_in_parentheses_before_alias_span(
                                doc,
                                doc[parentheses[0]:parentheses[1]], 
                                doc[base_alias[0]:base_alias[1]],
                                expressions=["together", "and with", "collectively"]
                                ) is False:
                                result.append(doc[base_alias[0]:base_alias[1]])
                    if base_alias_found_in_parentheses is False:
                        logger.warning(f"unhandled case for single compound alias found outside of parentheses, sent:base_alias{sent_start_end_tuple}:{base_alias}")
        return result if result != [] else None

                                # add control flow to add if not present in parentheses 
                                # return result

    
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
    logger.debug(f"calculating similarity_score with parameters: |{very_similar}:{dep_distance}:{span_distance}| |very_simliar:dep_distance:span_distance|")
    total_score = dep_distance_score + span_distance_score + very_similar_score
    logger.debug(f"-> similarity_score: {total_score}")
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
    else:
        logger.debug(f"missing one of the parameters dep,span,similarity_map: {dep_distance, span_distance, similarity_map}")
    return 0
    # start with aliasSetter component which handles base alias to origin mapping based on a given list of Spans
    # using similarity score from filing_nlp.py
    







