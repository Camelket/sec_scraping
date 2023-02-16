from collections import defaultdict
from itertools import product
from spacy.tokens import Token, Span
import logging

logger = logging.getLogger(__name__)


from main.nlp.filing_nlp_constants import (
    ATTRIBUTE_KEY_TO_STRINGKEY,
    DEPENDENCY_ATTRIBUTE_MATCHER_IS_OPTIONAL_FLAG,
    SECUQUANTITY_UNITS,
)
from main.nlp.filing_nlp_utils import (
    extend_token_ent_to_span,
    get_dep_distance_between,
    MatchFormater,
)
from main.nlp.filing_nlp_dateful_relations import DatetimeRelation
from main.nlp.filing_nlp_patterns import (
    AT_CONTEXT_UNIT_TAIL_PATTERNS,
    add_anchor_pattern_to_patterns,
    SECU_SECUQUANTITY_PATTERNS,
    SECU_SOURCE_SECU_SECUQUANTITY_PATTERNS,
    SECU_DATE_RELATION_PATTERNS_FROM_ROOT_VERB,
    SECU_DATE_RELATION_FROM_ROOT_VERB_CONTEXT_PATTERNS,
    SECU_EXERCISE_PRICE_PATTERNS,
    PARENT_VERB_AT_CONTEXT,
)

formater = MatchFormater()

class SourceContext:
    def __init__(self, context: Token, sconj: Token, action: Token):
        self.context = context
        self.sconj = sconj
        self.action = action

class DependencyMatchHelper:
    def _convert_key_to_stringkey(self, key: str) -> str:
        return key.lower() + "_"

    def _has_attributes(self, token: Token, attrs: dict) -> bool:
        if not isinstance(attrs, dict):
            raise TypeError(f"expecting a dict, got: {type(attrs)}")
        if attrs == {}:
            return True
        for attr, value in attrs.items():
            attribute_name = ATTRIBUTE_KEY_TO_STRINGKEY[attr]
            if isinstance(value, dict):
                for modifier, values in value.items():
                    if modifier in ["IN", "NOT_IN"]:
                        if modifier == "IN":
                            if not getattr(token, attribute_name) in values:
                                return False
                            else:
                                pass
                        if modifier == "NOT_IN":
                            if getattr(token, attribute_name) in values:
                                return False
                            else:
                                pass
            else:
                if not getattr(token, attribute_name) == value:
                    return False
        return True

    def check_head(self, token: Token, attrs: dict) -> Token:
        result = []
        if not token.head:
            return result
        if self._has_attributes(token.head, attrs):
            result.append(token.head)
            return result
        else:
            return result

    def check_children(self, token: Token, attrs: dict) -> list[Token]:
        result = []
        if not token.children:
            return result

        for i in filter(
            lambda x: x is not None,
            [
                child if self._has_attributes(child, attrs) else None
                for child in token.children
            ],
        ):
            result.append(i)
        return result

    def check_ancestors(self, token: Token, attrs: dict) -> list[Token]:
        result = []
        if not token.ancestors:
            return result
        for ancestor in token.ancestors:
            if self._has_attributes(ancestor, attrs):
                result.append(ancestor)
        return result

    def check_descendants(self, token: Token, attrs: dict) -> list[Token]:
        result = []
        if not token.subtree:
            return result
        for descendant in token.subtree:
            if token == descendant:
                continue
            if self._has_attributes(descendant, attrs):
                result.append(descendant)
        return result


class DependencyAttributeMatcher:
    """
    Find a specific dependency from an origin token.
    Currently not all of the standard operations of a SpaCy pattern dict
    is covered, covered attributes and modifiers are:
        standard attributes:
            "POS"
            "LOWER"
            "TAG"
            "ORTH"
            "DEP"
            "ENT_TYPE"
            "LEMMA"
        set modifiers:
            "IN"
            "NOT_IN"

    additional entries to the pattern dict:
        "IS_OPTIONAL": bool (wether a match on this token is optional)

    example pattern dict with anchor:
        [
            {
                "RIGHT_ID": "anchor",
                "TOKEN": origin_token
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">>",
                "RIGHT_ID": "aux_verb",
                "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}},
            }
        ]

    """

    # need to resolve what happens if an optional token isnt found and what happens to its children
    dep_getter = DependencyMatchHelper()
    DEPENDENCY_OPS = {
        "<<": dep_getter.check_ancestors,
        "<": dep_getter.check_head,
        ">": dep_getter.check_children,
        ">>": dep_getter.check_descendants,
    }

    def run_patterns_for_match(self, complete_patterns: list[list[dict]]) -> list[list]:
        matches = []
        for pattern in complete_patterns:
            matches += self.get_candidate_matches(pattern)
        return matches if matches != [] else None
    
    def get_aux_verbs(self, token: Token) -> list[Token] | None:
        """get the child aux verbs of token."""
        candidate_matches = self.get_candidate_matches(
            [
                {"RIGHT_ID": "anchor", "TOKEN": token},
                {
                    "LEFT_ID": "anchor",
                    "REL_OP": ">>",
                    "RIGHT_ID": "aux_verb",
                    "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}},
                },
            ]
        )
        # logger.debug(f"canddiate_matches for aux_verb: {candidate_matches}")
        if candidate_matches != []:
            aux_verbs = []
            for match in candidate_matches:
                for aux in self._get_tokens_with_tag_from_match_tuples(
                    match, "aux_verb"
                ):
                    if aux is None:
                        break
                    aux_verbs.append(aux)
            return aux_verbs if aux_verbs != [] else None
        return None

    def get_parent_verb(self, token: Token) -> Token | None:
        """get the closest parent verb in the dependency tree of token."""
        candidate_matches = self.get_candidate_matches(
            [
                {"RIGHT_ID": "anchor", "TOKEN": token},
                {
                    "LEFT_ID": "anchor",
                    "REL_OP": "<<",
                    "RIGHT_ID": "parent_verb",
                    "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}},
                    "IS_OPTIONAL": False,
                },
            ]
        )
        if candidate_matches != []:
            if len(candidate_matches) > 1:
                verbs = []
                for candidate_match in candidate_matches:
                    for entry in candidate_match:
                        candidate_token, right_id = entry
                        if right_id == "parent_verb":
                            verbs.append(candidate_token)
                dep_distances = [None] * len(verbs)
                for idx, verb in enumerate(verbs):
                    dep_distances[idx] = get_dep_distance_between(token, verb)
                min_idx = dep_distances.index(min(dep_distances))
                return verbs[min_idx]
            else:
                for entry in candidate_matches[0]:
                    candidate_token, right_id = entry
                    if right_id == "parent_verb":
                        return candidate_token
        return None

    def get_root_verb(self, token: Token) -> Token | None:
        """get the root parent verb of the dependency tree of token."""
        candidate_matches = self.get_candidate_matches(
            [
                {"RIGHT_ID": "anchor", "TOKEN": token},
                {
                    "LEFT_ID": "anchor",
                    "REL_OP": "<<",
                    "RIGHT_ID": "root_verb",
                    "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}},
                    "IS_OPTIONAL": False,
                },
            ]
        )
        # logger.debug(f"canddiate_matches for root_verb: {candidate_matches}")
        if candidate_matches != []:
            if len(candidate_matches) > 1:
                verbs = []
                for candidate_match in candidate_matches:
                    for entry in candidate_match:
                        candidate_token, right_id = entry
                        if right_id == "root_verb":
                            verbs.append(candidate_token)
                dep_distances = [None] * len(verbs)
                for idx, verb in enumerate(verbs):
                    dep_distances[idx] = get_dep_distance_between(token, verb)
                max_idx = dep_distances.index(max(dep_distances))
                return verbs[max_idx]
            else:
                for entry in candidate_matches[0]:
                    candidate_token, right_id = entry
                    if right_id == "root_verb":
                        return candidate_token
        return None
    
    def get_date_relation(self, token: Token, relation_patterns: list[list[dict]], context_patterns: list[list[dict]]=None) -> dict:
        '''
        fetch the datetime relation associated with token, currently done through the root verb of token.
        Must be supplied by relation_patterns, will raise ValueError otherwise.

        The anchor for the relation pattern will be the root_verb and 
        for the context pattern the root of the DATE entity.
        Patterns should be like the patterns for spacy dependency parsers, expect for:
        omit the root entry, there must be a dict with LEFT_ID = "anchor" for each pattern.
        eg: 
            [
                [   
                    {
                        "LEFT_ID": "anchor",
                        "REL_OP": ">>",
                        "RIGHT_ID": "prep_phrase_start",
                        "RIGHT_ATTRS": {"DEP": "prep", "LOWER":  "as"}
                    },
                    {
                        "LEFT_ID": "prep_phrase_start",
                        "REL_OP": ">",
                        "RIGHT_ID": "second_prep",
                        "RIGHT_ATTRS": {"DEP": "prep", "LOWER": "of"}
                    },
                ],
                [
                    ...next pattern would go here...
                ]
            ]
        '''
        return self._get_date_relation(token, relation_patterns, context_patterns)
    
    def _get_date_relation(self, token: Token, relation_patterns: list[list[dict]], context_patterns: list[list[dict]]) -> dict:
        if relation_patterns is None:
            raise ValueError("get_date_relation must be supplied by relation_patterns got None.")
        if context_patterns is None:
            logger.debug(f"Ensure you didnt want to supply a context_patterns parameter when calling 'get_date_relation'.")
        unformatted_dates = []
        unformatted_dates += self._get_date_relation_through_root_verb(token, relation_patterns)
        # TODO: think and write how i will need the timedelta part of this and how it could be used
        dates = {"datetime": [], "timedelta": []}
        contexts = []
        for unformatted in unformatted_dates:
            if context_patterns:
                context = self._get_context_for_date_relation(unformatted.root, context_patterns)
            else:
                context = dict()
            date = formater.coerce_tokens_to_datetime(unformatted)
            if date:
                relation = DatetimeRelation(unformatted, date, context)
                if relation not in dates["datetime"]:
                    dates["datetime"].append(relation)
            else:
                date = formater.coerce_tokens_to_timedelta(unformatted)
                if date:
                    dates["timedelta"].append({"date": date, "context": context})
        return dates
        # TODO: fix the implementation of date_relation in a SECU context since we should only have one date in a specific context, right?
        # we should only have one date in a quantity context otherwise we cant be certain how they relate ?
        # options: 1) create a set of the datetimerelations and check if len <= 1 when assigning to quantity
    
    def _get_date_relation_through_root_verb(self, token: Token, relation_patterns: list[list[dict]]=None) -> list[Span]:
        if token.sent.root == token:
            root_verb = token
        else:
            root_verb = self.get_root_verb(token)
        result = []
        if root_verb:
            anchor_pattern = self._get_anchor_pattern(root_verb)
            date_relation_root_verb_patterns = add_anchor_pattern_to_patterns(
                anchor_pattern,
                relation_patterns
            )
            for pattern in date_relation_root_verb_patterns:
                candidate_matches = self.get_candidate_matches(pattern)
                for match in candidate_matches:
                    for rel in self._get_tokens_with_tag_from_match_tuples(
                        match, "date_start"
                    ):
                        if rel is None:
                            break
                        result.append(extend_token_ent_to_span(rel, rel.doc))
        return result

    def _get_context_for_date_relation(self, date_root_token: Token, context_patterns: list[list[dict]]) -> dict:
        """
        Returns:
            {"original": list[tuple], "formatted": dict[str, dict]}
        """
        logger.debug(
            f"getting context for date relation with root token: {date_root_token}"
        )
        date_root_pattern = self._get_anchor_pattern(date_root_token)
        complete_patterns = add_anchor_pattern_to_patterns(
            date_root_pattern,
            context_patterns
        )
        matches = self.run_patterns_for_match(complete_patterns)
        logger.debug(f"matches for date_relation context: {matches}")
        # TODO: Filter the context matches to eliminate duplicates (or a set that is fully contained within another)
        if not matches:
            return None
        merged_set = self._merge_wanted_tags_into_set(matches)
        formatted_tags = self._format_wanted_tags_from_set_as_dict(merged_set)
        return {"original": merged_set, "formatted": formatted_tags}

    def _merge_wanted_tags_into_set(
        self,
        matches: list[list[tuple]],
        wanted_tags: list[str] = [
            "prep1",
            "prep2",
            "prep_end",
            "adj_to_aux",
            "adj_to_verb",
            "aux_verb",
            "verb",
        ],
    ) -> list[tuple]:
        merged_set = set()
        for match in matches:
            for entry in match:
                if entry[1] in wanted_tags:
                    if entry not in merged_set:
                        merged_set.add(entry)
        return merged_set if len(merged_set) != 0 else None

    def _format_wanted_tags_from_set_as_dict(
        self,
        wanted_tags_set,
        default_groups: list[str] = ["prep", "aux_verb", "adj", "verb"],
        group_mapping: dict[str, str] = {
            "prep1": "prep",
            "prep2": "prep",
            "prep_end": "prep",
            "verb": "verb",
            "aux_verb": "aux_verb",
            "adj_to_aux": "adj",
            "adj_to_verb": "adj",
        },
        sort_order: dict[str, list] = {"prep": ["prep_end", "prep2", "prep1"]},
    ) -> dict[str, dict]:
        finished_grouping = {k: [] for k in default_groups}
        unsorted_grouping = defaultdict(list)
        # wanted_tags_set = self._merge_wanted_tags_into_set(matches)
        def _index_for_sort_group(element, sort_group):
            idx = -1
            try:
                idx = sort_group.index(element)
            except ValueError:
                return len(sort_group) + 1
            else:
                return idx

        for entry in wanted_tags_set:
            parent_tag = group_mapping.get(entry[1], None)
            if parent_tag:
                unsorted_grouping[parent_tag].append(entry)
        for key in unsorted_grouping.keys():
            if sort_order.get(key, None):
                finished_group = [
                    i[0]
                    for i in sorted(
                        unsorted_grouping[key],
                        key=lambda x: _index_for_sort_group(x[1], sort_order.get(key)),
                    )
                ]
                finished_grouping[key] = finished_group
            else:
                finished_grouping[key] = [i[0] for i in unsorted_grouping[key]]
        return finished_grouping

    def _get_tokens_with_tag_from_match_tuples(
        self, match: list[tuple[Token, str]], tag: str
    ) -> Token | None:
        if not match:
            yield None
        if len(match) == 0:
            yield None
        for entry in match:
            token, right_id = entry
            if right_id == tag:
                yield token
        yield None

    def _get_anchor_pattern(self, anchor: Token):
        if not isinstance(anchor, Token):
            raise TypeError(
                f"anchor must be a Token to correctly match, got {type(anchor)}"
            )
        return [{"RIGHT_ID": "anchor", "TOKEN": anchor}]

    def _get_matches(self, dependant: Token, attr: list[dict]):
        # logger.debug(f"getting matches for {dependant} and attr: {attr}")
        rel_op = attr["REL_OP"]
        matches = self.DEPENDENCY_OPS[rel_op](dependant, attr["RIGHT_ATTRS"])
        # logger.debug(f"got matches: {matches}; for (dependant: {dependant}; attr: {attr})")
        return matches

    def _get_attr_tree(self, attrs: list[dict]):
        right_id_to_idx = {}
        for idx, attr in enumerate(attrs):
            # logger.debug(f"currently working on attr: {attr}")
            # logger.debug(f"with idx: {idx}")
            right_id_to_idx[attr["RIGHT_ID"]] = idx
        tree = defaultdict(list)
        root = None
        for idx, attr in enumerate(attrs):
            if "REL_OP" in attr:
                tree[right_id_to_idx[attr["LEFT_ID"]]].append((attr["REL_OP"], idx))
            else:
                root = idx
        return tree, root

    def _conditions_optional(self, attr: dict):
        booleans = []
        for key, value in attr["RIGHT_ATTRS"].items():
            if key == "OP":
                if value == "?":
                    booleans.append(True)
            else:
                booleans.append(False)
        return any(booleans)

    def get_candidate_matches(self, attrs: list[dict]) -> list[list[tuple[Token, str]]]:
        tree, root = self._get_attr_tree(attrs)
        # logger.debug(f"tree: {tree}; root: {root}")
        candidates_cache = {k: [] for k, _ in enumerate(attrs)}
        root_attr = None
        candidates_cache[root].append((attrs[root]["TOKEN"], attrs[root]["RIGHT_ID"]))
        # logger.debug(f"inital candidates_cache: {candidates_cache}")
        def resolve_matching_from_node(node, candidates_cache, tree):
            if not tree.get(node):
                return
            else:
                children = tree.get(node)
                # print(f"children: {children}")
                for child in children:
                    rel_op, child_idx = child
                    # print(f"\t rel_op, child_idx: {rel_op}, {child_idx}")
                    for parent, parent_right_id in candidates_cache[node]:
                        # print(f"\t\t parent, parent_right_id: {parent}, {parent_right_id}")
                        matches = self._get_matches(parent, attrs[child_idx])
                        # print("\t\t ",matches)
                        is_optional = attrs[child_idx].get("IS_OPTIONAL", None)
                        right_id = attrs[child_idx]["RIGHT_ID"]
                        if is_optional and not matches:
                            candidates_cache[child_idx].append(
                                (
                                    DEPENDENCY_ATTRIBUTE_MATCHER_IS_OPTIONAL_FLAG,
                                    right_id,
                                )
                            )
                        if matches:
                            for match in matches:
                                candidates_cache[child_idx].append((match, right_id))
                        else:
                            if is_optional:
                                continue
                            # continue and add placeholder if matching this token is optional else return -1
                            # logger.debug(f"breaking out and returning -1")
                            return -1
                    # logger.debug(f"continuing with recursion.")
                    resolve_matching_from_node(child_idx, candidates_cache, tree)

        found_matches = resolve_matching_from_node(root, candidates_cache, tree)
        # logger.debug(f"final candidates_cache: {candidates_cache}")
        if found_matches == -1:
            return []
        else:

            def build_matches_from_candidates(candidates_cache, tree):
                unprocessed_matches = [
                    i for i in product(*[candidates_cache[k] for k in candidates_cache])
                ]
                processed_matches = []
                for unprocessed in unprocessed_matches:
                    processed_entry = []
                    for entry in unprocessed:
                        if isinstance(
                            entry[0],
                            type(DEPENDENCY_ATTRIBUTE_MATCHER_IS_OPTIONAL_FLAG),
                        ):
                            if (
                                entry[0]
                                == DEPENDENCY_ATTRIBUTE_MATCHER_IS_OPTIONAL_FLAG
                            ):
                                pass
                        else:
                            processed_entry.append(entry)
                    processed_matches.append(processed_entry)
                return processed_matches

            return build_matches_from_candidates(candidates_cache, tree)

class ContractDependencyAttributeMatcher(DependencyAttributeMatcher):
    def __init__(self):
        super().__init__()
    
    def get_date_relation(self, token: Token):
        raise NotImplementedError()
    
    def get_actions(self, token: Token):
        raise NotImplementedError()
    
    def get_subjects(self, token: Token):
        raise NotImplementedError()
    
    def get_scope(self, token: Token):
        # THIS IS WRONG APPROACH TO SCOPE
        # scope root is the parent verb of token and the scope is the left and rightmost children of scope root ()
        scope_root = None
        for ancestor in token.ancestors:
            if ancestor.pos_ == "VERB":
                scope_root = ancestor
                break
        if not scope_root:
            if token.pos_ == "VERB":
                scope_root = token
        if not scope_root:
            return None
        else:
            left_boundary = token.doc[min([d.i for d in scope_root.descendants])]
            right_boundary = token.doc[max([d.i for d in scope_root.descendants])]
            return (left_boundary, scope_root, right_boundary)
        

class SecurityDependencyAttributeMatcher(DependencyAttributeMatcher):
    def __init__(self):
        super().__init__()
    
    def _get_at_context(self, parent_verb: Token):
        # example: we sold 10000 shares of our common stock [at a purchase price of 2 $ per share] will match things in bracket.
        core_anchor_pattern = self._get_anchor_pattern(parent_verb)
        complete_core_patterns = add_anchor_pattern_to_patterns(
            core_anchor_pattern, PARENT_VERB_AT_CONTEXT
        )
        result = []
        matches = self.run_patterns_for_match(complete_core_patterns)
        if matches:
            for match in matches:
                result.append(match)
        for r in result:
            tail_anchor_pattern = self._get_anchor_pattern(r[-2][0])
            complete_tail_patterns = add_anchor_pattern_to_patterns(
                tail_anchor_pattern, AT_CONTEXT_UNIT_TAIL_PATTERNS
            )
            tail_matches = self.run_patterns_for_match(complete_tail_patterns)
            if tail_matches:
                if len(tail_matches) == 1:
                    for x in tail_matches[0][1:]:
                        r.append(x)
        return result

    def get_date_relation(self, token: Token):
        return self._get_date_relation(token, SECU_DATE_RELATION_PATTERNS_FROM_ROOT_VERB, SECU_DATE_RELATION_FROM_ROOT_VERB_CONTEXT_PATTERNS)

    def get_exercise_price(self, secu: Token) -> tuple[float, str]|None:
        prices = self._get_exercise_price(secu)
        if not prices:
            return None
        if len(prices) > 1:
            # NOTE: handling price ranges in a security class "warrants have exercise prices between 10 and 20" ect. I mostlikely cant link it back to a specific security i will just ignore it for now
            logger.info(
                "Multiple exercise prices found, current case only handles one. pretending we didnt find any."
            )
            logger.info(f"actual prices found were: {prices}")
            return None
        else:
            return (
                formater.quantity_string_to_float(prices[0][0].text),
                prices[0][1].text,
            )

    def _get_exercise_price(self, secu: Token) -> list[tuple[float, str]]:
        anchor_pattern = self._get_anchor_pattern(secu)
        complete_pattern = add_anchor_pattern_to_patterns(
            anchor_pattern, SECU_EXERCISE_PRICE_PATTERNS
        )
        longest_match = []
        for pattern in complete_pattern:
            candidate_matches = self.get_candidate_matches(pattern)
            if candidate_matches:
                for match in candidate_matches:
                    if len(match) > len(longest_match):
                        longest_match = match
        result = []
        if len(longest_match) > 0:
            amount, symbol = None, None
            for currency_amount in self._get_tokens_with_tag_from_match_tuples(
                longest_match, "pobj_CD"
            ):
                for currency_symbol in self._get_tokens_with_tag_from_match_tuples(
                    longest_match, "currency_symbol"
                ):
                    if currency_symbol is None:
                        break
                    symbol = currency_symbol
                if currency_amount is None:
                    break
                amount = currency_amount
            if amount is None or symbol is None:
                logger.debug(
                    f"couldnt extract the amount:{amount} or the symbol:{symbol} from the match: {longest_match}"
                )
            result.append((amount, symbol))
        return result

    
    def _get_source_secu_context_through_secuquantity(
        self, token: Token
    ) -> SourceContext | None:
        '''
        currently not usable as the context is so variable i need a different approach than below.
        '''
        # context, sconj, action pattern: secuquantity -> unit_word ->> context_word -> upon -> (the) -> action_word -> of -> pobj SECU
        # TODO[epic=maybe]: make this pattern more specific by adding the source into the pattern since we look for context from the source secu anyhow 
        if token.ent_type_ != "SECUQUANTITY":
            return None

        context_sconj_action = self._get_context_sconj_action_from_secuquantity(token)
        if context_sconj_action:
            if len(context_sconj_action) > 1:
                logger.warning(
                    f"more than one context_sconj_action found for {token} while trying to get source_secu. context_sconj_action found: {context_sconj_action}"
                )
                logger.warning(f"currently unhandled so we return None")
                return None
            if len(context_sconj_action) == 1:
                match = context_sconj_action[0]["match"]
                source_context = SourceContext(
                    context=match["context_token"],
                    sconj=match["sconj_token"],
                    action=match["action_token"],
                    source=match["source_secu"],
                )
                return source_context
        return None

    def _get_context_sconj_action_from_secuquantity(self, secuquantity: Token):
        '''
        example sentence: Includes 7,111,112 shares of common stock that may be issued upon the exercise of warrants.
        secuquantity is: 7,111,112
        context would be: context="issued", sconj="upon", action="exercise", source_secu="warrants"
        '''
        result = []
        if secuquantity.head:
            if secuquantity.head.lower_ in SECUQUANTITY_UNITS:
                anchor = secuquantity.head
                pattern = [
                    {"RIGHT_ID": "anchor", "TOKEN": anchor},
                    {
                        "LEFT_ID": "anchor",
                        "REL_OP": ">",
                        "RIGHT_ID": "context_token",
                        "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "ADJ"]}},
                    },
                    {
                        "LEFT_ID": "context_token",
                        "REL_OP": ">",
                        "RIGHT_ID": "sconj_token",
                        "RIGHT_ATTRS": {"POS": "SCONJ"},
                    },
                    {
                        "LEFT_ID": "sconj_token",
                        "REL_OP": ">",
                        "RIGHT_ID": "action_token",
                        "RIGHT_ATTRS": {"POS": "NOUN"},
                    },
                    {
                        "LEFT_ID": "action_token",
                        "REL_OP": ">",
                        "RIGHT_ID": "of",
                        "RIGHT_ATTRS": {"LOWER": "of"},
                    },
                    {
                        "LEFT_ID": "of",
                        "REL_OP": ">",
                        "RIGHT_ID": "source_secu",
                        "RIGHT_ATTRS": {"ENT_TYPE": "SECU"},
                    },
                ]
                candidate_matches = self.get_candidate_matches(pattern)
                if len(candidate_matches) > 0:
                    for match in candidate_matches:
                        result.append(
                            {
                                "match": {k[1]: k[0] for k in match},
                                "sent": secuquantity.sent,
                            }
                        )
        return result if len(result) != 0 else None

    def get_quantities(self, token: Token) -> list[Token] | None:
        anchor_pattern = self._get_anchor_pattern(token)
        patterns = add_anchor_pattern_to_patterns(
            anchor_pattern, SECU_SECUQUANTITY_PATTERNS
        )
        result = []
        for pattern in patterns:
            # logger.debug(f"currently working on pattern {pattern}")
            candidate_matches = self.get_candidate_matches(pattern)
            if len(candidate_matches) > 0:
                for match in candidate_matches:
                    quant = None
                    for entry in match:
                        candidate_token, right_id = entry
                        if right_id == "secuquantity":
                            quant = candidate_token
                    if quant is not None:
                        result.append(quant)
        return result if result != [] else None
    
    def get_possible_source_quantities(self, token: Token) -> list[Token]:
        anchor_pattern = self._get_anchor_pattern(token)
        patterns = add_anchor_pattern_to_patterns(
            anchor_pattern, SECU_SOURCE_SECU_SECUQUANTITY_PATTERNS
        )
        result = []
        for pattern in patterns:
            # logger.debug(f"currently working on pattern {pattern}")
            candidate_matches = self.get_candidate_matches(pattern)
            if len(candidate_matches) > 0:
                for match in candidate_matches:
                    quant = None
                    for entry in match:
                        candidate_token, right_id = entry
                        if right_id == "secuquantity":
                            quant = candidate_token
                    if quant is not None:
                        result.append(quant)
        return result if result != [] else None


    
