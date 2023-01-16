from datetime import timedelta
import re
from spacy.tokens import Token, Span, Doc
import pandas as pd
import logging
import string
from typing import Iterable, Set

logger = logging.getLogger(__name__)

def filter_matches(matches):
    """works as spacy.util.filter_spans but for matches"""
    if len(matches) <= 1:
        return matches
    # logger.debug(f"pre filter matches: {[m for m in matches]}")
    get_sort_key = lambda match: (match[2] - match[1], -match[1])
    sorted_matches = sorted(matches, key=get_sort_key, reverse=True)
    result = []
    seen_tokens: Set[int] = set()
    for match in sorted_matches:
        # Check for end - 1 here because boundaries are inclusive
        if match[1] not in seen_tokens and match[2] - 1 not in seen_tokens:
            result.append(match)
            seen_tokens.update(range(match[1], match[2]))
    result = sorted(result, key=lambda match: match[1])
    return result


def filter_dep_matches(matches):
    """take the longest of the dep matches with same start token, discard rest"""
    if len(matches) <= 1:
        return matches
    len_map = {}
    result_map = {}
    # logger.debug(f"filtering dep matches: {matches}")
    for match in matches:
        if match[1][0] not in len_map.keys():
            len_map[match[1][0]] = len(match[1])
            result_map[match[1][0]] = match
        else:
            current_len = len(match[1])
            if current_len <= len_map[match[1][0]]:
                pass
            else:
                len_map[match[1][0]] = len(match[1])
                result_map[match[1][0]] = match
    return [v for _, v in result_map.items()]

def _set_extension(cls, name, kwargs):
    if not cls.has_extension(name):
        cls.set_extension(name, **kwargs)

def get_none_alias_ent_type_tuples(doc: Doc, ent_type: str) -> list[tuple[int, int]]:
    result = []
    for ent in doc.ents:
        if ent.label_ == ent_type:
            if ent._.is_alias:
                continue
            else:
                result.append((ent.start, ent.end))
    return result

def get_none_alias_ent_type_spans(doc: Doc, ent_type: str) -> list[Span]:
        result = []
        for ent in doc.ents:
            if ent.label_ == ent_type:
                if ent._.is_alias:
                    continue
                else:
                    result.append(ent)
        return result

def extend_token_ent_to_span(token: Token, doc: Doc) -> list[Token]:
    span_tokens = [token]
    logger.debug(
        f"extending ent span to surrounding for origin token: {token, token.i}"
    )
    for i in range(token.i - 1, 0, -1):
        if doc[i].ent_type_ == token.ent_type_:
            span_tokens.insert(0, doc[i])
            logger.debug(f"found preceding token matching ent: {doc[i]}")
        else:
            break
    for i in range(token.i + 1, 10000000, 1):
        if doc[i].ent_type_ == token.ent_type_:
            span_tokens.append(doc[i])
            logger.debug(f"found following token matching ent: {doc[i]}")
        else:
            break
    logger.debug(f"making tokens to span: {span_tokens}")
    if len(span_tokens) <= 1:
        span = Span(doc, token.i, token.i + 1, label=token.ent_type_)
        logger.debug(f"returning span: {span, span.text}")
        return span
    span = Span(doc, span_tokens[0].i, span_tokens[-1].i + 1, label=token.ent_type_)
    logger.debug(f"span: {span, span.text}")
    return span

def BFS_non_recursive(origin, target) -> list:
    """
    Breadth First Search algorithm

    Returns:
        list: list of nodes in the path from origin to target
        None: if no path is found
    """
    path = []
    queue = [[origin]]
    node = origin
    visited = set()
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == target:
            return path
        visited.add(node)
        adjacents = (list(node.children) if node.children else []) + (
            list(node.ancestors) if node.ancestors else []
        )
        for adjacent in adjacents:
            if adjacent in visited:
                continue
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)


def get_dep_distance_between(origin: Token, target: Token) -> int:
    """
    get the distance between two tokens in a spaCy dependency tree.

    Returns:
        int: distance between origin and target
        None: if origin and target arent part of the same tree
    """
    is_in_same_tree = False
    is_in_same_sent = False
    if origin.sent == target.sent:
        is_in_same_sent = True
        if origin.is_ancestor(target):
            is_in_same_tree = True
            start = origin
            end = target
        elif target.is_ancestor(origin):
            is_in_same_tree = True
            start = target
            end = origin
    if is_in_same_sent:
        if is_in_same_tree:
            path = BFS_non_recursive(start, end)
            return len(path)
        else:
            common_ancestor = None
            for token in origin.ancestors:
                if token.is_ancestor(target):
                    common_ancestor = token
                    break
            else:
                return None
            return len(BFS_non_recursive(common_ancestor, target)) + len(BFS_non_recursive(common_ancestor, origin))
    else:
        return None


def get_dep_distance_between_spans(origin: Span, target: Span) -> int:
    """
    get the distance between two spans (counting from their root) in a spaCy dependency tree.

    Returns:
        int: distance between origin and target
        None: if origin and target arent part of the same tree
    """
    distance = get_dep_distance_between(origin.root, target.root)
    return distance


def get_span_distance(secu: Iterable[Token], alias: Iterable[Token] | Span) -> int:
    if secu[0].i > alias[0].i:
        first = alias
        second = secu
    else:
        first = secu
        second = alias
    mean_distance = ((second[0].i - first[-1].i) + (second[-1].i - first[0].i)) / 2
    return mean_distance

def int_to_roman(input):
    """ Convert an integer to a Roman numeral. """
    if not isinstance(input, type(1)):
        raise TypeError(f"expected integer, got {type(input)}")
    if not (0 < input < 4000):
        print(input)
        raise ValueError("Argument must be between 1 and 3999")
    ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
    nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
    result = []
    for i in range(len(ints)):
        count = int(input / ints[i])
        result.append(nums[i] * count)
        input -= ints[i] * count
    return ''.join(result).lower()

def roman_list():
    return ["(" + int_to_roman(i)+")" for i in range(1, 50)]

def alphabetic_list():
    return ["(" + letter +")" for letter in list(string.ascii_lowercase)]

def numeric_list():
    return ["(" + str(number) + ")" for number in range(150)]

class WordToNumberConverter():
    numbers_map = {
        "one": 1,
        "first": 1,
        "two": 2,
        "second": 2,
        "three": 3,
        "third": 3,
        "four": 4,
        "fourth": 4,
        "five": 5,
        "fifth": 5,
        "six": 6,
        "sixth": 6,
        "seven": 7,
        "seventh": 7,
        "eight": 8,
        "eighth": 8,
        "nine": 9,
        "ninth": 9,
        "ten": 10,
        "tenth": 10,
        "eleven": 11,
        "eleventh": 11,
        "twelve": 12,
        "twelfth": 12
    }
    timedelta_map = {
        "day": timedelta(days=1),
        "week": timedelta(weeks=1),
        "month": timedelta(days=30),
        "year": timedelta(days=365.25),
        "days": timedelta(days=1),
        "weeks": timedelta(weeks=1),
        "months": timedelta(days=30),
        "years": timedelta(days=365.25)
    }

    def convert_str_number_name(self, s: str):
        if self.numbers_map.get(s):
            return self.numbers_map[s]
        return None
    
    def convert_str_timedelta(self, s: str):
        if self.timedelta_map.get(s):
            return self.timedelta_map[s]
        return None

    def convert_spacy_token(self, token: Token):
        number = self.convert_str_number_name(token.lower_)
        if number:
            return number
        else:
            tdelta = self.convert_str_timedelta(token.lower_)
            if tdelta:
                return tdelta
        return None

class MatchFormater:
    def __init__(self):
        self.w2n = WordToNumberConverter()
    
    def parse_american_number(self, text):
        if text is None:
            return None
        text = text.strip()
        if text == "":
            return None
        if not isinstance(text, str):
            raise TypeError(f"{self} is expecting a string got: {type(text)}, value: {text}")
        # assure we only have a pattern of ,xxx. and not .xxx,
        matches = re.findall("[,.]", text)
        if matches:
            symbols = list(matches)
        else:
            return text
        if symbols:
            comma_before_dot, comma_after_dot = True, False
            previous_symbol, next_symbol = None, None
            if len(symbols) == 1:
                if symbols[0] == ",":
                    number_parts = text.split(",")
                    right = number_parts[1]
                    if (len(right) == 1) or (len(right) == 2):
                            # this is wrong format for an american number but mostlikely a european decimal
                            return text.replace(",", ".")
                    return text.replace(",", "")
                else:
                    #regular decimal notation which should be easily converted by float()
                    return text
            else:
                for idx, symbol in enumerate(symbols):
                    if (idx == 0):
                        pass
                    else:
                        if previous_symbol == "," and symbol == ".":
                            comma_before_dot = True
                        if previous_symbol == "." and symbol == ",":
                            comma_after_dot = True
                    previous_symbol = symbol
                    if (comma_after_dot is True) and (comma_before_dot is True):
                        return None
                if comma_after_dot is True:
                    # this isnt the number format we are trying to parse
                    return None
                if comma_before_dot is True:
                    return text.replace(",", "")

    def parse_number(self, text):
        # WILL BE DEPRECATED MOSTLIKELY
        # REPLACED BY parse_american_number
        '''from https://github.com/hayj/SystemTools/blob/master/systemtools/number.py'''
        try:
            # First we return None if we don't have something in the text:
            if text is None:
                return None
            if isinstance(text, int) or isinstance(text, float):
                return text
            text = text.strip()
            if text == "":
                return None
            # Next we get the first "[0-9,. ]+":
            n = re.search("-?[0-9]*([,. ]?[0-9]+)+", text).group(0)
            n = n.strip()
            if not re.match(".*[0-9]+.*", text):
                return None
            # Then we cut to keep only 2 symbols:
            while " " in n and "," in n and "." in n:
                index = max(n.rfind(','), n.rfind(' '), n.rfind('.'))
                n = n[0:index]
            n = n.strip()
            # We count the number of symbols:
            symbolsCount = 0
            for current in [" ", ",", "."]:
                if current in n:
                    symbolsCount += 1
            # If we don't have any symbol, we do nothing:
            if symbolsCount == 0:
                pass
            # With one symbol:
            elif symbolsCount == 1:
                # If this is a space, we just remove all:
                if " " in n:
                    n = n.replace(" ", "")
                # Else we set it as a "." if one occurence, or remove it:
                else:
                    theSymbol = "," if "," in n else "."
                    if n.count(theSymbol) > 1:
                        n = n.replace(theSymbol, "")
                    else:
                        n = n.replace(theSymbol, ".")
            else:
                rightSymbolIndex = max(n.rfind(','), n.rfind(' '), n.rfind('.'))
                rightSymbol = n[rightSymbolIndex:rightSymbolIndex+1]
                if rightSymbol == " ":
                    return self.parse_number(n.replace(" ", "_"))
                n = n.replace(rightSymbol, "R")
                leftSymbolIndex = max(n.rfind(','), n.rfind(' '), n.rfind('.'))
                leftSymbol = n[leftSymbolIndex:leftSymbolIndex+1]
                n = n.replace(leftSymbol, "L")
                n = n.replace("L", "")
                n = n.replace("R", ".")
            n = float(n)
            if n.is_integer():
                return int(n)
            else:
                return n
        except:
            pass
        return None

    def quantity_string_to_float(self, quantity_string: str):
        multiplier = 1
        digits = re.findall("[0-9.,]+", quantity_string)
        parsed_number = self.parse_american_number("".join(digits))
        if parsed_number is None:
            parsed_number = self.w2n.convert_str_number_name(quantity_string)
        if parsed_number is not None:
            amount_float = float(parsed_number)
            if re.search(re.compile("million(?:s)?", re.I), quantity_string):
                multiplier = 1000000
            if re.search(re.compile("billion(?:s)?", re.I), quantity_string):
                multiplier = 1000000000
            return amount_float*multiplier
        logger.warning(f"failed to convert quantity_string: {quantity_string} to float")
        return None
    
    def coerce_tokens_to_datetime(self, tokens: list[Token]|Span):
        try:
            date = pd.to_datetime("".join([i.text_with_ws for i in tokens]))
        except Exception as e:
            logger.debug(e, exc_info=True)
            return None
        else:
            return date

    
    def coerce_tokens_to_timedelta(self, tokens: list[Token]):
        multipliers = []
        timdelta_ = None
        current_idxs = []
        converted = []
        for idx, token in enumerate(tokens):
            w2n_conversion = self.w2n.convert_spacy_token(token)
            if w2n_conversion:
                if isinstance(w2n_conversion, timedelta):
                    timedelta_ = w2n_conversion
                    current_idxs.append(idx)
                    for prev_idx in range(idx-1, -1, -1):
                        prev_token = tokens[prev_idx]
                        if prev_token.is_punct:
                            continue
                        try:
                            current_idxs.append(prev_idx)
                            number = int(prev_token.lower_)
                            multipliers.append(number)
                        except ValueError:
                            number = self.w2n.convert_spacy_token(prev_token)
                            if isinstance(number, int):
                                multipliers.append(number)
                            else:
                                break
                    if multipliers != [] and timedelta_ is not None:
                        if len(multipliers) > 1:
                            raise NotImplementedError(f"multiple numbers before a timedelta token arent handled yet")
                        converted.append((multipliers[0]*timedelta_, current_idxs))
                timedelta_ = None
                multipliers = []
                current_idxs = []                    
        return converted if converted != [] else None
