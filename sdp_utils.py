from spacy.tokens import Span, Token, Doc, DocBin


def get_sdp_path(doc, head, tail, lca_matrix):
    lca = lca_matrix[head, tail]
    current_node = doc[head]
    head_path = [current_node]
    if lca != -1: 
        if lca != head: 
            while current_node.head.i != lca:
                current_node = current_node.head
                head_path.append(current_node)
            head_path.append(current_node.head)
    current_node = doc[tail]
    tail_path = [current_node]
    if lca != -1: 
        if lca != tail: 
            while current_node.head.i != lca:
                current_node = current_node.head
                tail_path.append(current_node)
            tail_path.append(current_node.head)
    return head_path + tail_path[::-1][1:]

from main.nlp.filing_nlp_utils import extend_token_ent_to_span
from spacy.tokens import Span, Token
seen_sents = set()

def default_ent_filter(ents):
    f = list(filter(lambda x: x.label_ not in ["MONEY", "ORDINAL", "CARDINAL"], ents))
    for ent1 in f:
        for ent2 in f:
            if ent1 == ent2:
                continue
            else:
                yield ent1, ent2
                # if ent1.start < ent2.start:
                #     yield ent1, ent2
                # else:
                #     yield ent2, ent1

def distance_ent_filter(ents, max_distance: int=150):
    valid_ent_pairs = set()
    ent_to_idx_map = {x: ent for ent in ents for x in range(ent.start, ent.end + 1, 1)}
    for ent in ents:
        max_left = ent.start - max_distance
        max_left = max_left if max_left > 0 else 0
        left = max_left if max_left > ent.sent.start else ent.sent.start
        max_right = ent.end + max_distance
        right = max_right if max_right < ent.sent.end else ent.sent.end
        for i in range(left, ent.start):
            try:
                l_ent = ent_to_idx_map[i]
            except KeyError:
                continue
            if l_ent:
                if (l_ent, ent) not in valid_ent_pairs:
                    valid_ent_pairs.add((l_ent, ent))
                    yield l_ent, ent
        for ri in range(ent.end, right+1):
            try:
                r_ent = ent_to_idx_map[ri]
            except KeyError:
                continue
            if r_ent:
                if (ent, r_ent) not in valid_ent_pairs:
                    valid_ent_pairs.add((ent, r_ent))
                    yield ent, r_ent

def is_valid_ent_type_combination(ent1, ent2):
    for left, right in [
        ("DATE", "DATE"),
        ("CONTRACT", "CONTRACT"),
    ]:  
        if ent1.label_ == left and ent2.label_ == right:
            return False
        else:
            labels = [ent1.label_, ent2.label_]
            blocked_labels = ["ODRINAL", "CARDINAL", "MONEY", "SECU"] 
            if (ent1.label_ in blocked_labels) or (ent2.label_ in blocked_labels):
                return False
    return doesnt_contain_base_alias(ent1, ent2)


def doesnt_contain_base_alias(ent1, ent2):
    doc = ent1.doc
    for ent in [ent1, ent2]:
        aliases = set([i._.containing_alias_span for i in ent])
        if aliases != set([None]):
            for alias in aliases:
                if alias in doc._.alias_cache._base_alias_set:
                    return False
    return True
                
def filter_unwanted_tokens_from_sdp(sdp_tuple: tuple[Span, Span, list[Token]]):
    new_sdp = []
    left, right, sdp = sdp_tuple
    parents_to_remove = set()
    for t in sdp:
        if t.dep_ in ["conj", "appos"]:
            if t.pos_ == t.head.pos_:
                if t.head not in left and t.head not in right:
                    parents_to_remove.add(t.head.i)
    for t in sdp:
        if t.i in parents_to_remove:
            continue
        # if (t not in left) and (t not in right):
        #     if t.dep_ in ["conj", "appos"]:
        #         continue
        #     else:
        #         pass
        new_sdp.append(t)
    return (sdp_tuple[0], sdp_tuple[1], new_sdp)

def is_valid_sdp(sdp_tuple: tuple[Span, Span, list[Token]]):
    left, right, sdp = sdp_tuple
    # verbs = list(filter(lambda x: x.pos_ == "VERB", sdp))
    if (
        (len(sdp) > 2)
    ):
        return True
    # print(f"not a valid sdp, (left, right, sdp): {left, right, sdp}")
    return False

def group_elements_of_sdp(sdp):
    # TODO: implement patterns
    # assuming sdp is sorted by token idx
    groups = []
    used_idx = set()
    for idx, token in enumerate(sdp):
        if idx in used_idx:
            continue
        if token.pos_ == "VERB":
            group = [token]
            used_idx.add(idx)
            for i in range(idx+1, len(sdp)):
                if i in used_idx:
                    break
                if sdp[i].dep_ == "prep":
                    group.append(sdp[i])
                    used_idx.add(i)
                else:
                    break
            groups.append({"name": "verbal", "group": group})

    for idx, token in enumerate(sdp):
        if idx in used_idx:
            continue
        if token.pos_ == "ADJ" and token.dep_ == "amod":
            group = [token]
            used_idx.add(idx)
            for i in range(idx+1, len(sdp)):
                if i in used_idx:
                    break
                if sdp[i].dep_ == "prep":
                    group.append(sdp[i])
                    used_idx.add(i)
                else:
                    break
            groups.append({"name": "amod", "group": group})

    for idx, token in enumerate(sdp):
        if idx in used_idx:
            continue
        if token.pos_ == "NOUN":
            group = [token]
            used_idx.add(idx)
            for i in range(idx+1, len(sdp)):
                if i in used_idx:
                    break
                if sdp[i].pos_ in ["NOUN", "PROPN"] or sdp[i].dep_ == "prep":
                    group.append(sdp[i])
                    used_idx.add(i)
                else:
                    break
            for i in range(idx-1, -1, -1):
                if i in used_idx:
                    break
                if (sdp[i].pos_ in ["NOUN", "PROPN"]) or (sdp[i].dep_ in ["prep", "nummod", "nmod", "compound", "det"]):
                    group.insert(0, sdp[i])
                    used_idx.add(i)
                else:
                    break
            groups.append({"name": "noun", "group": group})
        
    
    for idx, token in enumerate(sdp):
        if idx in used_idx:
            continue
        if token.pos_ == "ADV":
            group = [token]
            used_idx.add(idx)
            for i in range(idx+1, len(sdp)):
                if i in used_idx:
                    break
                if sdp[i].dep_ == "prep":
                    group.append(sdp[i])
                    used_idx.add(i)
                else:
                    break
            groups.append({"name": "adverbal", "group": group})

    for idx, token in enumerate(sdp):
        if idx in used_idx: continue
        group = [token]
        for i in range(idx+1, len(sdp)):
            if i in used_idx:
                break
            else:
                group.append(sdp[i])
                used_idx.add(i)

        for i in range(idx-1, -1, -1):
            if i in used_idx:
                break
            else:
                group.insert(0, sdp[i])
                used_idx.add(i)
        groups.append({"name": "unknown", "group": group})
        
    return sorted(groups, key=lambda x: x["group"][0].i)

            

def sdp_counts(sdp_tuple):
    left, right, sdp = sdp_tuple
    verb_count, noun_count, prep_count, obj_count, pobj_count, dobj_count = 0, 0, 0, 0, 0, 0
    for x in sdp:
        if x.pos_ == "VERB":
            verb_count += 1
        if x.pos_ == "NOUN":
            noun_count += 1
        if x.dep_ in ["pobj", "dobj"]:
            obj_count += 1
        if x.dep_ == "pobj":
            pobj_count += 1
        if x.dep_ == "dobj":
            dobj_count += 1
        if x.dep_ == "prep":
            prep_count += 1
    return (verb_count, noun_count, prep_count, obj_count, pobj_count, dobj_count)

def sdp_between_ents(doc, ent_filter) -> dict[tuple[Span, Span], tuple[Token]]:
    shortest_dependency_paths = []
    lca = doc.get_lca_matrix()
    seen_combinations = set()
    for sent in doc.sents:
        for ent1, ent2 in ent_filter(sent.ents):
            left, right = None, None
            if ent1.root.i < ent2.root.i:
                left, right = ent1, ent2
            else:
                left, right = ent2, ent1
            if is_valid_ent_type_combination(left, right):
                if ((left is not None) or (right is not None)) and (left, right) not in seen_combinations:
                    seen_combinations.add((left, right))
                    sdp = get_sdp_path(doc, left.root.i, right.root.i, lca)
                    shortest_dependency_paths.append((left, right, sdp))
    return shortest_dependency_paths

def get_sdps(doc, ent_filter=default_ent_filter, sdp_token_filter=filter_unwanted_tokens_from_sdp, sdp_validation_function=is_valid_sdp):
    valid_sdps = []
    discarded = []
    sdp_tuples = sdp_between_ents(doc, ent_filter=ent_filter)
    for sdp_tuple in sdp_tuples:
        filtered_tuple = sdp_token_filter(sdp_tuple)
        if sdp_validation_function(filtered_tuple) is True:
            valid_sdps.append(filtered_tuple)
        else:
            discarded.append((sdp_tuple, filtered_tuple))
    return valid_sdps, discarded

def get_sdps_noun_verbal(docs):
    sdp_groups = []
    for doc in docs:
        sdps, discarded = get_sdps(doc)
        for i, x in enumerate(sdps):
            left, right, sdp = x
            sorted_sdp = sorted(sdp, key=lambda x: x.i)
            sdp_grouped = group_elements_of_sdp(sorted_sdp)
            sdp_groups.append((left, right, sdp_grouped, x[2]))
    noun_verbal_unknown_sdps = list(filter(lambda x: (all([i["name"] in ["noun", "verbal", "unknown"] for i in x[2]]) and (len(x[2]) <= 4) and (any([i["name"] == "verbal"] for i in x[2]))) , sdp_groups))
    return noun_verbal_unknown_sdps

def create_sdp_noun_verbal_dict(left, right, sdp_grouped, sdp):
    nvu_info = {"groups": [], "relation": None}
    for sdp_group in sdp_grouped:
        values = sdp_group["group"]
        key = sdp_group["name"]
        group = {
            "type": key,
            "tokens": [t.i for t in values]
        }
        nvu_info["groups"].append(group)
        if key == "verbal":
            relation = ""
            for t in values:
                if t.pos_ == "VERB":
                    relation = "_".join([relation, t.lemma_])
                else:
                    relation = "_".join([relation, t.lower_])
            if nvu_info["relation"] is not None:
                relation = "+".join([nvu_info["relation"], relation])
            nvu_info["relation"] = relation

    nvu_info["sdp"] = [{"token": t, "lemma": t.lemma_, "pos": t.pos_, "dep": t.dep_, "i": t.i} for t in sdp]
    for name, ent in zip(["left", "right"], [left, right]):
        nvu_info[name] = {
            "tokens": [{"token": t, "lemma": t.lemma_, "pos": t.pos_, "dep": t.dep_, "i": t.i} for t in ent],
            "text": ent.text
        }
    nvu_info["sent"] = left.sent
    return nvu_info

def get_subj(token):
    subjs = []
    visited = set()
    queue = [[token]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)
        for child in node.children:
            if child.dep_ not in ["nsubj"]:
                new_path = list(path)
                new_path.append(child)
                queue.append(new_path)
            else:
                subjs.append(child)
    return subjs

def get_subjs_for_sdp(sdp_tuple):
    left, right, sdp = sdp_tuple
    has_subj = False
    subjs = []
    for t in sdp:
        if t.dep_ in ["nsubj", "nsubjpass", "dobj", "pobj"]:
            has_subj = True
            subjs.append(t)
    if not has_subj:
        subjs = get_subj(left.sent.root)         
    return subjs