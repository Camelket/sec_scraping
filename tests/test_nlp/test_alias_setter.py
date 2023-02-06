from main.nlp.filing_nlp_alias_setter import get_all_overlapping_intervals, get_longest_from_overlapping_groups, AliasCache, AliasMatcher
import pytest
import spacy


@pytest.fixture
def get_test_doc():
    nlp = spacy.load("en_core_web_lg")
    text = "The contract (the 'Contract') was completed. It wasn't renewed."
    yield nlp(text)

@pytest.mark.parametrize(["input", "expected_tuples", "expected_groups"], [
    # no overlaps
    (
        [
            (1, 2),
            (3, 5),
            (6, 8)
        ],
        [],
        []
    ),
    # overlaps
    (
        [
            (1, 3),
            (2, 5),
            (6, 8),
            (9, 11),
            (11, 12)
        ],
        [
            (1, 3),
            (2, 5),
            (9, 11),
            (11, 12)
        ],
        [[(1, 3), (2, 5)],  [(9, 11), (11, 12)]]
    ),
    (
        [
            (2, 5),
            (9, 11),
            (6, 8),
            (1, 3),
            (11, 12)
        ],
        [
            (1, 3),
            (2, 5),
            (9, 11),
            (11, 12)
        ],
        [[(2, 5), (1, 3)],  [(9, 11), (11, 12)]]
    ),
    (
        [
            (1, 3),
            (3, 5),
            (5, 8),
            (9, 11),
            (11, 12)
        ],
        [
            (1, 3),
            (3, 5),
            (5, 8),
            (9, 11),
            (11, 12)
        ],
        [[(1, 3), (3, 5), (5, 8)],  [(9, 11), (11, 12)]]
    ),
    (
        [
            (1, 3),
            (2, 5),
            (4, 9),
            (6, 8),
            (9, 11),
            (11, 14)
        ],
        [
            (1, 3),
            (2, 5),
            (4, 9),
            (6, 8),
            (9, 11),
            (11, 14) 
        ],
        [[(1, 3), (2, 5), (4, 9), (6, 8), (9, 11), (11, 14)]]
    )


])
def test_get_all_overlapping_intervals_sweep_line(input, expected_tuples, expected_groups):
    received_tuples, received_groups = get_all_overlapping_intervals(input)
    assert received_tuples.sort(key=lambda x: x[0]) == expected_tuples.sort(key=lambda x: x[0])
    assert received_groups == expected_groups

@pytest.mark.parametrize(["input", "expected"], [
    (
        [
            (1, 3),
            (2, 5),
            (6, 8),
            (9, 11),
            (11, 14)
        ],
        [(2,5), (11, 14)]
    ),
    (
        [
            (1, 3),
            (2, 5),
            (4, 9),
            (6, 8),
            (9, 11),
            (11, 14)
        ],
        [(4, 9)]
    )
])
def test_get_longest_from_overlapping(input, expected):
    _, groups = get_all_overlapping_intervals(input)
    print(groups)
    longest = get_longest_from_overlapping_groups(groups)
    assert longest == expected



def test_add_alias(get_test_doc):
    doc = get_test_doc
    cache = AliasCache()
    base_alias = doc[5:6]
    cache.add_alias(base_alias)
    assert cache._base_alias_set == set([(5, 6)])

def test_assign_alias(get_test_doc):
    doc = get_test_doc
    cache = AliasCache()
    base_alias = doc[5:6]
    origin = doc[1:2]
    print(base_alias, origin)
    cache.add_alias(base_alias)
    cache.assign_alias(base_alias, origin, 1.5)
    assert cache._parent_map[(5, 6)] == [(1, 2)]
    assert cache._children_map[(1, 2)] == [(5, 6)]
    assert cache._alias_origin_map[(5, 6)] == (1, 2)
    assert cache._alias_assignment_score[(5, 6)] == 1.5

def test_reassign_alias(get_test_doc):
    doc = get_test_doc
    cache = AliasCache()
    base_alias = doc[5:6]
    origin1 = doc[1:2]
    origin2 = doc[3:4]
    cache.add_alias(base_alias)
    cache.assign_alias(base_alias, origin1, 1.5)
    assert cache._parent_map[(5, 6)] == [(1, 2)]
    assert cache._children_map[(1, 2)] == [(5, 6)]
    assert cache._alias_origin_map[(5, 6)] == (1, 2)
    assert cache._alias_assignment_score[(5, 6)] == 1.5
    cache.assign_alias(base_alias, origin2, 1.6)
    assert cache._parent_map[(5, 6)] == [(3, 4)]
    assert cache._children_map[(3, 4)] == [(5, 6)]
    assert cache._alias_origin_map[(5, 6)] == (3, 4)
    assert cache._alias_assignment_score[(5, 6)] == 1.6

# def test_assign_alias_with_references()


#TODO: add tests for AliasMatcher
#TODO: add tests for SimpleAliasSetter
#TODO: add tests for MultiComponentAliasSetter
        
