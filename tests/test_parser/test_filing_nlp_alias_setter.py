from main.parser.filing_nlp_alias_setter import get_all_overlapping_intervals, get_longest_from_overlapping_groups, AliasCache, AliasMatcher
import pytest


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


#TODO: add below tests for AliasCache
# def test_add_alias():
# def test_assign_alias_without_references()
# def test_assign_alias_with_references()
#TODO: add tests for AliasMatcher
#TODO: add tests for AliasSetter
        
