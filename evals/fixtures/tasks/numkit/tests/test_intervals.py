from numkit.intervals import merge_intervals


def test_intervals():
    assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]
    assert merge_intervals([(1, 3), (3, 5)]) == [(1, 5)]
    assert merge_intervals([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]
    assert merge_intervals([]) == []
