from numkit.median import running_median


def test_median():
    assert running_median([5, 15, 1, 3]) == [5, 10, 5, 4]
    assert running_median([]) == []
    assert running_median([2, 2, 2]) == [2, 2, 2]
    assert running_median(list(range(1, 10001)))[-1] == 5000.5
