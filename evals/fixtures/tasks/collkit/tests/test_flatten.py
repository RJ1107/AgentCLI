from collkit.flatten import flatten


def test_flatten():
    assert flatten([1, [2, [3, (4, 5)]], 6]) == [1, 2, 3, 4, 5, 6]
    assert flatten(["ab", ["cd"]]) == ["ab", "cd"]
    assert flatten([]) == []
    assert flatten([{"a": 1}, [[]]]) == [{"a": 1}]
