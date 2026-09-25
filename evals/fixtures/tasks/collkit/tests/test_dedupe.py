from collkit.dedupe import dedupe


def test_dedupe():
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert dedupe(["a", "A", "b"], key=str.lower) == ["a", "b"]
    assert dedupe([{"id": 1}, {"id": 1}, {"id": 2}], key=lambda d: d["id"]) == [
        {"id": 1},
        {"id": 2},
    ]
    assert dedupe([]) == []
