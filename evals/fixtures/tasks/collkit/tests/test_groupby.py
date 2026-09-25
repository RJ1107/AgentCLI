from collkit.groupby import group_by


def test_group_by():
    words = ["apple", "bob", "avocado", "cat", "banana"]
    grouped = group_by(words, lambda w: w[0])
    assert grouped == {"a": ["apple", "avocado"], "b": ["bob", "banana"], "c": ["cat"]}
    assert list(grouped) == ["a", "b", "c"]
    assert group_by([], len) == {}
