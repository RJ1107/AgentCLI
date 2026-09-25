from collkit.topk import top_k_frequent


def test_topk():
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
    assert top_k_frequent(["b", "a", "b", "a", "c"], 2) == ["b", "a"]
    assert top_k_frequent([4], 3) == [4]
    assert top_k_frequent([], 1) == []
