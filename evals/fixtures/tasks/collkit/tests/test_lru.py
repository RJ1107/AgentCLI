from collkit.lru import LRUCache


def test_lru():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("c") == 3
    cache.put("a", 10)
    assert cache.get("a") == 10
    assert len(cache) == 2
