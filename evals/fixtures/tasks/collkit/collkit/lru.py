class LRUCache:
    """Fixed-capacity cache. get(key) returns the value or None and marks the key
    as recently used; put(key, value) inserts or updates and marks it recently used,
    evicting the least recently used key when over capacity. len() gives the size."""

    def __init__(self, capacity: int):
        raise NotImplementedError("TODO")

    def get(self, key):
        raise NotImplementedError("TODO")

    def put(self, key, value) -> None:
        raise NotImplementedError("TODO")

    def __len__(self) -> int:
        raise NotImplementedError("TODO")
