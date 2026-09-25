def dedupe(items: list, key=None) -> list:
    """Drop repeats, keeping the first occurrence and the original order. With key,
    two items are repeats when key(item) is equal. Items may be unhashable when a
    hashable key is given."""
    raise NotImplementedError("TODO")
