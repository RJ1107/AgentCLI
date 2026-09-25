def resolve(document, pointer: str):
    """Resolve an RFC 6901 JSON pointer. "" is the whole document; otherwise the pointer must
    start with "/", and each token has "~1" decoded to "/" and then "~0" to "~". A token
    indexes a dict by key or a list by a non-negative decimal index without leading zeros.
    Raise ValueError for a pointer not starting with "/" or a malformed list index ("01",
    "-", "x"), KeyError for a missing key, and IndexError for an index past the end."""
    raise NotImplementedError("TODO")
