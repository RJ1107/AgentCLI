def parse_query(query: str) -> dict[str, list[str]]:
    """Parse a URL query string ("a=1&b=2&a=3", optional leading "?") into a dict of
    lists in order of appearance. "+" is a space and %XX escapes are decoded; a key
    without "=" gets "". Do not use urllib."""
    raise NotImplementedError("TODO")
