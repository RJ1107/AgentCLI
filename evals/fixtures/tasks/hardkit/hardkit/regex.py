def match(pattern: str, text: str) -> bool:
    """True when the whole text matches pattern. Supported: literal characters; "." (any one
    character); classes like "[abc]", ranges "[a-z0-9]", and negated classes "[^...]"; the
    quantifiers "*", "+", "?" applied to the preceding atom (greedy, with backtracking); and
    "\\" escaping the next character, so "\\." is a literal dot. No groups, alternation, or
    anchors: the match is always of the whole text. Do not use the re module."""
    raise NotImplementedError("TODO")
