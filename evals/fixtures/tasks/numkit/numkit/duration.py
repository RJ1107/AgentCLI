def parse_duration(text: str) -> int:
    """Parse durations like "1h30m", "45s", "2d4h", "1h 5m 10s" into seconds. Units:
    d, h, m, s, each at most once, in that order; spaces allowed between parts.
    Raise ValueError for anything else, including an empty string."""
    raise NotImplementedError("TODO")
