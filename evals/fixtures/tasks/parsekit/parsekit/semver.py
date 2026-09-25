def compare_versions(a: str, b: str) -> int:
    """Compare semantic versions "MAJOR.MINOR.PATCH" with an optional "-prerelease"
    of dot-separated identifiers. Return -1, 0, or 1. A prerelease sorts before the
    release; numeric identifiers compare as numbers and sort before text ones."""
    raise NotImplementedError("TODO")
