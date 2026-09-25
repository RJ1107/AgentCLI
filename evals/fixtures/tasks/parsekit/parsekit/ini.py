def parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Parse INI text into {section: {key: value}}. Keys before any [section] go in
    "default". Lines starting with ";" or "#" and blank lines are skipped; keys and
    values are stripped; a repeated key keeps the last value. Do not use configparser."""
    raise NotImplementedError("TODO")
