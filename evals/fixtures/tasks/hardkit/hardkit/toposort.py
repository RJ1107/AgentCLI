def order(graph: dict[str, list[str]]) -> list[str]:
    """Order nodes so each comes after all of its dependencies. graph maps a node to the
    nodes it depends on; a node that appears only as a dependency is included too. Among
    nodes that are ready at the same time, take the alphabetically smallest first. Raise
    ValueError if there is a cycle."""
    raise NotImplementedError("TODO")
