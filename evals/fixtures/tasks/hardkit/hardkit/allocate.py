def allocate(total: int, weights: list[int]) -> list[int]:
    """Split a non-negative integer total in proportion to non-negative weights, using the
    largest remainder method: each share is first rounded down, then the leftover units go
    one at a time to the shares with the largest fractional remainders, ties going to the
    earlier item. The result sums to total. Raise ValueError when total is negative, a
    weight is negative, or no weight is positive."""
    raise NotImplementedError("TODO")
