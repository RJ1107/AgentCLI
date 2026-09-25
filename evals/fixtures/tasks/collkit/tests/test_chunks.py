import pytest
from collkit.chunks import chunked


def test_chunks():
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunked([], 3) == []
    assert chunked([1, 2], 5) == [[1, 2]]
    with pytest.raises(ValueError):
        chunked([1], 0)
