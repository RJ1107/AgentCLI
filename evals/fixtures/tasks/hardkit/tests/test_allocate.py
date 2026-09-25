import pytest
from hardkit.allocate import allocate


def test_allocate():
    assert allocate(100, [1, 1, 1]) == [34, 33, 33]
    assert allocate(10, [3, 3, 3]) == [4, 3, 3]
    assert allocate(7, [1, 2, 4]) == [1, 2, 4]
    assert allocate(0, [5, 5]) == [0, 0]
    assert allocate(101, [50, 50, 1]) == [50, 50, 1]
    assert allocate(5, [0, 1, 0]) == [0, 5, 0]
    assert allocate(1000, [7, 13, 29, 51]) == [70, 130, 290, 510]
    assert allocate(3, [1, 1, 1, 1, 1]) == [1, 1, 1, 0, 0]
    with pytest.raises(ValueError):
        allocate(-1, [1])
    with pytest.raises(ValueError):
        allocate(5, [0, 0])
    with pytest.raises(ValueError):
        allocate(5, [1, -1])

