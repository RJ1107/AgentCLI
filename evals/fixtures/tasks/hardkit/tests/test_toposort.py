import pytest
from hardkit.toposort import order


def test_order():
    assert order({'b': ['a'], 'c': ['a'], 'a': []}) == ['a', 'b', 'c']
    assert order({'deploy': ['build', 'test'], 'test': ['build'], 'build': ['fetch']}) == ['fetch', 'build', 'test', 'deploy']
    assert order({'z': [], 'y': [], 'x': []}) == ['x', 'y', 'z']
    assert order({'app': ['lib', 'utils'], 'lib': ['utils'], 'docs': []}) == ['docs', 'utils', 'lib', 'app']
    assert order({}) == []
    with pytest.raises(ValueError):
        order({'a': ['b'], 'b': ['a']})
    with pytest.raises(ValueError):
        order({'a': ['a']})
    with pytest.raises(ValueError):
        order({'a': ['b'], 'b': ['c'], 'c': ['a'], 'd': []})

