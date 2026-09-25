import pytest
from hardkit.jsonpointer import resolve

DOC = {'a': {'b': [10, 20, {'c': 'deep'}]}, 'm~n': 1, 'x/y': 2, '': 'empty-key', 'list': []}


def test_resolve():
    assert resolve(DOC, '') == {'a': {'b': [10, 20, {'c': 'deep'}]}, 'm~n': 1, 'x/y': 2, '': 'empty-key', 'list': []}
    assert resolve(DOC, '/a/b/0') == 10
    assert resolve(DOC, '/a/b/2/c') == 'deep'
    assert resolve(DOC, '/m~0n') == 1
    assert resolve(DOC, '/x~1y') == 2
    assert resolve(DOC, '/') == 'empty-key'
    assert resolve(DOC, '/a/b') == [10, 20, {'c': 'deep'}]
    with pytest.raises(ValueError):
        resolve(DOC, 'a')
    with pytest.raises(ValueError):
        resolve(DOC, '/a/b/01')
    with pytest.raises(ValueError):
        resolve(DOC, '/a/b/-')
    with pytest.raises(KeyError):
        resolve(DOC, '/nope')
    with pytest.raises(IndexError):
        resolve(DOC, '/a/b/3')
    with pytest.raises(IndexError):
        resolve(DOC, '/list/0')

