from pathlib import Path

from hardkit.regex import match


def test_regex():
    assert match('a*b', 'b') is True
    assert match('a*b', 'aaab') is True
    assert match('a*b', 'ab') is True
    assert match('a*b', 'a') is False
    assert match('a*b', 'abb') is False
    assert match('a.c', 'abc') is True
    assert match('a.c', 'a-c') is True
    assert match('a.c', 'ac') is False
    assert match('a.c', 'abbc') is False
    assert match('colou?r', 'color') is True
    assert match('colou?r', 'colour') is True
    assert match('colou?r', 'colouur') is False
    assert match('[a-c]+x', 'abcx') is True
    assert match('[a-c]+x', 'x') is False
    assert match('[a-c]+x', 'cax') is True
    assert match('[a-c]+x', 'adx') is False
    assert match('[^0-9]+', 'abc') is True
    assert match('[^0-9]+', 'a1c') is False
    assert match('[^0-9]+', '') is False
    assert match('[^0-9]+', '--') is True
    assert match('a\\.b', 'a.b') is True
    assert match('a\\.b', 'axb') is False
    assert match('x*y*z*', '') is True
    assert match('x*y*z*', 'xxz') is True
    assert match('x*y*z*', 'zy') is False
    assert match('x*y*z*', 'xyz') is True
    assert match('a*a', 'a') is True
    assert match('a*a', 'aaaa') is True
    assert match('a*a', '') is False
    assert match('[abc]*c', 'abcc') is True
    assert match('[abc]*c', 'c') is True
    assert match('[abc]*c', 'ab') is False
    assert match('.*end', 'the end') is True
    assert match('.*end', 'end') is True
    assert match('.*end', 'endless') is False
    assert match('ab*c?d+', 'ad') is True
    assert match('ab*c?d+', 'abbbcdd') is True
    assert match('ab*c?d+', 'abc') is False
    assert match('ab*c?d+', 'acd') is True
    assert match('[0-9]+\\.[0-9]*', '3.14') is True
    assert match('[0-9]+\\.[0-9]*', '3.') is True
    assert match('[0-9]+\\.[0-9]*', '.5') is False
    assert match('[0-9]+\\.[0-9]*', '10.0') is True
    assert match('h.?llo', 'hllo') is True
    assert match('h.?llo', 'hello') is True
    assert match('h.?llo', 'heello') is False

def test_no_shortcut():
    source = Path('hardkit/regex.py').read_text(encoding='utf-8')
    for banned in ['import re', 'from re ']:
        assert banned not in source, f'{banned} is not allowed'

