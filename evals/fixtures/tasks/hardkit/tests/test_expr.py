from pathlib import Path

import pytest
from hardkit.expr import evaluate


def test_expr():
    assert evaluate('1 + 2 * 3') == pytest.approx(7)
    assert evaluate('(1+2)*3') == pytest.approx(9)
    assert evaluate('2**3**2') == pytest.approx(512)
    assert evaluate('-2**2') == pytest.approx(-4)
    assert evaluate('2**-1') == pytest.approx(0.5)
    assert evaluate('-(3-5)*2') == pytest.approx(4)
    assert evaluate('10/4') == pytest.approx(2.5)
    assert evaluate('3 - -2') == pytest.approx(5)
    assert evaluate('+4') == pytest.approx(4)
    assert evaluate('2*(3+(4-1))**2') == pytest.approx(72)
    assert evaluate('1.5*4-0.5') == pytest.approx(5.5)
    assert evaluate('100 - 2*3**2 / 6') == pytest.approx(97.0)
    with pytest.raises(ZeroDivisionError):
        evaluate('1/0')
    with pytest.raises(ValueError):
        evaluate('1 +')
    with pytest.raises(ValueError):
        evaluate('(1')
    with pytest.raises(ValueError):
        evaluate('1 2')
    with pytest.raises(ValueError):
        evaluate('')
    with pytest.raises(ValueError):
        evaluate('2**')
    with pytest.raises(ValueError):
        evaluate('3 * / 4')
    with pytest.raises(ValueError):
        evaluate(')(')

def test_no_shortcut():
    source = Path('hardkit/expr.py').read_text(encoding='utf-8')
    for banned in ['eval(', 'exec(', 'compile(']:
        assert banned not in source, f'{banned} is not allowed'

