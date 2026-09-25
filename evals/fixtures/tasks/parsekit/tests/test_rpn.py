import pytest
from parsekit.rpn import eval_rpn


def test_rpn():
    assert eval_rpn("3 4 + 2 *") == 14
    assert eval_rpn("5 1 2 + 4 * + 3 -") == 14
    assert eval_rpn("-3 2 *") == -6
    assert eval_rpn("1.5 2 /") == 0.75
    for bad in ["1 +", "1 2", "1 x +"]:
        with pytest.raises(ValueError):
            eval_rpn(bad)
    with pytest.raises(ZeroDivisionError):
        eval_rpn("1 0 /")
