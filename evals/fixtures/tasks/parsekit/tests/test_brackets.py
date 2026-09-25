from parsekit.brackets import balanced


def test_brackets():
    assert balanced("([]{})")
    assert not balanced("([)]")
    assert balanced("f(x) = [1, {2}]")
    assert not balanced("((")
    assert balanced("")
