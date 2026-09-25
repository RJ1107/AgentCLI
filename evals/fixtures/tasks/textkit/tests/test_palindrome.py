from textkit.palindrome import is_palindrome


def test_palindrome():
    assert is_palindrome("A man, a plan, a canal: Panama")
    assert not is_palindrome("race a car")
    assert is_palindrome("")
    assert is_palindrome("No 'x' in Nixon")
