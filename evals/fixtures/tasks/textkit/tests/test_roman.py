import pytest
from textkit.roman import to_roman


def test_roman():
    assert to_roman(4) == "IV"
    assert to_roman(1994) == "MCMXCIV"
    assert to_roman(3999) == "MMMCMXCIX"
    assert to_roman(58) == "LVIII"
    with pytest.raises(ValueError):
        to_roman(0)
