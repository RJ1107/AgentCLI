import pytest
from numkit.duration import parse_duration


def test_duration():
    assert parse_duration("1h30m") == 5400
    assert parse_duration("45s") == 45
    assert parse_duration("2d4h") == 187200
    assert parse_duration("1h 5m 10s") == 3910
    for bad in ["", "10", "5m1h", "1x"]:
        with pytest.raises(ValueError):
            parse_duration(bad)
