from parsekit.ini import parse_ini


def test_ini():
    text = "name = top\n; comment\n[db]\nhost = localhost\nport=5432\n\n[db]\nport = 6543\n"
    assert parse_ini(text) == {
        "default": {"name": "top"},
        "db": {"host": "localhost", "port": "6543"},
    }
    assert parse_ini("") == {}
