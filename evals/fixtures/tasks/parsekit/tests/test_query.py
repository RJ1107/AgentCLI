from parsekit.query import parse_query


def test_query():
    assert parse_query("?a=1&b=2&a=3") == {"a": ["1", "3"], "b": ["2"]}
    assert parse_query("q=hello+world%21") == {"q": ["hello world!"]}
    assert parse_query("flag") == {"flag": [""]}
    assert parse_query("") == {}
