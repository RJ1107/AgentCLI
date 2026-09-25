from parsekit.csvline import parse_csv_line


def test_csv():
    assert parse_csv_line("a,b,c") == ["a", "b", "c"]
    assert parse_csv_line('"a,b",c') == ["a,b", "c"]
    assert parse_csv_line('"say ""hi""",x') == ['say "hi"', "x"]
    assert parse_csv_line("a,,b") == ["a", "", "b"]
    assert parse_csv_line("") == [""]
