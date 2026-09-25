from numkit.bytesize import format_bytes


def test_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(1048576) == "1.0 MB"
    assert format_bytes(5 * 1024**4) == "5.0 TB"
    assert format_bytes(0) == "0 B"
