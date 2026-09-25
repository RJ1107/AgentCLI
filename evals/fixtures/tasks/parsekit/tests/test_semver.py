from parsekit.semver import compare_versions


def test_semver():
    assert compare_versions("1.2.3", "1.2.3") == 0
    assert compare_versions("1.10.0", "1.9.9") == 1
    assert compare_versions("1.0.0-alpha", "1.0.0") == -1
    assert compare_versions("1.0.0-alpha.2", "1.0.0-alpha.10") == -1
    assert compare_versions("1.0.0-beta", "1.0.0-alpha.1") == 1
