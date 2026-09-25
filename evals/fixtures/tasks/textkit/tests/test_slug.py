from textkit.slug import slugify


def test_slug():
    assert slugify("  Hello, World!  ") == "hello-world"
    assert slugify("a--b__c") == "a-b-c"
    assert slugify("Already-Slug") == "already-slug"
    assert slugify("!!!") == ""
    assert slugify("Python 3.12 Release") == "python-3-12-release"
