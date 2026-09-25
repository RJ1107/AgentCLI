from numkit.primes import primes_up_to


def test_primes():
    assert primes_up_to(10) == [2, 3, 5, 7]
    assert primes_up_to(1) == []
    assert primes_up_to(2) == [2]
    assert len(primes_up_to(100000)) == 9592
