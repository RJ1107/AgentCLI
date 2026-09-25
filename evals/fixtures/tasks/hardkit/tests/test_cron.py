import datetime

import pytest
from hardkit.cron import next_run


def test_cron():
    assert next_run('*/15 * * * *', datetime.datetime(2026, 9, 25, 10, 7)) == datetime.datetime(2026, 9, 25, 10, 15)
    assert next_run('0 9 * * 1-5', datetime.datetime(2026, 9, 25, 18, 0)) == datetime.datetime(2026, 9, 28, 9, 0)
    assert next_run('30 2 1 * *', datetime.datetime(2026, 9, 25, 0, 0)) == datetime.datetime(2026, 10, 1, 2, 30)
    assert next_run('0 0 29 2 *', datetime.datetime(2026, 3, 1, 0, 0)) == datetime.datetime(2028, 2, 29, 0, 0)
    assert next_run('0 12 13 * 5', datetime.datetime(2026, 9, 25, 12, 0)) == datetime.datetime(2026, 10, 2, 12, 0)
    assert next_run('5,35 */6 * * 0', datetime.datetime(2026, 9, 26, 23, 59)) == datetime.datetime(2026, 9, 27, 0, 5)
    assert next_run('0 0 * * 7', datetime.datetime(2026, 9, 25, 0, 0)) == datetime.datetime(2026, 9, 27, 0, 0)
    assert next_run('15 10-12/2 * 1,6 *', datetime.datetime(2026, 9, 25, 0, 0)) == datetime.datetime(2027, 1, 1, 10, 15)
    assert next_run('59 23 31 12 *', datetime.datetime(2026, 12, 31, 23, 59)) == datetime.datetime(2027, 12, 31, 23, 59)
    assert next_run('0 8 1-7 * 1', datetime.datetime(2026, 9, 25, 9, 0)) == datetime.datetime(2026, 9, 28, 8, 0)
    with pytest.raises(ValueError):
        next_run('61 * * * *', datetime.datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        next_run('* * *', datetime.datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        next_run('*/0 * * * *', datetime.datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        next_run('* 24 * * *', datetime.datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        next_run('a * * * *', datetime.datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        next_run('5-2 * * * *', datetime.datetime(2026, 1, 1))

