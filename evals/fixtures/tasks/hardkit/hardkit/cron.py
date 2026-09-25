from datetime import datetime


def next_run(expression: str, after: datetime) -> datetime:
    """The first minute strictly after `after` (seconds ignored) matching a five-field cron
    expression "minute hour day-of-month month day-of-week". Each field is "*", a number, a
    range "a-b", a step "*/n" or "a-b/n", or a comma list of those. Minutes 0-59, hours 0-23,
    days 1-31, months 1-12, weekdays 0-7 where 0 and 7 are Sunday. When both day-of-month and
    day-of-week are restricted (neither is "*"), a day matches if either matches; otherwise
    both must. Raise ValueError for a malformed expression or out-of-range value."""
    raise NotImplementedError("TODO")
