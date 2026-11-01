"""Acquire an Internet UTC reference, falling back to the host clock."""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from time import perf_counter
from urllib.error import URLError
from urllib.request import Request, urlopen


TIME_URLS = (
    "https://www.google.com/generate_204",
    "https://www.cloudflare.com/",
)


def reference_clock(timeout_seconds=1.0):
    """Return UTC, its matching monotonic sample, and the clock source.

    HTTP Date headers provide the Internet reference. Request midpoint
    compensation limits the effect of network round-trip latency. If both
    endpoints fail or omit Date, use the computer's UTC clock.
    """
    for url in TIME_URLS:
        started = perf_counter()
        try:
            request = Request(url, headers={"User-Agent": "MOSS-ASR-XPU/1.0"})
            with urlopen(request, timeout=timeout_seconds) as response:
                date_header = response.headers.get("Date")
            completed = perf_counter()
            if not date_header:
                continue
            internet_utc = parsedate_to_datetime(date_header)
            if internet_utc.tzinfo is None:
                internet_utc = internet_utc.replace(tzinfo=timezone.utc)
            round_trip = completed - started
            return (internet_utc.astimezone(timezone.utc) + timedelta(seconds=round_trip / 2),
                    (started + completed) / 2, "internet")
        except (OSError, URLError, TimeoutError, ValueError, TypeError):
            continue
    return datetime.now(timezone.utc), perf_counter(), "computer"
