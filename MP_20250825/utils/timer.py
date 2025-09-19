from datetime import datetime, timezone

from utils.coordinates import true_round


def get_timestamp():
    start = datetime(2015, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    curr = datetime.now(timezone.utc)
    diff = curr - start
    ms_diff = diff.total_seconds() * 1000
    return true_round(ms_diff)
