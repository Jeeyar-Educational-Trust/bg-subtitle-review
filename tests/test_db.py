"""db.direct_url: migrations derive Neon's direct host from the one DATABASE_URL."""

import pytest

import db


@pytest.mark.parametrize("url, expected", [
    # Neon's pooled host -> its direct host; the rest of the string is kept as it is.
    ("postgresql://u:p@ep-cool-sun-a1b2-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
     "postgresql://u:p@ep-cool-sun-a1b2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"),
    # Already direct, or not Neon at all: unchanged.
    ("postgresql://u:p@ep-cool-sun-a1b2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require",
     "postgresql://u:p@ep-cool-sun-a1b2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"),
    ("postgresql://postgres@localhost:5432/subtitle_test", "postgresql://postgres@localhost:5432/subtitle_test"),
    # Only the host changes, never a password that happens to contain "-pooler.".
    ("postgres://u:x-pooler.y@ep-a-pooler.eu-central-1.aws.neon.tech/db",
     "postgres://u:x-pooler.y@ep-a.eu-central-1.aws.neon.tech/db"),
    ("not a url", "not a url"),
])
def test_direct_url(url, expected):
    assert db.direct_url(url) == expected
