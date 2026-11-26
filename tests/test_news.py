from crypto_sentiment import news

SAMPLE = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Bitcoin hits new high</title><link>http://x/1</link>
<pubDate>Tue, 24 Nov 2026 10:00:00 +0000</pubDate></item>
<item><title>  Ethereum
 upgrade live </title><link>http://x/2</link>
<pubDate>Tue, 24 Nov 2026 15:30:00 +0530</pubDate></item>
<item><title></title><pubDate>Tue, 24 Nov 2026 10:00:00 +0000</pubDate></item>
<item><title>No date</title></item>
<item><title>Bad date</title><pubDate>not a date</pubDate></item>
</channel></rss>"""


def test_parse_rss_keeps_valid_items_only():
    df = news.parse_rss(SAMPLE, "test")
    assert list(df["title"]) == ["Bitcoin hits new high", "Ethereum upgrade live"]


def test_timestamps_are_utc_naive():
    df = news.parse_rss(SAMPLE, "test")
    assert df["published_at"].dt.tz is None
    # 15:30 +05:30 == 10:00 UTC
    assert df["published_at"].iloc[0] == df["published_at"].iloc[1]
