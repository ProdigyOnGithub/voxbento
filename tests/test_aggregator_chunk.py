import asyncio
from unittest.mock import patch

import pytest

from portal.transcription.aggregator import CaptionAggregator


@pytest.mark.anyio
async def test_bug_50_word_limit_in_handle_chunk():
    received = []

    async def fake_callback(booth_id, message):
        received.append(message)

    aggregator = CaptionAggregator(fake_callback)

    # 60 words with no punctuation
    long_text = "word " * 60

    await aggregator.handle_chunk("booth-1", long_text)

    finals = [m for m in received if m.get("status") == "final"]

    # This assertion will fail because handle_chunk never triggers forced finalization!
    assert len(finals) > 0, "BUG: Passed 60 words, but no final caption was emitted!"


@pytest.mark.anyio
@patch("time.time")
async def test_bug_15_second_limit_in_handle_chunk(mock_time):
    received = []

    async def fake_callback(booth_id, message):
        received.append(message)

    aggregator = CaptionAggregator(fake_callback)

    # Start at time 0
    mock_time.return_value = 0.0
    await aggregator.handle_chunk("booth-1", "first chunk ")

    # Jump 20 seconds into the future
    mock_time.return_value = 20.0
    await aggregator.handle_chunk("booth-1", "second chunk")

    finals = [m for m in received if m.get("status") == "final"]

    # This assertion will fail because the time limit is defeated!
    assert len(finals) > 0, "BUG: 20 seconds passed, but no final caption was emitted!"
