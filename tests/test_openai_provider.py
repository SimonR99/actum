"""OpenAI provider conversation-history maintenance."""

from actum.inference.openai import _MAX_HISTORY, OpenAIProvider


def _provider():
    return OpenAIProvider(model="test", pop_pending_frame=lambda: None)


def test_trim_history_keeps_short_conversations():
    provider = _provider()
    provider._messages = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": "hi"} for _ in range(5)
    ]
    provider._trim_history()
    assert len(provider._messages) == 6


def test_trim_history_never_orphans_tool_replies():
    provider = _provider()
    messages = [{"role": "system", "content": "sys"}]
    # Alternate assistant tool_calls and their tool replies so any naive cut
    # point can land between an assistant message and its tool replies.
    for i in range(_MAX_HISTORY):
        messages.append({"role": "assistant", "tool_calls": [{"id": f"c{i}"}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok2"})
    provider._messages = messages

    provider._trim_history()

    kept = provider._messages
    assert kept[0]["role"] == "system"
    assert len(kept) <= _MAX_HISTORY + 1
    # The first non-system message must not be a tool reply.
    assert kept[1]["role"] != "tool"
    # Every tool reply must be preceded by its assistant tool_calls message.
    ids_seen = set()
    for message in kept[1:]:
        if message["role"] == "assistant":
            ids_seen.update(call["id"] for call in message.get("tool_calls", []))
        elif message["role"] == "tool":
            assert message["tool_call_id"] in ids_seen
