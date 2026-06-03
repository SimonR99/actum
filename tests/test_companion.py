from actum.core.companion import CompanionPolicy


def test_direct_text_is_processed():
    policy = CompanionPolicy({"always_on": True})

    decision = policy.decide({"source": "text", "text": "what do you see?"})

    assert decision.process is True
    assert decision.reason == "direct user input"


def test_direct_chat_and_language_are_processed():
    policy = CompanionPolicy({"always_on": True})

    assert policy.decide({"source": "chat", "text": "status"}).process is True
    assert policy.decide({"source": "language", "text": "status"}).process is True


def test_passive_vision_is_ignored_when_routine():
    policy = CompanionPolicy({"always_on": True, "proactive_mode": "conservative"})

    decision = policy.decide({"source": "vision", "text": "ambient motion near desk"})

    assert decision.process is False
    assert "did not require action" in decision.reason


def test_passive_safety_language_is_processed():
    policy = CompanionPolicy({"always_on": True, "proactive_mode": "conservative"})

    decision = policy.decide({"source": "vision", "text": "person fell and needs help"})

    assert decision.process is True
    assert decision.matched == "help"


def test_high_importance_passive_event_is_processed():
    policy = CompanionPolicy({"always_on": True, "proactive_mode": "conservative"})

    decision = policy.decide({"source": "vision", "importance": 0.9})

    assert decision.process is True
    assert decision.matched == "importance"
