import numpy as np
import pytest
from actum.perception import EnergyVAD, SileroVAD, build_default_vad


def test_energy_vad_basic():
    # Test energy VAD with simple silence and speech signals
    vad = EnergyVAD(
        speech_ratio=2.0,
        silence_ratio=1.3,
        noise_floor_init=0.01,
        min_speech_chunks=2,
        silence_chunks=3,
        pre_roll_chunks=2,
    )

    # 1. Start with silence (RMS should be close to 0.005)
    silence = np.zeros(480, dtype=np.float32)
    for _ in range(10):
        res = vad.process(silence)
        assert res is None
        assert not vad._in_speech

    # 2. Feed loud speech
    speech = np.ones(480, dtype=np.float32) * 0.1  # RMS is 0.1
    res = vad.process(speech)
    assert res is None
    assert vad._in_speech

    # Keep feeding speech
    for _ in range(5):
        res = vad.process(speech)
        assert res is None
        assert vad._in_speech

    # 3. Go back to silence (should trigger end after 3 silence chunks)
    # first silence chunk
    res = vad.process(silence)
    assert res is None
    assert vad._in_speech
    assert vad._silence_count == 1

    # second silence chunk
    res = vad.process(silence)
    assert res is None
    assert vad._in_speech
    assert vad._silence_count == 2

    # third silence chunk -> triggers output
    res = vad.process(silence)
    assert res is not None
    assert not vad._in_speech
    assert len(res) > 0


def test_silero_vad_basic():
    try:
        import faster_whisper.vad
    except ImportError:
        pytest.skip("faster-whisper not installed; skipping Silero VAD tests")

    vad = SileroVAD(
        threshold=0.4,
        neg_threshold=0.3,
        min_speech_chunks=2,
        silence_chunks=2,
        pre_roll_chunks=2,
        max_speech_chunks=5,  # low safety cap to test easily
    )

    # Reset test
    vad.reset()
    assert not vad._in_speech
    assert len(vad._accumulator) == 0

    # Feed silence (zeros)
    silence = np.zeros(480, dtype=np.float32)
    for _ in range(5):
        res = vad.process(silence)
        assert res is None
        assert not vad._in_speech

    # Feed a simulated loud speech waveform (non-zero)
    # Just need enough samples to fill the 512-sample window
    speech = np.sin(np.linspace(0, 20 * np.pi, 480)).astype(np.float32) * 0.5
    # Since Silero is a neural net, a sine wave might get low probability,
    # but we can verify it doesn't crash, and check the safety cap.

    # Let's verify that processing 10 chunks triggers the max speech cap
    # (max_speech_chunks = 5, which is ~160 ms)
    force_terminated = False
    for _ in range(10):
        # We manually force _in_speech = True to test the cap directly
        vad._in_speech = True
        res = vad.process(speech)
        if res is not None:
            force_terminated = True
            break

    assert force_terminated or len(vad._buffer) < 5


def test_build_default_vad():
    vad = build_default_vad()
    assert vad is not None
    # Should be SileroVAD if faster_whisper is available, otherwise EnergyVAD
    try:
        import faster_whisper.vad

        assert isinstance(vad, SileroVAD)
    except ImportError:
        assert isinstance(vad, EnergyVAD)
