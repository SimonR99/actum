from actum.backends.factory import create_backend
from actum.backends.laptop import LaptopBackend


def test_laptop_backend_reports_local_io_and_refuses_motion():
    backend = create_backend(
        {
            "robot": {
                "backend": "laptop",
                "laptop": {"webcam": False, "microphone": True, "speaker": True},
            }
        }
    )

    assert isinstance(backend, LaptopBackend)
    assert backend.connect() is True

    state = backend.get_state()
    assert state.backend == "laptop"
    assert state.mode == "companion"
    assert state.metadata["webcam"] is False
    assert state.metadata["microphone"] is True

    result = backend.drive("forward", 1.0)
    assert result.ok is False
    assert "stationary" in result.message
