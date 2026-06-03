"""Actum - on-device multimodal robot agent."""

__all__ = ["AudioCapture", "RobotAgent", "RobotRuntime", "RobotTools"]


def __getattr__(name: str):
    if name == "RobotAgent":
        from actum.agent import RobotAgent

        return RobotAgent
    if name == "RobotTools":
        from actum.tools import RobotTools

        return RobotTools
    if name == "RobotRuntime":
        from actum.runtime import RobotRuntime

        return RobotRuntime
    if name == "AudioCapture":
        from actum.perception import AudioCapture

        return AudioCapture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
