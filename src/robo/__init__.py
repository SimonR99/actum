"""Robo - on-device multimodal robot agent."""

__all__ = ["AudioCapture", "RobotAgent", "RobotRuntime", "RobotTools"]


def __getattr__(name: str):
    if name == "RobotAgent":
        from robo.agent import RobotAgent

        return RobotAgent
    if name == "RobotTools":
        from robo.tools import RobotTools

        return RobotTools
    if name == "RobotRuntime":
        from robo.runtime import RobotRuntime

        return RobotRuntime
    if name == "AudioCapture":
        from robo.perception import AudioCapture

        return AudioCapture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
