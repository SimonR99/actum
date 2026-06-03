"""Robot backend adapters."""

from robo.backends.base import RobotBackend
from robo.backends.factory import create_backend
from robo.backends.fake import FakeBackend

__all__ = ["FakeBackend", "RobotBackend", "create_backend"]
