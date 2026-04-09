"""Robot action tools — Python functions the LLM calls to act in the world.

Each method is registered with litert_lm as a callable tool. The LLM chains
multiple tool calls within a single turn (e.g. look → navigate → speak → done).

Hardware integration points are marked with # --- hardware hook ---
Replace those lines with your actual motor driver / ROS publisher / serial command.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo.agent import RobotAgent


class RobotTools:
    """All tools available to the robot agent LLM."""

    def __init__(self, agent: RobotAgent):
        self._agent = agent
        self.actions_taken: list[dict] = []

    def _reset(self):
        self.actions_taken.clear()

    def _record(self, action_type: str, **kwargs):
        self.actions_taken.append({"type": action_type, "time": time.time(), **kwargs})

    # ── Terminal ───────────────────────────────────────────────────────────

    def done(self, summary: str) -> str:
        """Signal that the current task is complete.

        Args:
            summary: One sentence describing what was accomplished.
        """
        self._record("done", summary=summary)
        print(f"[done] {summary}")
        return "Task marked complete."

    # ── Communication ──────────────────────────────────────────────────────

    def speak(self, text: str) -> str:
        """Say something aloud through the robot's speaker.

        Args:
            text: What to say. 1-2 sentences max. Do not repeat yourself.
        """
        self._agent._pending_speech.append(text)
        self._record("speak", text=text)
        print(f"[speak] {text!r}")
        return "Queued for speech."

    # ── Locomotion ─────────────────────────────────────────────────────────

    def navigate(self, direction: str, distance_m: float = 0.5) -> str:
        """Move the robot in a direction.

        Args:
            direction: One of: forward, backward, left, right, stop.
            distance_m: Distance in metres (ignored for stop).
        """
        VALID = {"forward", "backward", "left", "right", "stop"}
        if direction not in VALID:
            return f"Invalid direction. Use one of: {', '.join(sorted(VALID))}"
        # --- hardware hook ---
        # robot_hw.drive(direction, distance_m)
        self._record("navigate", direction=direction, distance_m=distance_m)
        print(f"[navigate] {direction} {distance_m:.2f} m")
        return f"Moved {direction} {distance_m:.2f} m."

    def rotate(self, degrees: float) -> str:
        """Rotate the robot in place.

        Args:
            degrees: Degrees to rotate. Positive = clockwise, negative = counter-clockwise.
        """
        # --- hardware hook ---
        # robot_hw.rotate(degrees)
        self._record("rotate", degrees=degrees)
        print(f"[rotate] {degrees:+.0f}°")
        return f"Rotated {degrees:+.0f}°."

    # ── Gestures ───────────────────────────────────────────────────────────

    def wave(self, gesture: str = "high wave") -> str:
        """Perform a wave or greeting gesture.

        Args:
            gesture: Gesture name. Common options:
                'high wave'   — both arms waving high (default)
                'face wave'   — wave near the face
                'shake hand'  — extend hand for handshake
                'clap'        — clapping motion
                'high five'   — raise hand for high five
                'hands up'    — both hands raised
        """
        hw = getattr(self._agent, "_hardware", None)
        if hw is not None:
            ok = hw.gesture(gesture)
            self._record("wave", gesture=gesture, hardware=True)
            print(f"[wave] {gesture}")
            return f"Gesture '{gesture}' executed." if ok else f"Gesture '{gesture}' failed."
        # No hardware — stub
        self._record("wave", gesture=gesture, hardware=False)
        print(f"[wave] {gesture} (stub — no hardware)")
        return f"Wave gesture '{gesture}' (no hardware connected)."

    # ── Manipulation ───────────────────────────────────────────────────────

    def gripper(self, action: str) -> str:
        """Control the robot's gripper or end-effector.

        Args:
            action: One of: open, close, grab, release.
        """
        VALID = {"open", "close", "grab", "release"}
        if action not in VALID:
            return f"Invalid action. Use one of: {', '.join(sorted(VALID))}"
        # --- hardware hook ---
        # robot_hw.gripper(action)
        self._record("gripper", action=action)
        print(f"[gripper] {action}")
        return f"Gripper: {action}."

    # ── Vision ─────────────────────────────────────────────────────────────

    def look(self) -> str:
        """Capture a fresh camera frame to observe the environment.

        The image will be delivered to you in a follow-up message immediately
        after this tool call completes. Wait for it before deciding what to do.
        """
        frame = self._agent.capture_frame()
        if frame is None:
            return "Camera not available."
        self._agent._pending_frame = frame
        self._record("look")
        print("[look] frame captured")
        return "Frame captured. You will receive the image in the next message."

    # ── Memory ─────────────────────────────────────────────────────────────

    def remember(self, key: str, value: str) -> str:
        """Persist a fact to long-term memory across turns.

        Args:
            key: Short identifier, e.g. 'user_name', 'dock_location'.
            value: Value to store. Overwrites any existing value for this key.
        """
        self._agent.memory[key] = value
        self._record("remember", key=key, value=value)
        print(f"[remember] {key!r} = {value!r}")
        return f"Stored: {key} = {value}"

    def recall(self, key: str) -> str:
        """Retrieve a value from long-term memory.

        Args:
            key: Key to look up.
        """
        val = self._agent.memory.get(key)
        return f"{key} = {val}" if val is not None else f"No memory for key '{key}'."

    def list_memories(self) -> str:
        """List all keys currently stored in memory."""
        if not self._agent.memory:
            return "Memory is empty."
        return "Memory:\n" + "\n".join(f"  {k}: {v}" for k, v in self._agent.memory.items())

    # ── Diagnostics ────────────────────────────────────────────────────────

    def report_status(self, message: str) -> str:
        """Report current reasoning or ask for clarification.

        Use when uncertain what to do, or to communicate what you are about to do.

        Args:
            message: Status message or question for the operator.
        """
        self._record("status", message=message)
        print(f"[status] {message}")
        return "Status logged."

    # ── Registry ───────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        """Return all tool callables for litert_lm registration."""
        return [
            self.done,
            self.speak,
            self.wave,
            self.navigate,
            self.rotate,
            self.gripper,
            self.look,
            self.remember,
            self.recall,
            self.list_memories,
            self.report_status,
        ]
