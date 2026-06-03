"""Robot action tools — Python functions the LLM calls to act in the world.

Each method is registered with litert_lm as a callable tool. The LLM chains
multiple tool calls within a single turn (e.g. look → navigate → speak → done).

Physical actions are routed through RobotRuntime and RobotBackend so hardware,
simulation, ROS, and MCP adapters share one contract.
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
        action = {"type": action_type, "time": time.time(), **kwargs}
        self.actions_taken.append(action)
        runtime = getattr(self._agent, "runtime", None)
        if runtime is not None:
            runtime.record_tool(action)
        return action

    def _record_result(self, action: dict, result):
        runtime = getattr(self._agent, "runtime", None)
        if runtime is not None:
            runtime.record_tool(action, result)

    @property
    def _backend(self):
        runtime = getattr(self._agent, "runtime", None)
        return runtime.backend if runtime is not None else None

    # ── Terminal ───────────────────────────────────────────────────────────

    def done(self, summary: str) -> str:
        """Signal that the current task is complete.

        Args:
            summary: One sentence describing what was accomplished.
        """
        action = self._record("done", summary=summary)
        runtime = getattr(self._agent, "runtime", None)
        if runtime is not None:
            runtime.finish_task(summary)
        print(f"[done] {summary}")
        return "Task marked complete."

    def set_plan(self, goal: str, steps: str) -> str:
        """Set or update your current task plan.

        Args:
            goal: The current operator goal in one sentence.
            steps: Newline-separated plan steps. Keep each step concrete and observable.
        """
        runtime = getattr(self._agent, "runtime", None)
        if runtime is not None:
            runtime.set_plan(goal, steps)
        parsed = [line.strip() for line in steps.splitlines() if line.strip()]
        self._record("plan", goal=goal, steps=parsed)
        print(f"[plan] {goal} | {len(parsed)} steps")
        return f"Plan updated with {len(parsed)} steps."

    def mark_step(self, step: str) -> str:
        """Mark one plan step as active.

        Args:
            step: Step id or exact label to mark active.
        """
        runtime = getattr(self._agent, "runtime", None)
        if runtime is not None:
            runtime.mark_step(step)
        self._record("step", step=step, status="active")
        return f"Active step: {step}"

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

        backend = self._backend
        if backend is not None:
            result = backend.drive(direction, distance_m)
            action = self._record(
                "navigate",
                direction=direction,
                distance_m=distance_m,
                backend=backend.name,
                ok=result.ok,
            )
            self._record_result(action, result)
            print(f"[navigate] {direction} {distance_m:.2f} m ({'ok' if result.ok else 'failed'})")
            return result.message

        self._record("navigate", direction=direction, distance_m=distance_m, backend=None, ok=False)
        print(f"[navigate] {direction} {distance_m:.2f} m")
        return "No robot backend configured."

    def rotate(self, degrees: float) -> str:
        """Rotate the robot in place.

        Args:
            degrees: Degrees to rotate. Positive = clockwise, negative = counter-clockwise.
        """
        backend = self._backend
        if backend is not None:
            result = backend.rotate(degrees)
            action = self._record("rotate", degrees=degrees, backend=backend.name, ok=result.ok)
            self._record_result(action, result)
            print(f"[rotate] {degrees:+.0f}° ({'ok' if result.ok else 'failed'})")
            return result.message

        self._record("rotate", degrees=degrees, backend=None, ok=False)
        print(f"[rotate] {degrees:+.0f}°")
        return "No robot backend configured."

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
        backend = self._backend
        if backend is not None:
            result = backend.gesture(gesture)
            action = self._record("wave", gesture=gesture, backend=backend.name, ok=result.ok)
            self._record_result(action, result)
            print(f"[wave] {gesture}")
            return result.message
        self._record("wave", gesture=gesture, backend=None, ok=False)
        print(f"[wave] {gesture} (no backend)")
        return "No robot backend configured."

    # ── Manipulation ───────────────────────────────────────────────────────

    def gripper(self, action: str) -> str:
        """Control the robot's gripper or end-effector.

        Args:
            action: One of: open, close, grab, release.
        """
        VALID = {"open", "close", "grab", "release"}
        if action not in VALID:
            return f"Invalid action. Use one of: {', '.join(sorted(VALID))}"
        backend = self._backend
        if backend is not None:
            result = backend.gripper(action)
            record = self._record("gripper", action=action, backend=backend.name, ok=result.ok)
            self._record_result(record, result)
            print(f"[gripper] {action}")
            return result.message
        self._record("gripper", action=action, backend=None, ok=False)
        print(f"[gripper] {action}")
        return "No robot backend configured."

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
        runtime = getattr(self._agent, "runtime", None)
        if runtime is not None:
            runtime.events.append("intent.status", "agent", message=message)
        self._record("status", message=message)
        print(f"[status] {message}")
        return "Status logged."

    # ── Registry ───────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        """Return all tool callables for litert_lm registration."""
        return [
            self.done,
            self.set_plan,
            self.mark_step,
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
