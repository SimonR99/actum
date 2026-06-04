"""Robot action tools — Python functions the LLM calls to act in the world.

Each method is registered with litert_lm as a callable tool. The LLM chains
multiple tool calls within a single turn (e.g. look → navigate → speak → done).

Physical actions are routed through RobotRuntime and RobotBackend so hardware,
simulation, ROS, and MCP adapters share one contract.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from actum.core.schema import ActionResult, now
from actum.integrations import mcp as mcp_bridge
from actum.integrations.web import fetch_text

if TYPE_CHECKING:
    from actum.agent import RobotAgent

# Gemma 4's tool-call serialization wraps string values with <|"|> delimiters.
# litert_lm passes these through to Python callables instead of stripping them.
_GEMMA_TOKENS = ('<|"|>',)


def _clean(text: str) -> str:
    for tok in _GEMMA_TOKENS:
        text = text.replace(tok, "")
    return text


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
    def _config(self) -> dict:
        return getattr(self._agent, "config", {}) or {}

    @property
    def _backend(self):
        runtime = getattr(self._agent, "runtime", None)
        return runtime.backend if runtime is not None else None

    @property
    def _memory_store(self):
        runtime = getattr(self._agent, "runtime", None)
        return getattr(runtime, "memory", None) if runtime is not None else None

    # ── Terminal ───────────────────────────────────────────────────────────

    def done(self, summary: str) -> str:
        """Signal that the current task is complete.

        Args:
            summary: One sentence describing what was accomplished.
        """
        summary = _clean(summary)
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

    def set_behavior_tree(self, goal: str, nodes_json: str) -> str:
        """Publish a behavior-tree-like state for the operator.

        Args:
            goal: Current autonomy goal.
            nodes_json: JSON array of nodes, or an object with a "nodes" array.
                Each node may include id, label, kind, status, detail, children.
        """
        runtime = getattr(self._agent, "runtime", None)
        if runtime is None:
            return "Runtime is not available."
        try:
            parsed = json.loads(nodes_json or "[]")
            if isinstance(parsed, dict):
                parsed_goal = str(parsed.get("goal") or goal)
                parsed = parsed.get("nodes", [])
                goal = parsed_goal
            if not isinstance(parsed, list):
                raise ValueError("nodes_json must be a JSON array or object with nodes.")
            nodes = [item for item in parsed if isinstance(item, dict)]
        except Exception:
            nodes = [{"label": line.strip()} for line in str(nodes_json).splitlines() if line.strip()]
        runtime.set_behavior_tree(goal, nodes)
        self._record("behavior_tree", goal=goal, nodes=len(nodes))
        return f"Behavior tree updated with {len(nodes)} node(s)."

    def mark_behavior_node(self, node: str, status: str = "active", detail: str = "") -> str:
        """Mark a behavior tree node active, waiting, done, blocked, or failed.

        Args:
            node: Node id or exact label.
            status: waiting, active, done, blocked, or failed.
            detail: Optional operator-visible detail.
        """
        runtime = getattr(self._agent, "runtime", None)
        if runtime is None:
            return "Runtime is not available."
        ok = runtime.mark_behavior_node(node, status, detail)
        self._record("behavior_node", node=node, status=status, detail=detail, ok=ok)
        return f"Behavior node {node} marked {status}." if ok else f"Behavior node {node} was not found."

    # ── Communication ──────────────────────────────────────────────────────

    def speak(self, text: str) -> str:
        """Say something aloud through the robot's speaker.

        Args:
            text: What to say. 1-2 sentences max. Do not repeat yourself.
        """
        text = _clean(text)
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
        store = self._memory_store
        if store is not None:
            store.remember_fact(key, value)
            self._agent.memory = store.facts
        else:
            self._agent.memory[key] = value
        self._record("remember", key=key, value=value)
        print(f"[remember] {key!r} = {value!r}")
        return f"Stored: {key} = {value}"

    def recall(self, key: str) -> str:
        """Retrieve a value from long-term memory.

        Args:
            key: Key to look up.
        """
        store = self._memory_store
        val = store.recall_fact(key) if store is not None else self._agent.memory.get(key)
        return f"{key} = {val}" if val is not None else f"No memory for key '{key}'."

    def list_memories(self) -> str:
        """List all keys currently stored in memory."""
        store = self._memory_store
        facts = store.facts if store is not None else self._agent.memory
        if not facts:
            return "Memory is empty."
        return "Memory:\n" + "\n".join(f"  {k}: {v}" for k, v in facts.items())

    def remember_person(self, name: str, note: str) -> str:
        """Store a note about a person you have interacted with.

        Args:
            name: Person name or stable identifier.
            note: Durable fact or interaction note to remember.
        """
        store = self._memory_store
        if store is None:
            return "Persistent memory is not available."
        store.remember_person(name, note)
        self._record("remember_person", name=name, note=note)
        return f"Remembered person note for {name}."

    def remember_place(self, name: str, note: str) -> str:
        """Store a note about a place, landmark, room, or docking location.

        Args:
            name: Place or landmark name.
            note: Durable note about the place.
        """
        store = self._memory_store
        if store is None:
            return "Persistent memory is not available."
        store.remember_place(name, note)
        self._record("remember_place", name=name, note=note)
        return f"Remembered place note for {name}."

    def record_observation(self, summary: str, tags: str = "") -> str:
        """Store a short observation from vision, audio, or task execution.

        Args:
            summary: One concise sentence describing what was observed.
            tags: Optional comma-separated labels such as kitchen, user, obstacle.
        """
        store = self._memory_store
        if store is None:
            return "Persistent memory is not available."
        record = store.record_observation(summary, source="agent", tags=_csv(tags))
        self._record("record_observation", summary=summary, tags=_csv(tags), memory_id=record.id)
        return f"Recorded observation {record.id}."

    def remember_spatial_note(self, summary: str, place: str = "") -> str:
        """Store a mapping or navigation note.

        Args:
            summary: Spatial relation or navigation note.
            place: Optional related place or landmark name.
        """
        store = self._memory_store
        if store is None:
            return "Persistent memory is not available."
        record = store.remember_spatial_note(summary, source="agent", place=place)
        self._record("remember_spatial_note", summary=summary, place=place, memory_id=record.id)
        return f"Recorded spatial note {record.id}."

    def record_map_observation(
        self,
        summary: str,
        place: str = "",
        x: float | None = None,
        y: float | None = None,
        yaw_deg: float | None = None,
        confidence: float = 1.0,
    ) -> str:
        """Add a landmark or spatial observation to the live map.

        Args:
            summary: Concise spatial observation.
            place: Optional place/room/landmark name.
            x: Optional x position in the current map frame.
            y: Optional y position in the current map frame.
            yaw_deg: Optional heading in degrees.
            confidence: 0-1 confidence score.
        """
        runtime = getattr(self._agent, "runtime", None)
        if runtime is None:
            return "Runtime is not available."
        obs = runtime.record_map_observation(summary, place, x, y, yaw_deg, confidence)
        store = self._memory_store
        memory_id = ""
        if store is not None:
            record = store.remember_spatial_note(summary, source="agent", place=place)
            memory_id = record.id
        self._record("map_observation", summary=summary, place=place, map_id=obs["id"], memory_id=memory_id)
        return f"Recorded map observation {obs['id']}."

    def update_body_perception(
        self,
        summary: str,
        posture: str = "",
        holding: str = "",
        contacts: str = "",
        joints_json: str = "{}",
    ) -> str:
        """Update the robot's visible self/body perception.

        Args:
            summary: Concise body-state summary.
            posture: Optional posture/motion state.
            holding: Optional object currently held.
            contacts: Optional comma-separated contact points.
            joints_json: Optional JSON object of joint names to numeric values.
        """
        runtime = getattr(self._agent, "runtime", None)
        if runtime is None:
            return "Runtime is not available."
        try:
            joints = json.loads(joints_json or "{}")
            if not isinstance(joints, dict):
                joints = {}
        except Exception:
            joints = {}
        runtime.update_body_perception(summary, posture=posture, holding=holding, contacts=_csv(contacts), joints=joints)
        self._record("body_perception", summary=summary, posture=posture, holding=holding)
        return "Body perception updated."

    def recent_memories(self, limit: int = 8, kind: str = "") -> str:
        """Read recent memory records.

        Args:
            limit: Maximum records to return.
            kind: Optional filter: episode, observation, or spatial.
        """
        store = self._memory_store
        if store is None:
            return "Persistent memory is not available."
        records = store.recent(limit, kind or None)
        if not records:
            return "No recent memories."
        return json.dumps(records, indent=2)[:6000]

    def search_memory(self, query: str, limit: int = 6) -> str:
        """Retrieve the memory entries most relevant to a query.

        Use this to recall facts, people, places, or past events related to what
        you are doing now, instead of guessing.

        Args:
            query: What you want to recall, in a few words.
            limit: Maximum entries to return.
        """
        store = self._memory_store
        if store is None:
            return "Persistent memory is not available."
        hits = store.search(query, limit=int(limit))
        self._record("search_memory", query=query, count=len(hits))
        if not hits:
            return f"No memory related to '{query}'."
        return "\n".join(f"{kind}: {text}" for kind, text in hits)

    def consolidate_memory(self) -> str:
        """Organise memory by removing duplicate and stale records.

        Run this when memory feels cluttered or after a long session.
        """
        store = self._memory_store
        if store is None:
            return "Persistent memory is not available."
        report = store.consolidate()
        self._record("consolidate_memory", **report)
        return (
            f"Memory consolidated: removed {report['removed_episodes']} duplicate episode(s) "
            f"and {report['removed_spatial_notes']} duplicate spatial note(s)."
        )

    def schedule_job(self, name: str, every_seconds: float, instruction: str) -> str:
        """Create a background scheduled instruction.

        Args:
            name: Short job name.
            every_seconds: Interval in seconds.
            instruction: What the agent should do when the job fires.
        """
        runtime = getattr(self._agent, "runtime", None)
        if runtime is None:
            return "Runtime is not available."
        job = runtime.add_cron_job(name, every_seconds, instruction)
        self._record("schedule_job", name=name, every_seconds=every_seconds, job_id=job["id"])
        return f"Scheduled {job['name']} every {job['every_seconds']:.0f} seconds."

    # ── External data / MCP ────────────────────────────────────────────────

    def web_fetch(self, url: str, max_chars: int = 4000) -> str:
        """Fetch readable text from an HTTP(S) URL.

        Args:
            url: Fully qualified http:// or https:// URL.
            max_chars: Maximum text characters to return, capped for safety.
        """
        started = now()
        try:
            data = fetch_text(url, max_chars=max_chars)
            result = ActionResult(
                action="web_fetch",
                ok=True,
                message=f"Fetched {data['final_url']}.",
                data={key: value for key, value in data.items() if key != "text"} | {"chars": len(data["text"])},
                started_at=started,
                ended_at=now(),
            )
            action = self._record(
                "web_fetch",
                url=url,
                ok=True,
                chars=len(data["text"]),
                truncated=bool(data.get("truncated")),
            )
            self._record_result(action, result)
            return f"{data['final_url']} ({data['status']}):\n{data['text']}"
        except Exception as exc:
            result = ActionResult(
                action="web_fetch",
                ok=False,
                message=str(exc),
                data={"url": url},
                started_at=started,
                ended_at=now(),
            )
            action = self._record("web_fetch", url=url, ok=False)
            self._record_result(action, result)
            return f"Web fetch failed: {exc}"

    def list_mcp_servers(self) -> str:
        """List configured MCP servers available to this robot runtime."""
        servers = mcp_bridge.describe_servers(self._config)
        sdk_state = "installed" if mcp_bridge.sdk_available() else "not installed"
        payload = {"sdk": sdk_state, "servers": servers}
        action = self._record("list_mcp_servers", count=len(servers), sdk=sdk_state)
        result = ActionResult(
            action="list_mcp_servers",
            ok=True,
            message=f"{len(servers)} MCP server(s) configured.",
            data=payload,
            started_at=action["time"],
            ended_at=now(),
        )
        self._record_result(action, result)
        if not servers:
            return f"No MCP servers are enabled in config. MCP SDK: {sdk_state}."
        return json.dumps(payload, indent=2, sort_keys=True)

    def list_mcp_tools(self, server: str) -> str:
        """List tools exposed by a configured MCP server.

        Args:
            server: Name of the configured MCP server.
        """
        started = now()
        try:
            payload = mcp_bridge.list_tools(self._config, server)
            action = self._record("list_mcp_tools", server=server, ok=True, count=len(payload.get("tools", [])))
            result = ActionResult(
                action="list_mcp_tools",
                ok=True,
                message=f"Listed tools for MCP server {server}.",
                data=payload,
                started_at=started,
                ended_at=now(),
            )
            self._record_result(action, result)
            return json.dumps(payload, indent=2, sort_keys=True)
        except Exception as exc:
            action = self._record("list_mcp_tools", server=server, ok=False)
            result = ActionResult(
                action="list_mcp_tools",
                ok=False,
                message=str(exc),
                data={"server": server},
                started_at=started,
                ended_at=now(),
            )
            self._record_result(action, result)
            return f"MCP tool listing failed: {exc}"

    def call_mcp_tool(self, server: str, tool: str, arguments_json: str = "{}") -> str:
        """Call a tool exposed by a configured MCP server.

        Args:
            server: Name of the configured MCP server.
            tool: Tool name exposed by that server.
            arguments_json: JSON object with tool arguments.
        """
        started = now()
        try:
            arguments = json.loads(arguments_json or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments_json must decode to a JSON object.")
            payload = mcp_bridge.call_tool(self._config, server, tool, arguments)
            action = self._record("call_mcp_tool", server=server, tool=tool, ok=True)
            result = ActionResult(
                action="call_mcp_tool",
                ok=True,
                message=f"Called MCP tool {tool} on {server}.",
                data=payload,
                started_at=started,
                ended_at=now(),
            )
            self._record_result(action, result)
            return json.dumps(payload, indent=2, sort_keys=True)
        except Exception as exc:
            action = self._record("call_mcp_tool", server=server, tool=tool, ok=False)
            result = ActionResult(
                action="call_mcp_tool",
                ok=False,
                message=str(exc),
                data={"server": server, "tool": tool},
                started_at=started,
                ended_at=now(),
            )
            self._record_result(action, result)
            return f"MCP tool call failed: {exc}"

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
        tools = [
            self.done,
            self.set_plan,
            self.mark_step,
            self.set_behavior_tree,
            self.mark_behavior_node,
            self.speak,
            self.wave,
            self.navigate,
            self.rotate,
            self.gripper,
            self.look,
            self.remember,
            self.recall,
            self.list_memories,
            self.remember_person,
            self.remember_place,
            self.record_observation,
            self.remember_spatial_note,
            self.record_map_observation,
            self.update_body_perception,
            self.recent_memories,
            self.search_memory,
            self.consolidate_memory,
            self.schedule_job,
            self.web_fetch,
            self.list_mcp_servers,
            self.list_mcp_tools,
            self.call_mcp_tool,
            self.report_status,
        ]
        runtime = getattr(self._agent, "runtime", None)
        settings = getattr(runtime, "settings", None) if runtime is not None else None
        if settings is None:
            return tools
        return [tool for tool in tools if settings.tool_enabled(tool.__name__)]


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]
