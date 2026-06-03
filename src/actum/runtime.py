"""Runtime state shared by the LLM agent, tools, backend, and web UI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from actum.backends.base import RobotBackend
from actum.backends.factory import create_backend
from actum.core.autonomy import BehaviorTreeState, BodyPerception, CronRegistry, SpatialMap
from actum.core.capabilities import default_capabilities
from actum.core.companion import CompanionDecision, CompanionPolicy
from actum.core.events import EventLog
from actum.core.intent import IntentState
from actum.core.memory import MemoryStore
from actum.core.schema import ActionResult
from actum.core.settings import RuntimeSettings
from actum.integrations.mcp import describe_servers


class RobotRuntime:
    def __init__(self, config: dict[str, Any], robot_name: str):
        self.config = config
        self.robot_name = robot_name
        self.backend: RobotBackend = create_backend(config)
        self.events = EventLog()
        self.intent = IntentState()
        self.capabilities = default_capabilities()
        capability_names = [item["name"] for item in self.capabilities.list()]
        self.settings = RuntimeSettings(config, capability_names)
        self.personality = config.get("personality", {}) if isinstance(config.get("personality"), dict) else {}
        self.companion = CompanionPolicy(config.get("companion", {}) if isinstance(config.get("companion"), dict) else {})
        self.memory = MemoryStore(config.get("memory", {}) if isinstance(config.get("memory"), dict) else {})
        behavior_cfg = config.get("behavior_loop", {}) if isinstance(config.get("behavior_loop"), dict) else {}
        self.behavior = BehaviorTreeState(behavior_cfg)
        cron_cfg = config.get("cron", []) if isinstance(config.get("cron"), list) else []
        self.cron = CronRegistry(cron_cfg)
        self.spatial_map = SpatialMap()
        self.body = BodyPerception()
        self.tool_graph: list[dict[str, Any]] = []

    def connect(self) -> bool:
        ok = self.backend.connect()
        self.events.append("backend.connected" if ok else "backend.unavailable", self.backend.name, backend=self.backend.name)
        return ok

    def close(self):
        self.backend.close()
        self.events.append("backend.closed", self.backend.name)

    def configure_robot(self, robot_config: dict[str, Any]) -> tuple[bool, str]:
        """Replace the active robot backend from a validated robot config."""
        candidate = deepcopy(self.config)
        candidate["robot"] = deepcopy(robot_config)
        new_backend = create_backend(candidate)

        old_backend = self.backend
        old_name = old_backend.name
        if old_backend.connected:
            old_backend.close()

        self.config["robot"] = deepcopy(robot_config)
        self.backend = new_backend
        try:
            connected = self.backend.connect()
        except Exception as exc:
            connected = False
            self.events.append("backend.unavailable", self.backend.name, backend=self.backend.name, error=str(exc))
            return False, f"Robot backend changed to {self.backend.name}, but connection failed: {exc}"

        self.events.append(
            "backend.reconfigured",
            "operator",
            previous_backend=old_name,
            backend=self.backend.name,
            connected=connected,
        )
        if connected:
            return True, f"Robot backend changed to {self.backend.name}."
        return False, f"Robot backend changed to {self.backend.name}, but it is not connected."

    def set_plan(self, goal: str, plan_text: str):
        self.intent.set_plan(goal, plan_text)
        self.behavior.set_tree(
            goal,
            [{"id": step.id, "label": step.label, "kind": "plan_step", "status": step.status} for step in self.intent.steps],
        )
        self.events.append("intent.plan", "agent", goal=goal, steps=[step.to_dict() for step in self.intent.steps])

    def mark_step(self, label_or_id: str):
        self.intent.mark_active(label_or_id)
        self.behavior.mark_node(label_or_id, "active")
        self.events.append("intent.step_active", "agent", step=label_or_id)

    def finish_step(self, detail: str = ""):
        self.intent.complete_active(detail)
        self.events.append("intent.step_done", "agent", detail=detail)

    def finish_task(self, summary: str):
        self.intent.finish(summary)
        self.behavior.mark_node(self.behavior.active_node_id, "done", summary)
        self.events.append("intent.done", "agent", summary=summary)

    def fail_task(self, message: str):
        self.intent.fail(message)
        self.behavior.mark_node(self.behavior.active_node_id, "blocked", message)
        self.events.append("intent.blocked", "agent", message=message)

    def set_behavior_tree(self, goal: str, nodes: list[dict[str, Any]]):
        self.behavior.set_tree(goal, nodes)
        self.events.append("behavior.tree", "agent", goal=goal, nodes=self.behavior.to_dict()["nodes"])

    def mark_behavior_node(self, node_id: str, status: str = "active", detail: str = "") -> bool:
        ok = self.behavior.mark_node(node_id, status, detail)
        self.events.append("behavior.node", "agent", node=node_id, status=status, detail=detail, ok=ok)
        return ok

    def add_cron_job(self, name: str, every_seconds: float, instruction: str) -> dict[str, Any]:
        job = self.cron.add(name, every_seconds, instruction)
        self.events.append("cron.add", "operator", job=job.to_dict())
        return job.to_dict()

    def record_map_observation(
        self,
        summary: str,
        place: str = "",
        x: float | None = None,
        y: float | None = None,
        yaw_deg: float | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        obs = self.spatial_map.record(summary, place, x, y, yaw_deg, confidence)
        self.events.append("map.observation", "agent", observation=obs.to_dict())
        return obs.to_dict()

    def update_body_perception(
        self,
        summary: str,
        posture: str = "",
        holding: str = "",
        contacts: list[str] | None = None,
        joints: dict[str, float] | None = None,
    ):
        self.body.update(summary, posture=posture, holding=holding, contacts=contacts, joints=joints)
        self.events.append("body.perception", "agent", body=self.body.to_dict())

    def record_tool(self, action: dict[str, Any], result: ActionResult | None = None) -> str:
        node_id = action.get("_node_id")
        if node_id:
            for node in self.tool_graph:
                if node["id"] == node_id:
                    if result:
                        node["result"] = result.to_dict()
                        self.events.append("tool.result", "tool", tool=node["type"], node=node)
                    return node_id

        node_id = f"tool-{len(self.tool_graph) + 1}"
        action["_node_id"] = node_id
        node = {
            "id": node_id,
            "type": action.get("type", "tool"),
            "action": action,
            "result": result.to_dict() if result else None,
        }
        self.tool_graph.append(node)
        self.events.append("tool.result" if result else "tool.call", "tool", tool=node["type"], node=node)
        return node_id

    def should_process_event(self, event: dict[str, Any]) -> CompanionDecision:
        decision = self.companion.decide(event)
        event_type = "companion.process" if decision.process else "companion.ignore"
        self.events.append(event_type, decision.source, decision=decision.to_dict())
        return decision

    def snapshot(self) -> dict[str, Any]:
        robot_state = self.backend.get_state()
        self.body.update_from_robot_state(robot_state)
        return {
            "robot_name": self.robot_name,
            "backend": self.backend.name,
            "robot_config": deepcopy(self.config.get("robot", {})),
            "robot_state": robot_state.to_dict(),
            "personality": _public_personality(self.robot_name, self.personality),
            "companion": self.companion.to_dict(),
            "memory": self.memory.snapshot(),
            "mcp_servers": describe_servers(self.config),
            "intent": self.intent.to_dict(),
            "behavior": self.behavior.to_dict(),
            "cron": self.cron.to_dict(),
            "map": self.spatial_map.to_dict(),
            "body": self.body.to_dict(),
            "settings": self.settings.to_dict(),
            "events": self.events.tail(200),
            "tool_graph": self.tool_graph[-200:],
            "capabilities": self.capabilities.list(),
        }


def _public_personality(robot_name: str, personality: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(personality.get("name") or robot_name),
        "persona": str(personality.get("persona", "")),
        "likes": list(personality.get("likes", [])) if isinstance(personality.get("likes"), list) else [],
        "principles": list(personality.get("principles", [])) if isinstance(personality.get("principles"), list) else [],
        "speaking_style": str(personality.get("speaking_style", "")),
    }
