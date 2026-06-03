"""Runtime state shared by the LLM agent, tools, backend, and web UI."""

from __future__ import annotations

from typing import Any

from robo.backends.base import RobotBackend
from robo.backends.factory import create_backend
from robo.core.capabilities import default_capabilities
from robo.core.events import EventLog
from robo.core.intent import IntentState
from robo.core.schema import ActionResult


class RobotRuntime:
    def __init__(self, config: dict[str, Any], robot_name: str):
        self.config = config
        self.robot_name = robot_name
        self.backend: RobotBackend = create_backend(config)
        self.events = EventLog()
        self.intent = IntentState()
        self.capabilities = default_capabilities()
        self.tool_graph: list[dict[str, Any]] = []

    def connect(self) -> bool:
        ok = self.backend.connect()
        self.events.append("backend.connected" if ok else "backend.unavailable", self.backend.name, backend=self.backend.name)
        return ok

    def close(self):
        self.backend.close()
        self.events.append("backend.closed", self.backend.name)

    def set_plan(self, goal: str, plan_text: str):
        self.intent.set_plan(goal, plan_text)
        self.events.append("intent.plan", "agent", goal=goal, steps=[step.to_dict() for step in self.intent.steps])

    def mark_step(self, label_or_id: str):
        self.intent.mark_active(label_or_id)
        self.events.append("intent.step_active", "agent", step=label_or_id)

    def finish_step(self, detail: str = ""):
        self.intent.complete_active(detail)
        self.events.append("intent.step_done", "agent", detail=detail)

    def finish_task(self, summary: str):
        self.intent.finish(summary)
        self.events.append("intent.done", "agent", summary=summary)

    def fail_task(self, message: str):
        self.intent.fail(message)
        self.events.append("intent.blocked", "agent", message=message)

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

    def snapshot(self) -> dict[str, Any]:
        return {
            "robot_name": self.robot_name,
            "backend": self.backend.name,
            "robot_state": self.backend.get_state().to_dict(),
            "intent": self.intent.to_dict(),
            "events": self.events.tail(200),
            "tool_graph": self.tool_graph[-200:],
            "capabilities": self.capabilities.list(),
        }
