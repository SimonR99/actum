# Actum Reboot Plan

This repo is now moving from prototype toward a reference architecture for agentic robotics.

## Current Baseline

Implemented:

- `RobotRuntime`: shared runtime state for the agent, tools, backend, event log, tool graph, and dashboard.
- `RobotBackend`: common adapter interface.
- Backends: `laptop`, `fake`, `unitree_g1`, `lekiwi`.
- `CompanionPolicy`: always-on gate for direct vs passive events.
- `MemoryStore`: JSON-backed facts, people, places, spatial notes, and episodic observations.
- Configurable personality: name, persona, likes, principles, and speaking style.
- Local software tools: direct URL fetch plus optional MCP server bridge.
- `IntentState`: live goal, plan steps, active step, done/blocked state.
- `BehaviorTreeState`: operator-visible waiting/perception/action nodes driven by the background loop.
- Background autonomy loop for idle perception review and scheduled cron-style jobs.
- Live map observations and body perception state.
- Dashboard stream for intent, behavior tree, map/body state, settings, and tool graph.
- Runtime settings for model provider/API-key state and selectable tools.
- Swappable inference providers (the "brain"): on-device LiteRT/Gemma and OpenAI, routed by config.
- Selectable local compute backend (GPU/CPU/NPU) instead of a hardcoded GPU engine.
- Speed profiles (fast/balanced/power_saver) bundling provider, compute, loop rates, camera fps, and deliberation cadence.
- Self-organizing memory: relevance retrieval per turn plus periodic dedupe/trim consolidation.
- Self-directed deliberation tick so the robot can set its own tasks when idle.
- Lightweight imports and tests without requiring a robot or model runtime.
- No vendored Unitree SDK tree.

Still needed:

- Safety supervisor with speed, workspace, force, and approval gates.
- Capability registry that can be exported as MCP tools.
- Full MCP server adapter that exports Actum capabilities to external agents.
- ROS 2 adapter for teams already using ROS graph/action/service contracts.
- LeRobot policy execution adapter, including SmolVLA/pi0/GR00T policy wrappers where practical.
- Durable event store, vector retrieval, and dataset recorder.
- Real spatial map integration: SLAM/localization, object landmarks, and robot-frame/world-frame transforms.
- Richer frontend graph layout, timeline replay, pause/resume/cancel/e-stop controls, and persisted settings profiles.

## Architecture Direction

Keep the robotics runtime independent from agent frameworks:

```
RobotRuntime
  IntentState
  BehaviorTreeState
  MemoryStore
  CompanionPolicy
  CronRegistry
  SpatialMap
  BodyPerception
  RuntimeSettings
  EventLog
  ToolGraph
  CapabilityRegistry
  SafetySupervisor
  RobotBackend
```

Adapters can expose or consume this runtime:

- MCP server adapter for external LLM agents.
- MCP client bridge for trusted software/data tools such as search, files, databases, or browser automation.
- ROS 2 adapter for topics, services, actions, TF, MoveIt, Nav2.
- LeRobot adapter for robot state/action schemas, datasets, policies, and rollouts.
- Web dashboard adapter for human operators.

The important rule: agent frameworks call into the robot kernel. They do not own the robot kernel.

## Model and Policy Direction

Use a two-layer autonomy model:

1. **Executive agent**
   - Plans, asks questions, chooses skills, monitors progress.
   - Uses local VLM/LLM where possible.
   - Emits structured intent, not hidden chain-of-thought.

2. **Embodied policies**
   - Perform short-horizon visuomotor actions.
   - Prefer LeRobot-compatible policies first.
   - Candidate policy families to track: SmolVLA, pi0/pi0.5, GR00T, OpenVLA, Gemini Robotics SDK where accessible.

## Milestones

### Milestone 1: Runtime Kernel

- Harden capability registry schemas.
- Expand persistent memory from JSON to an append-only event store with retrieval.
- Add safety supervisor.
- Add fake backend simulation scenarios.
- Add laptop companion scenarios for passive vision/timer events.

### Milestone 2: Operator Control

- Add pause/resume/cancel/e-stop.
- Add approval gates for risky actions.
- Add timeline replay.
- Add graph layout for tool usage and plan execution.
- Surface companion decisions and personality in the dashboard.

### Milestone 3: Robot Integrations

- Harden Unitree G1 backend against real hardware.
- Harden LeKiwi backend against current LeRobot APIs.
- Add ROS 2 backend or bridge.
- Add backend conformance tests.

### Milestone 4: Policy Integrations

- Add LeRobot dataset recorder.
- Add policy rollout capability.
- Support async inference/chunked actions.
- Add model cards/config recipes for SmolVLA and pi0-family policies.

### Milestone 5: Reference Quality

- Documentation site.
- Reproducible examples.
- CI for tests/lint/build.
- Contribution and safety guidelines.
- Benchmark tasks and regression traces.
