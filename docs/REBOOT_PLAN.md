# Robo Reboot Plan

This repo is now moving from prototype toward a reference architecture for agentic robotics.

## Current Baseline

Implemented:

- `RobotRuntime`: shared runtime state for the agent, tools, backend, event log, tool graph, and dashboard.
- `RobotBackend`: common adapter interface.
- Backends: `fake`, `unitree_g1`, `lekiwi`.
- `IntentState`: live goal, plan steps, active step, done/blocked state.
- Dashboard stream for intent and tool graph.
- Lightweight imports and tests without requiring a robot or model runtime.
- No vendored Unitree SDK tree.

Still needed:

- Safety supervisor with speed, workspace, force, and approval gates.
- Capability registry that can be exported as MCP tools.
- ROS 2 adapter for teams already using ROS graph/action/service contracts.
- LeRobot policy execution adapter, including SmolVLA/pi0/GR00T policy wrappers where practical.
- Persistent event store and dataset recorder.
- Richer frontend with graph layout, timeline replay, pause/resume/cancel/e-stop controls.

## Architecture Direction

Keep the robotics runtime independent from agent frameworks:

```
RobotRuntime
  IntentState
  EventLog
  ToolGraph
  CapabilityRegistry
  SafetySupervisor
  RobotBackend
```

Adapters can expose or consume this runtime:

- MCP server adapter for external LLM agents.
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

- Finish capability registry.
- Add safety supervisor.
- Add persistent event log.
- Add fake backend simulation scenarios.

### Milestone 2: Operator Control

- Add pause/resume/cancel/e-stop.
- Add approval gates for risky actions.
- Add timeline replay.
- Add graph layout for tool usage and plan execution.

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
