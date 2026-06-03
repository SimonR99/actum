# Architecture

Robo is a robot runtime first and an agent framework integration point second.

## Why Not Put an Agent Framework at the Center?

Agent frameworks are useful for orchestration, tool routing, memory, MCP servers, and multi-agent workflows. They are not the right owner for physical safety, robot state, embodiment limits, or low-level action contracts.

The project therefore uses this split:

- **Core runtime owns:** robot state, capabilities, intent, event log, tool graph, backend actions, safety policy.
- **Adapters expose:** MCP tools, ROS 2 interfaces, LeRobot policies, web dashboard, CLI.

This keeps the robot dependable even if the agent framework changes.

## Runtime Layers

1. `RobotBackend`
   - Connects to physical or simulated robots.
   - Returns typed `ActionResult` and `RobotState`.
   - Current adapters: `fake`, `unitree_g1`, `lekiwi`.

2. `RobotRuntime`
   - Holds backend, `IntentState`, `EventLog`, capability registry, and tool graph.
   - Provides a serializable snapshot for dashboard and future MCP/ROS bridges.

3. `RobotTools`
   - LLM-visible tools.
   - Calls the runtime/backend instead of hardware directly.

4. `RobotAgent`
   - Owns the LLM conversation and speech/camera event loop.
   - Produces operator-visible intent with `set_plan`, `mark_step`, and `done`.

5. Dashboard
   - Shows chat, camera, action trace, plan/intent, memory, and tool graph.

## Framework Strategy

Preferred integration order:

1. **LeRobot** for robot datasets, policy training, policy rollout, LeKiwi, and Unitree G1 policy paths.
2. **ROS 2** for teams that already depend on Nav2, MoveIt, TF, action servers, or robot-specific drivers.
3. **MCP** as an external control surface over Robo capabilities.
4. Optional higher-level agent frameworks only as clients of MCP or as wrappers around `RobotRuntime`.

## Model Strategy

Use two classes of models:

1. **Executive models**
   - LLM/VLM used for language, planning, question answering, tool selection, and summarizing state.
   - Should emit structured intent and tool calls.

2. **Embodied policies**
   - VLA or imitation-learning policies used for short-horizon manipulation/navigation.
   - Should be invoked as capabilities with timeouts, safety checks, and validation.

Track these policy families:

- SmolVLA for efficient open VLA experiments on affordable hardware.
- pi0/pi0.5 through OpenPI for broader manipulation policy research.
- NVIDIA GR00T for humanoid policy direction, especially Unitree G1-style embodiments.
- OpenVLA for open-source VLA baselines and fine-tuning recipes.
- Gemini Robotics SDK if access is available and licensing fits.

## Safety Principle

Every physical action should pass through:

```
agent proposal -> capability schema -> safety supervisor -> backend action -> result -> observation
```

No agent framework, MCP client, or dashboard control should bypass this path.
