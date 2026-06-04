---
name: actum-goals
description: Actum project north-star — maximally autonomous, configurable robot runtime
metadata:
  type: project
---

Actum's goal is a robot that is as autonomous as possible: it should auto-set its own
tasks, organise its own memory, and run on multiple hardware targets with a user-selectable
speed/cost/power tradeoff.

Implemented (2026-06-03): swappable `InferenceProvider` brain (LiteRT local + OpenAI cloud,
routed by `models.active_provider`), selectable LiteRT compute backend (GPU/CPU/NPU),
speed `profiles` (fast/balanced/power_saver) in config.json that bundle provider+compute+loop
rates+camera fps+deliberation cadence, memory retrieval-per-turn + periodic self-consolidation,
and a self-directed `deliberate` background tick. Legacy `hardware.*` config and
`SENSORIMOTOR_*`/`ROBO_*` env fallbacks were removed. `.env` is auto-loaded for API keys.

Still open (see [[docs/REBOOT_PLAN.md]]): safety supervisor (gating concern before more
physical autonomy), LeRobot/VLA policy execution, real SLAM/spatial map, vector memory,
Anthropic provider (currently raises NotImplemented), operator pause/resume/e-stop.
