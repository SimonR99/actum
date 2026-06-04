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

Dashboard (added 2026-06-03): combined chat+voice console in the right rail (Chat/Settings
toggle, default Chat), assistant replies render as Markdown (react-markdown). UI is bilingual
via `frontend/src/i18n.js` — **default French**, FR/EN toggle, remembered in localStorage.

Speech: TTS is local Kokoro (`tts.py`). STT is a selectable stage (`inference/stt.py`):
whisper (local faster-whisper, **default**) / openai (cloud) / model (multimodal passthrough),
switchable live in the UI. User preference: **local-first** — wanted Whisper default and the
engine pickable in the UI like a model chooser.

Voice bug fixed: the OpenAI provider was dropping audio; voice now transcribes via the
selected STT engine before reaching the LLM (agent `_build_content`), with audio passthrough
fallback for the multimodal model.

Still open (see [[docs/REBOOT_PLAN.md]]): safety supervisor (gating concern before more
physical autonomy), LeRobot/VLA policy execution, real SLAM/spatial map, vector memory,
Anthropic provider (currently raises NotImplemented), operator pause/resume/e-stop,
TTS voice/model selection in the UI (only STT is selectable so far).
