import {
  Bot,
  BrainCircuit,
  Camera,
  CheckCircle2,
  Clock3,
  Cpu,
  Eye,
  Map,
  MessageSquare,
  Mic2,
  Network,
  Save,
  Send,
  Settings,
  SlidersHorizontal,
  Wrench
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { LANGUAGES, useT } from "./i18n.js";

const DEFAULT_ROBOT = {
  backend: "laptop",
  laptop: { webcam: true, microphone: true, speaker: true },
  fake: {},
  unitree_g1: { network_interface: "eth0", speaker_id: 0, volume: 80 },
  lekiwi: { remote_ip: "127.0.0.1", port: 5555, id: "lekiwi" }
};

const DEFAULT_MODELS = {
  active_provider: "local",
  providers: {
    local: { enabled: true, model: "litert-community/gemma-4-E2B-it-litert-lm" },
    openai: { enabled: false, model: "", api_key_configured: false },
    anthropic: { enabled: false, model: "", api_key_configured: false }
  }
};

function cx(...parts) {
  return parts.filter(Boolean).join(" ");
}

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function short(value, max = 120) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || data.message || `Request failed: ${response.status}`);
  }
  return data;
}

// Pull the operator-visible reply out of a completed turn: prefer what the robot
// actually said, then fall back to status/done text for direct interactions.
function assistantTextFromTurn(turn) {
  const actions = turn.actions || [];
  const spoken = actions.filter((a) => a.type === "speak" && a.text).map((a) => a.text);
  if (spoken.length) return spoken.join("\n\n");
  const direct = ["chat", "language", "voice", "text"].includes(turn.source);
  if (!direct) return "";
  const status = actions.filter((a) => a.type === "status" && a.message).map((a) => a.message);
  if (status.length) return status.join("\n\n");
  const done = actions.find((a) => a.type === "done" && a.summary);
  return done ? done.summary : "";
}

async function startBrowserWavRecorder() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Browser microphone capture is unavailable");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  });
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const chunks = [];

  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(audioContext.destination);

  async function close() {
    processor.disconnect();
    source.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    if (audioContext.state !== "closed") {
      await audioContext.close();
    }
  }

  return {
    async stop() {
      const sampleRate = audioContext.sampleRate;
      await close();
      return arrayBufferToBase64(encodeWav(chunks, sampleRate));
    },
    cancel: close
  };
}

function encodeWav(chunks, sampleRate) {
  const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + sampleCount * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, sampleCount * 2, true);

  let offset = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return buffer;
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return window.btoa(binary);
}

function useActumState() {
  const { t } = useT();
  const [state, setState] = useState(null);
  const [frame, setFrame] = useState("");
  const [wsLive, setWsLive] = useState(false);
  const [cameraLive, setCameraLive] = useState(false);
  const [lastFrameAt, setLastFrameAt] = useState("");
  const [toast, setToast] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const latestFrameId = useRef(0);
  const latestFrameSentAt = useRef(0);
  const frameRaf = useRef(0);

  const showToast = (message) => {
    setToast(message);
    window.clearTimeout(showToast._timer);
    showToast._timer = window.setTimeout(() => setToast(""), 2200);
  };

  const pushMessage = (message) => setMessages((current) => [...current, { id: uid(), ...message }]);

  const refresh = async () => {
    const payload = await api("/state");
    setState(payload);
  };

  useEffect(() => {
    refresh().catch((error) => showToast(error.message));
  }, []);

  useEffect(() => {
    let closed = false;
    let socket;
    let retry;

    function connect() {
      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${window.location.host}/events`);

      socket.onopen = () => setWsLive(true);
      socket.onclose = () => {
        setWsLive(false);
        if (!closed) retry = window.setTimeout(connect, 1200);
      };
      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "state") setState(msg.data);
        if (msg.type === "frame") {
          const frameId = Number(msg.frame_id || 0);
          const sentAt = Number(msg.sent_at || 0);
          if (frameId && frameId <= latestFrameId.current) return;
          if (!frameId && sentAt && sentAt <= latestFrameSentAt.current) return;

          latestFrameId.current = frameId || latestFrameId.current;
          latestFrameSentAt.current = sentAt || Date.now() / 1000;
          window.cancelAnimationFrame(frameRaf.current);
          frameRaf.current = window.requestAnimationFrame(() => {
            setFrame(`data:image/jpeg;base64,${msg.jpeg}`);
            setCameraLive(true);
            setLastFrameAt(new Date((sentAt || latestFrameSentAt.current) * 1000).toLocaleTimeString());
          });
        }
        if (msg.type === "turn") {
          if (msg.ignored) {
            setBusy(false);
            showToast(msg.reason || t("toast.passiveIgnored"));
            return;
          }
          if (msg.state_only) return;
          setBusy(false);
          const text = assistantTextFromTurn(msg);
          if (text) {
            setMessages((current) => [
              ...current,
              { id: uid(), role: "assistant", source: msg.source || "", text }
            ]);
          }
        }
      };
    }

    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      window.cancelAnimationFrame(frameRaf.current);
      if (socket) socket.close();
    };
  }, []);

  return {
    state,
    frame,
    wsLive,
    cameraLive,
    lastFrameAt,
    toast,
    showToast,
    refresh,
    messages,
    pushMessage,
    busy,
    setBusy
  };
}

export default function App() {
  const { t } = useT();
  const {
    state,
    frame,
    wsLive,
    cameraLive,
    lastFrameAt,
    toast,
    showToast,
    refresh,
    messages,
    pushMessage,
    busy,
    setBusy
  } = useActumState();
  const [rightView, setRightView] = useState("chat");

  const robotName = state?.personality?.name || state?.robot_name || "dino";
  const backend = state?.backend || "unknown";
  const connected = Boolean(state?.robot_state?.connected);
  const modelSettings = state?.settings?.models || DEFAULT_MODELS;
  const activeProvider = modelSettings.active_provider || "local";
  const serverStatus = state?.server?.status || "starting";
  const serverReady = Boolean(state?.server?.ready);

  return (
    <div className="min-h-screen text-slate-900">
      <Header
        robotName={robotName}
        wsLive={wsLive}
        cameraLive={cameraLive}
        backend={backend}
        connected={connected}
        activeProvider={activeProvider}
        serverStatus={serverStatus}
        serverReady={serverReady}
      />

      <main className="grid h-[calc(100vh-72px)] min-h-[760px] grid-cols-[330px_minmax(520px,1fr)_450px] grid-rows-[minmax(420px,1fr)_300px] gap-5 p-5 max-[1280px]:h-auto max-[1280px]:grid-cols-2 max-[1280px]:grid-rows-none max-[860px]:grid-cols-1">
        <section className="panel row-span-2 flex animate-rise flex-col" style={{ animationDelay: "0ms" }}>
          <PanelHeader title={t("panel.intent")} meta={state?.intent?.status || t("meta.idle")} icon={BrainCircuit} index="01" />
          <IntentPanel state={state} />
        </section>

        <section className="panel flex animate-rise flex-col" style={{ animationDelay: "70ms" }}>
          <PanelHeader title={t("panel.perception")} meta={lastFrameAt || t("perception.waiting")} icon={Eye} index="02" />
          <PerceptionPanel state={state} frame={frame} />
        </section>

        <section className="panel row-span-2 flex min-h-0 animate-rise flex-col max-[1280px]:min-h-[640px]" style={{ animationDelay: "140ms" }}>
          <RightRail
            view={rightView}
            setView={setRightView}
            state={state}
            showToast={showToast}
            refresh={refresh}
            messages={messages}
            pushMessage={pushMessage}
            busy={busy}
            setBusy={setBusy}
          />
        </section>

        <section className="grid min-h-0 animate-rise grid-cols-2 gap-5 max-[860px]:grid-cols-1" style={{ animationDelay: "210ms" }}>
          <div className="panel flex min-h-0 flex-col">
            <PanelHeader title={t("panel.toolgraph")} meta={`${state?.tool_graph?.length || 0} ${t("toolgraph.calls")}`} icon={Network} index="03" />
            <ToolGraph nodes={state?.tool_graph || []} />
          </div>
          <div className="panel flex min-h-0 flex-col">
            <PanelHeader title={t("panel.maptimeline")} meta="memory" icon={Map} index="04" />
            <MapTimeline state={state} />
          </div>
        </section>
      </main>

      <div
        className={cx(
          "fixed bottom-5 right-5 z-50 flex max-w-md items-center gap-3 rounded-xl border border-white/10 bg-slate-950 py-3 pl-4 pr-5 text-sm font-medium text-white shadow-lift transition duration-300",
          toast ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-2 opacity-0"
        )}
      >
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
        {toast}
      </div>
    </div>
  );
}

function Header({ robotName, wsLive, cameraLive, backend, connected, activeProvider, serverStatus, serverReady }) {
  const { t } = useT();
  return (
    <header className="grid min-h-[72px] grid-cols-[1fr_auto] items-center gap-4 border-b border-slate-200 bg-white/80 px-5 backdrop-blur-md max-[860px]:grid-cols-1 max-[860px]:py-3">
      <div className="flex items-center gap-3.5">
        <span className="relative grid h-9 w-9 shrink-0 place-items-center rounded-xl border border-slate-200 bg-slate-50">
          <span className={cx("h-2.5 w-2.5 rounded-full transition-colors", wsLive ? "animate-live-ping bg-emerald-500" : "bg-slate-300")} />
        </span>
        <div className="min-w-0">
          <div className="flex items-baseline gap-2">
            <span className="truncate text-lg font-extrabold leading-6 tracking-tight">{robotName}</span>
            <span className="mono-label hidden sm:inline">actum</span>
          </div>
          <div className="truncate font-mono text-[11px] text-slate-500">{t("app.subtitle")}</div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-end gap-2">
        <StatusChip ok={wsLive} label={wsLive ? t("status.ws.live") : t("status.ws.offline")} />
        <StatusChip ok={serverReady} label={`${t("status.agent")} ${serverStatus}`} />
        <StatusChip ok={cameraLive} label={cameraLive ? t("status.camera.live") : t("status.camera.waiting")} />
        <StatusChip ok={connected} label={`${t("status.backend")} ${backend}`} />
        <span className="chip">{t("status.model")} {activeProvider}</span>
        <LanguageToggle />
      </div>
    </header>
  );
}

function LanguageToggle() {
  const { lang, setLang } = useT();
  return (
    <div className="inline-flex overflow-hidden rounded-full border border-slate-200 bg-white p-0.5">
      {LANGUAGES.map((option) => (
        <button
          key={option.id}
          className={cx(
            "h-6 rounded-full px-3 font-mono text-[11px] font-semibold uppercase tracking-tight transition",
            lang === option.id ? "bg-slate-900 text-white" : "text-slate-500 hover:text-slate-900"
          )}
          onClick={() => setLang(option.id)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function StatusChip({ ok, label }) {
  return (
    <span className={cx("chip", ok ? "chip-ok" : "chip-warn")}>
      <span className={cx("h-1.5 w-1.5 rounded-full", ok ? "bg-emerald-500" : "bg-amber-700/70")} />
      {label}
    </span>
  );
}

function PanelHeader({ title, meta, icon: Icon, right, index }) {
  return (
    <div className="panel-header">
      <div className="flex min-w-0 items-center gap-2.5">
        {index ? <span className="mono-label text-blue-600/70">{index}</span> : null}
        <Icon className="h-4 w-4 text-slate-400" />
        <div className="panel-title">{title}</div>
      </div>
      {right || <div className="panel-meta">{meta}</div>}
    </div>
  );
}

// ── Right rail: Chat + Settings ────────────────────────────────────────────────

function RightRail({ view, setView, state, showToast, refresh, messages, pushMessage, busy, setBusy }) {
  const { t } = useT();
  const tabs = [
    { id: "chat", label: t("panel.chat"), icon: MessageSquare },
    { id: "settings", label: t("panel.settings"), icon: Settings }
  ];
  return (
    <>
      <div className="panel-header">
        <div className="inline-flex gap-1 rounded-lg bg-slate-100 p-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={cx(
                  "inline-flex h-8 items-center gap-2 rounded-md px-3 text-sm font-semibold transition",
                  view === tab.id ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"
                )}
                onClick={() => setView(tab.id)}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
        <div className="panel-meta">{view === "chat" ? t("panel.console") : t("settings.operator")}</div>
      </div>
      {view === "chat" ? (
        <ChatPanel
          state={state}
          showToast={showToast}
          refresh={refresh}
          messages={messages}
          pushMessage={pushMessage}
          busy={busy}
          setBusy={setBusy}
        />
      ) : (
        <SettingsPanel state={state} showToast={showToast} refresh={refresh} />
      )}
    </>
  );
}

function ChatPanel({ state, showToast, refresh, messages, pushMessage, busy, setBusy }) {
  const { t } = useT();
  const [command, setCommand] = useState("");
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef(null);
  const scrollRef = useRef(null);
  const robotName = state?.personality?.name || state?.robot_name || "dino";

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, busy]);

  useEffect(() => () => recorderRef.current?.cancel?.(), []);

  async function queueTrigger(source, payload = {}) {
    try {
      await api(`/trigger/${source}`, {
        method: "POST",
        body: JSON.stringify({
          text: payload.text || "",
          audio: payload.audio || "",
          force: source !== "vision",
          importance: source === "vision" ? 0.6 : 1
        })
      });
      await refresh();
    } catch (error) {
      setBusy(false);
      showToast(error.message);
    }
  }

  async function sendText() {
    const text = command.trim();
    if (!text) return;
    pushMessage({ role: "user", text });
    setCommand("");
    setBusy(true);
    await queueTrigger("chat", { text });
  }

  async function toggleVoice() {
    if (recording) {
      const recorder = recorderRef.current;
      recorderRef.current = null;
      setRecording(false);
      if (!recorder) return;
      try {
        const audio = await recorder.stop();
        pushMessage({ role: "user", kind: "voice", text: t("chat.voiceMessage") });
        setBusy(true);
        await queueTrigger("language", { audio });
      } catch (error) {
        showToast(error.message);
      }
      return;
    }
    try {
      recorderRef.current = await startBrowserWavRecorder();
      setRecording(true);
      showToast(t("chat.recording"));
    } catch (error) {
      showToast(error.message);
    }
  }

  async function lookNow() {
    setBusy(true);
    await queueTrigger("vision");
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-auto p-4">
        {messages.length ? (
          messages.map((message) => <ChatBubble key={message.id} message={message} robotName={robotName} />)
        ) : (
          <Empty text={t("chat.empty")} />
        )}
        {busy ? <ThinkingBubble robotName={robotName} /> : null}
      </div>

      <div className="border-t border-slate-200 p-3">
        <div className="flex items-end gap-2">
          <textarea
            className="field h-auto max-h-32 min-h-[40px] flex-1 resize-none py-2 leading-5"
            rows={1}
            value={command}
            placeholder={t("chat.placeholder")}
            onChange={(event) => setCommand(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendText();
              }
            }}
          />
          <button
            className={cx("btn h-10 w-10 shrink-0 p-0", recording && "btn-recording")}
            title={recording ? t("chat.stop") : t("chat.voice")}
            onClick={toggleVoice}
          >
            <Mic2 className="h-4 w-4" />
          </button>
          <button className="btn h-10 w-10 shrink-0 p-0" title={t("chat.vision")} onClick={lookNow}>
            <Camera className="h-4 w-4" />
          </button>
          <button className="btn btn-primary h-10 w-10 shrink-0 p-0" title={t("chat.send")} onClick={sendText}>
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

function ChatBubble({ message, robotName }) {
  const { t } = useT();
  const isUser = message.role === "user";
  return (
    <div className={cx("flex animate-rise", isUser ? "justify-end" : "justify-start")}>
      <div className={cx("max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm shadow-sm", isUser ? "rounded-br-md bg-blue-600 text-white shadow-[0_8px_18px_-10px_rgba(31,68,245,0.65)]" : "rounded-bl-md border border-slate-200 bg-white text-slate-900")}>
        <div className={cx("mb-1 font-mono text-[10px] font-semibold uppercase tracking-label", isUser ? "text-blue-100" : "text-slate-400")}>
          {isUser ? t("chat.you") : robotName}
          {!isUser && message.source && !["chat", "language", "voice", "text"].includes(message.source) ? ` · ${message.source}` : ""}
        </div>
        {isUser ? <div className="whitespace-pre-wrap break-words leading-relaxed">{message.text}</div> : <Markdown text={message.text} />}
      </div>
    </div>
  );
}

function ThinkingBubble({ robotName }) {
  const { t } = useT();
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl rounded-bl-md border border-slate-200 bg-white px-3.5 py-2.5 text-sm text-slate-500 shadow-sm">
        <div className="mb-1 font-mono text-[10px] font-semibold uppercase tracking-label text-slate-400">{robotName}</div>
        <div className="flex items-center gap-2">
          <span className="flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.2s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.1s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
          </span>
          {t("chat.thinking")}
        </div>
      </div>
    </div>
  );
}

function Markdown({ text }) {
  return (
    <div className="md break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{ a: (props) => <a {...props} target="_blank" rel="noreferrer" /> }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

// ── Intent ─────────────────────────────────────────────────────────────────────

function IntentPanel({ state }) {
  const { t } = useT();
  const intent = state?.intent || {};
  const behavior = state?.behavior || {};
  const goal = intent.goal || behavior.goal || t("intent.noTask");
  const steps = intent.steps || [];
  const nodes = behavior.nodes || [];

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="relative border-b border-slate-200 bg-gradient-to-b from-slate-50/80 to-transparent p-5">
        <div className="mono-label flex items-center gap-2">
          <span className="h-1 w-1 rounded-full bg-blue-600" />
          {t("intent.goal")}
        </div>
        <div className="mt-2.5 text-2xl font-extrabold leading-8 tracking-tight text-slate-900">{goal}</div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        <SectionTitle title={t("intent.plan")} meta={`${steps.length} ${t("intent.steps")}`} />
        <div className="mt-3 grid gap-2">
          {steps.length ? (
            steps.map((step, index) => <StepRow key={step.id || index} step={step} index={index} />)
          ) : (
            <Empty text={t("intent.waitingPlan")} />
          )}
        </div>

        <div className="mt-6">
          <SectionTitle title={t("intent.behaviorTree")} meta={`${behavior.tick_count || 0} ${t("intent.ticks")}`} />
          <div className="mt-3 grid gap-2">
            {nodes.length ? nodes.map((node) => <BehaviorNode key={node.id || node.label} node={node} />) : <Empty text={t("intent.noNodes")} />}
          </div>
        </div>
      </div>
    </div>
  );
}

function StepRow({ step, index }) {
  const status = step.status || "pending";
  return (
    <div className={cx("rounded-xl border p-3 transition", statusTone(status))}>
      <div className="flex items-start gap-3">
        <div className="grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-slate-200 bg-white font-mono text-[11px] font-bold tabular-nums text-slate-600">
          {status === "done" ? <CheckCircle2 className="h-4 w-4 text-emerald-600" /> : String(index + 1).padStart(2, "0")}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold leading-5 text-slate-900">{step.label || step.id}</div>
          <div className="mono-label mt-1">{status}</div>
        </div>
      </div>
    </div>
  );
}

function BehaviorNode({ node }) {
  return (
    <div className={cx("rounded-xl border p-3 transition", statusTone(node.status || "waiting"))}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-slate-900">{node.label || node.id}</div>
          <div className="mono-label mt-1 normal-case">
            {node.kind || "node"} · {node.status || "waiting"}
          </div>
        </div>
        <span className="shrink-0 rounded-md border border-slate-200 bg-white px-2 py-1 font-mono text-[10px] font-medium text-slate-500">{node.id}</span>
      </div>
      {node.detail ? <div className="mt-2 text-xs leading-relaxed text-slate-600">{node.detail}</div> : null}
    </div>
  );
}

function statusTone(status) {
  if (status === "active") return "border-blue-200 bg-blue-50 ring-1 ring-blue-100";
  if (status === "done") return "border-emerald-200 bg-emerald-50";
  if (status === "blocked" || status === "failed") return "border-red-200 bg-red-50";
  return "border-slate-200 bg-slate-50";
}

// ── Perception ─────────────────────────────────────────────────────────────────

function PerceptionPanel({ state, frame }) {
  const { t } = useT();
  const body = state?.body || {};
  const memory = state?.memory || {};
  const recent = (memory.recent || []).filter((item) => ["observation", "spatial"].includes(item.kind)).slice(-4).reverse();

  return (
    <div className="grid min-h-0 flex-1 grid-cols-[minmax(360px,1fr)_320px] gap-4 p-4 max-[1180px]:grid-cols-1">
      <div className="min-h-0">
        <div className="relative aspect-[4/3] overflow-hidden rounded-xl border border-slate-300 bg-slate-950 shadow-inner ring-1 ring-black/5">
          {frame ? (
            <img className="h-full w-full object-contain" src={frame} alt="Robot camera feed" />
          ) : (
            <div className="grid h-full place-items-center gap-2 text-center">
              <Camera className="mx-auto h-6 w-6 text-slate-600" />
              <span className="font-mono text-[11px] uppercase tracking-label text-slate-500">{t("perception.noFrame")}</span>
            </div>
          )}
          <span className="absolute left-2.5 top-2.5 flex items-center gap-1.5 rounded-md bg-black/45 px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-label text-white/90 backdrop-blur-sm">
            <span className={cx("h-1.5 w-1.5 rounded-full", frame ? "animate-live-ping bg-emerald-400" : "bg-slate-500")} />
            cam
          </span>
        </div>
      </div>

      <div className="grid min-h-0 content-start gap-4 overflow-auto">
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <SectionTitle title={t("perception.body")} meta={body.posture || t("perception.unknown")} />
          <div className="mt-3 text-sm font-semibold">{body.summary || t("perception.bodyUnavailable")}</div>
          <InfoGrid
            rows={[
              [t("perception.holding"), body.holding || t("perception.nothing")],
              [t("perception.contacts"), (body.contacts || []).join(", ") || t("perception.none")],
              [t("perception.pose"), Object.keys(body.base_pose || {}).length ? JSON.stringify(body.base_pose) : t("perception.none")],
              [t("perception.joints"), Object.keys(body.joints || {}).length ? `${Object.keys(body.joints).length} ${t("perception.values")}` : t("perception.none")]
            ]}
          />
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <SectionTitle title={t("perception.recent")} meta={`${recent.length} ${t("perception.items")}`} />
          <div className="mt-3 grid gap-2">
            {recent.length ? recent.map((item) => <MemoryRow key={item.id || item.summary} item={item} />) : <Empty text={t("perception.noObs")} />}
          </div>
        </div>
      </div>
    </div>
  );
}

function ToolGraph({ nodes }) {
  const { t } = useT();
  const visible = nodes.slice(-20);
  return (
    <div className="min-h-0 flex-1 overflow-auto p-4">
      <div className="grid gap-3">
        {visible.length ? (
          visible.map((node, index) => {
            const ok = node.result ? node.result.ok : null;
            return (
              <div key={node.id || index} className="grid grid-cols-[28px_minmax(0,1fr)] gap-3">
                <div className="relative flex justify-center">
                  {index < visible.length - 1 ? <span className="absolute top-7 h-[calc(100%-12px)] w-px bg-slate-200" /> : null}
                  <div className={cx("relative z-10 grid h-7 w-7 place-items-center rounded-lg font-mono text-[11px] font-bold tabular-nums", ok === false ? "bg-red-600 text-white" : ok === true ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-600")}>
                    {String(index + 1).padStart(2, "0")}
                  </div>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-mono text-sm font-semibold text-slate-900">{node.type || "tool"}</div>
                      <div className="mt-1 text-xs text-slate-500">{node.result ? node.result.message : t("toolgraph.queued")}</div>
                    </div>
                    <span className="shrink-0 rounded-md border border-slate-200 bg-slate-50 px-2 py-1 font-mono text-[10px] text-slate-500">{node.id}</span>
                  </div>
                  <div className="mt-2 font-mono text-[11px] leading-relaxed text-slate-600">{summarizeAction(node.action || {}, t)}</div>
                </div>
              </div>
            );
          })
        ) : (
          <Empty text={t("toolgraph.none")} />
        )}
      </div>
    </div>
  );
}

function summarizeAction(action, t) {
  const ignored = new Set(["type", "time", "_node_id", "ok", "backend"]);
  return (
    Object.entries(action)
      .filter(([key, value]) => !ignored.has(key) && value !== "" && value !== null && value !== undefined)
      .slice(0, 3)
      .map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`)
      .join(" · ") || t("toolgraph.noArgs")
  );
}

function MapTimeline({ state }) {
  const { t } = useT();
  const memory = state?.memory || {};
  const map = state?.map || {};
  const events = state?.events || [];
  const observations = map.observations || [];
  const counts = memory.counts || {};

  return (
    <div className="min-h-0 flex-1 overflow-auto p-4">
      <div className="grid grid-cols-2 gap-3">
        <Metric label={t("map.facts")} value={counts.facts || 0} />
        <Metric label={t("map.places")} value={counts.places || 0} />
        <Metric label={t("map.map")} value={observations.length} />
        <Metric label={t("map.events")} value={events.length} />
      </div>

      <div className="mt-4">
        <SectionTitle title={t("map.map")} meta={`${observations.length} ${t("map.observations")}`} />
        <div className="mt-3 grid gap-2">
          {observations.slice(-4).reverse().map((item) => (
            <MemoryRow key={item.id} item={{ kind: item.place || "map", summary: item.summary }} />
          ))}
          {!observations.length ? <Empty text={t("map.noObs")} /> : null}
        </div>
      </div>

      <div className="mt-4">
        <SectionTitle title={t("timeline.title")} meta={t("timeline.latest")} />
        <div className="mt-3 grid gap-2">
          {events.slice(-8).reverse().map((event, index) => {
            const type = event.type || "event";
            const data = event.data || {};
            return (
              <div key={`${type}-${index}`} className="rounded-xl border border-slate-200 bg-white p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-blue-600/60" />
                    <div className="truncate font-mono text-[12px] font-semibold text-slate-900">{type}</div>
                  </div>
                  <div className="shrink-0 font-mono text-[11px] tabular-nums text-slate-400">{event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : ""}</div>
                </div>
                <div className="mt-1.5 pl-3.5 text-xs leading-relaxed text-slate-500">{short(data.message || data.summary || data.goal || data.step || data.tool || event.message || JSON.stringify(data), 140)}</div>
              </div>
            );
          })}
          {!events.length ? <Empty text={t("timeline.none")} /> : null}
        </div>
      </div>
    </div>
  );
}

// ── Settings ───────────────────────────────────────────────────────────────────

function SettingsPanel({ state, showToast, refresh }) {
  const { t } = useT();
  const tabs = [
    { id: "robot", label: t("settings.tab.robot"), icon: Bot },
    { id: "model", label: t("settings.tab.model"), icon: Cpu },
    { id: "tools", label: t("settings.tab.tools"), icon: Wrench },
    { id: "cron", label: t("settings.tab.cron"), icon: Clock3 }
  ];
  const [activeTab, setActiveTab] = useState("robot");

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid grid-cols-4 border-b border-slate-200 p-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              className={cx("flex h-10 items-center justify-center gap-2 rounded-md text-sm font-semibold text-slate-600", activeTab === tab.id && "bg-slate-950 text-white")}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon className="h-4 w-4" />
              <span className="max-[1440px]:sr-only">{tab.label}</span>
            </button>
          );
        })}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-4">
        {activeTab === "robot" ? <RobotSettings state={state} showToast={showToast} refresh={refresh} /> : null}
        {activeTab === "model" ? <ModelSettings state={state} showToast={showToast} refresh={refresh} /> : null}
        {activeTab === "tools" ? <ToolSettings state={state} showToast={showToast} refresh={refresh} /> : null}
        {activeTab === "cron" ? <CronSettings state={state} showToast={showToast} refresh={refresh} /> : null}
      </div>
    </div>
  );
}

function RobotSettings({ state, showToast, refresh }) {
  const { t } = useT();
  const [dirty, setDirty] = useState(false);
  const [name, setName] = useState("dino");
  const [robot, setRobot] = useState(DEFAULT_ROBOT);
  const [persist, setPersist] = useState(true);

  useEffect(() => {
    if (dirty || !state) return;
    setName(state.personality?.name || state.robot_name || "dino");
    setRobot(mergeRobot(state.robot_config || {}));
  }, [state, dirty]);

  const backend = robot.backend || "laptop";

  function update(path, value) {
    setDirty(true);
    setRobot((current) => setPath(current, path, value));
  }

  async function apply() {
    try {
      const payload = await api("/settings/robot", {
        method: "POST",
        body: JSON.stringify({ name: name.trim() || "dino", robot, persist })
      });
      showToast(payload.message || t("toast.robotUpdated"));
      setDirty(false);
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  return (
    <div className="grid gap-4">
      <div className="grid gap-3">
        <Labeled label={t("robot.name")}>
          <input className="field" value={name} onChange={(event) => { setDirty(true); setName(event.target.value); }} />
        </Labeled>
        <Labeled label={t("robot.backend")}>
          <select className="field" value={backend} onChange={(event) => update(["backend"], event.target.value)}>
            <option value="laptop">Laptop</option>
            <option value="fake">Fake</option>
            <option value="unitree_g1">Unitree G1</option>
            <option value="lekiwi">LeKiwi</option>
          </select>
        </Labeled>
      </div>

      {backend === "laptop" ? (
        <div className="grid gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <Toggle label={t("robot.webcam")} checked={robot.laptop.webcam} onChange={(value) => update(["laptop", "webcam"], value)} />
          <Toggle label={t("robot.microphone")} checked={robot.laptop.microphone} onChange={(value) => update(["laptop", "microphone"], value)} />
          <Toggle label={t("robot.speaker")} checked={robot.laptop.speaker} onChange={(value) => update(["laptop", "speaker"], value)} />
        </div>
      ) : null}

      {backend === "unitree_g1" ? (
        <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <Labeled label={t("robot.networkInterface")}>
            <input className="field" value={robot.unitree_g1.network_interface} onChange={(event) => update(["unitree_g1", "network_interface"], event.target.value)} />
          </Labeled>
          <div className="grid grid-cols-2 gap-3">
            <Labeled label={t("robot.speakerId")}>
              <input className="field" type="number" min="0" max="1" value={robot.unitree_g1.speaker_id} onChange={(event) => update(["unitree_g1", "speaker_id"], Number(event.target.value || 0))} />
            </Labeled>
            <Labeled label={t("robot.volume")}>
              <input className="field" type="number" min="0" max="100" value={robot.unitree_g1.volume} onChange={(event) => update(["unitree_g1", "volume"], Number(event.target.value || 80))} />
            </Labeled>
          </div>
        </div>
      ) : null}

      {backend === "lekiwi" ? (
        <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <Labeled label={t("robot.remoteIp")}>
            <input className="field" value={robot.lekiwi.remote_ip} onChange={(event) => update(["lekiwi", "remote_ip"], event.target.value)} />
          </Labeled>
          <div className="grid grid-cols-2 gap-3">
            <Labeled label={t("robot.port")}>
              <input className="field" type="number" min="1" max="65535" value={robot.lekiwi.port} onChange={(event) => update(["lekiwi", "port"], Number(event.target.value || 5555))} />
            </Labeled>
            <Labeled label={t("robot.id")}>
              <input className="field" value={robot.lekiwi.id} onChange={(event) => update(["lekiwi", "id"], event.target.value)} />
            </Labeled>
          </div>
        </div>
      ) : null}

      <Toggle label={t("robot.saveConfig")} checked={persist} onChange={setPersist} />
      <button className="btn btn-primary w-full" onClick={apply}>
        <Save className="h-4 w-4" />
        {t("robot.apply")}
      </button>
    </div>
  );
}

function ModelSettings({ state, showToast, refresh }) {
  const { t } = useT();
  const models = state?.settings?.models || DEFAULT_MODELS;
  const providers = models.providers || DEFAULT_MODELS.providers;
  const profile = state?.profile || {};
  const profileOptions = profile.available || ["fast", "balanced", "power_saver"];
  const speech = state?.settings?.speech || { stt_engine: "whisper", available_stt: ["whisper", "openai", "model"] };
  const sttOptions = speech.available_stt || ["whisper", "openai", "model"];
  const [provider, setProvider] = useState(models.active_provider || "local");
  const [model, setModel] = useState("");
  const [enabled, setEnabled] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [persist, setPersist] = useState(false);
  const [persistSecret, setPersistSecret] = useState(false);

  useEffect(() => {
    const cfg = providers[provider] || {};
    setModel(cfg.model || "");
    setEnabled(Boolean(cfg.enabled));
    setApiKey("");
  }, [provider, state]);

  async function applyProfile(next) {
    try {
      const payload = await api("/settings/profile", {
        method: "POST",
        body: JSON.stringify({ profile: next, persist: true })
      });
      showToast(payload.message || t("toast.profileUpdated"));
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  async function applySpeech(next) {
    try {
      const payload = await api("/settings/speech", {
        method: "POST",
        body: JSON.stringify({ stt_engine: next, persist: true })
      });
      showToast(payload.message || t("toast.speechUpdated"));
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  async function apply() {
    try {
      const payload = await api("/settings/model", {
        method: "POST",
        body: JSON.stringify({ provider, model, api_key: apiKey, enabled, persist, persist_secret: persistSecret })
      });
      showToast(payload.message || t("toast.modelUpdated"));
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  const configured = Boolean(providers[provider]?.api_key_configured);

  return (
    <div className="grid gap-4">
      <Labeled label={t("profile.label")}>
        <select className="field" value={profile.active || "balanced"} onChange={(event) => applyProfile(event.target.value)}>
          {profileOptions.map((name) => (
            <option key={name} value={name}>{t(`profile.${name}`)}</option>
          ))}
        </select>
      </Labeled>

      <Labeled label={t("speech.stt")}>
        <select className="field" value={speech.stt_engine || "whisper"} onChange={(event) => applySpeech(event.target.value)}>
          {sttOptions.map((name) => (
            <option key={name} value={name}>{t(`stt.${name}`)}</option>
          ))}
        </select>
      </Labeled>

      <TranscriptionTest showToast={showToast} />

      <Labeled label={t("model.provider")}>
        <select className="field" value={provider} onChange={(event) => setProvider(event.target.value)}>
          <option value="local">Local</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
        </select>
      </Labeled>
      <Labeled label={t("model.model")}>
        <input className="field" value={model} onChange={(event) => setModel(event.target.value)} placeholder={t("model.model")} />
      </Labeled>
      <Labeled label={configured ? t("model.apiKeyConfigured") : t("model.apiKey")}>
        <input className="field" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder={configured ? t("model.apiKeyKeep") : t("model.apiKey")} />
      </Labeled>
      <div className="grid gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
        <Toggle label={t("model.providerEnabled")} checked={enabled} onChange={setEnabled} />
        <Toggle label={t("model.save")} checked={persist} onChange={setPersist} />
        <Toggle label={t("model.saveKey")} checked={persistSecret} onChange={setPersistSecret} />
      </div>
      <button className="btn btn-primary w-full" onClick={apply}>
        <SlidersHorizontal className="h-4 w-4" />
        {t("model.apply")}
      </button>
    </div>
  );
}

// Isolated STT probe — records a browser WAV and runs it through /debug/transcribe,
// which exercises only the speech engine (works even if the agent runtime is down).
function TranscriptionTest({ showToast }) {
  const { t } = useT();
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const recorderRef = useRef(null);

  useEffect(() => () => recorderRef.current?.cancel?.(), []);

  async function toggle() {
    if (busy) return;
    if (recording) {
      const recorder = recorderRef.current;
      recorderRef.current = null;
      setRecording(false);
      if (!recorder) return;
      setBusy(true);
      try {
        const audio = await recorder.stop();
        setResult(await api("/debug/transcribe", { method: "POST", body: JSON.stringify({ audio }) }));
      } catch (error) {
        setResult({ ok: false, error: error.message });
      } finally {
        setBusy(false);
      }
      return;
    }
    try {
      recorderRef.current = await startBrowserWavRecorder();
      setResult(null);
      setRecording(true);
      showToast(t("chat.recording"));
    } catch (error) {
      showToast(error.message);
    }
  }

  const tone = result?.ok && result?.transcript
    ? "border-emerald-200 bg-emerald-50"
    : result?.error
      ? "border-red-200 bg-red-50"
      : "border-slate-200 bg-white";

  return (
    <div className="card-inset grid gap-3">
      <SectionTitle title={t("stt.test")} meta={result?.engine || ""} />
      <p className="text-xs leading-relaxed text-slate-500">{t("stt.testHint")}</p>
      <button
        className={cx("btn w-full", recording ? "btn-recording" : !busy && "btn-primary")}
        onClick={toggle}
        disabled={busy}
      >
        {recording ? <span className="h-2 w-2 animate-pulse rounded-full bg-white" /> : <Mic2 className="h-4 w-4" />}
        {busy ? t("stt.testRunning") : recording ? t("stt.testStop") : t("stt.testRecord")}
      </button>

      {result ? (
        <div className={cx("rounded-xl border p-3", tone)}>
          {result.transcript ? (
            <>
              <div className="mono-label text-slate-500">{t("stt.testTranscript")}</div>
              <div className="mt-1.5 text-sm leading-relaxed text-slate-900">{result.transcript}</div>
            </>
          ) : null}
          {result.error ? <div className="text-xs leading-relaxed text-red-600">{result.error}</div> : null}
          {result.note && !result.transcript ? <div className="text-xs leading-relaxed text-slate-500">{result.note}</div> : null}
          {typeof result.audio_bytes === "number" || typeof result.elapsed === "number" ? (
            <div className="mono-label mt-2 flex flex-wrap gap-x-3 gap-y-1 normal-case text-slate-400">
              {typeof result.audio_bytes === "number" ? <span>{(result.audio_bytes / 1024).toFixed(1)} KB</span> : null}
              {typeof result.elapsed === "number" ? <span>{result.elapsed}s</span> : null}
            </div>
          ) : null}
        </div>
      ) : (
        <div className="mono-label normal-case text-slate-400">{t("stt.testEmpty")}</div>
      )}
    </div>
  );
}

function ToolSettings({ state, showToast, refresh }) {
  const { t } = useT();
  const tools = state?.settings?.tools || {};
  const available = tools.available || [];
  const enabled = new Set(tools.enabled || []);
  const [persist, setPersist] = useState(false);

  async function setTool(tool, value) {
    try {
      const payload = await api("/settings/tool", {
        method: "POST",
        body: JSON.stringify({ tool, enabled: value, persist })
      });
      showToast(payload.message || t("toast.toolUpdated"));
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  return (
    <div className="grid gap-4">
      <Toggle label={t("tools.save")} checked={persist} onChange={setPersist} />
      <div className="grid grid-cols-2 gap-2 max-[1440px]:grid-cols-1">
        {available.map((tool) => (
          <label key={tool} className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm">
            <span className="truncate">{tool}</span>
            <input className="h-4 w-4 accent-blue-600" type="checkbox" checked={enabled.has(tool)} onChange={(event) => setTool(tool, event.target.checked)} />
          </label>
        ))}
      </div>
    </div>
  );
}

function CronSettings({ state, showToast, refresh }) {
  const { t } = useT();
  const jobs = state?.cron?.jobs || [];
  const [name, setName] = useState("");
  const [seconds, setSeconds] = useState(300);
  const [instruction, setInstruction] = useState("");

  async function add() {
    if (!name.trim() || !instruction.trim()) return;
    try {
      await api("/cron", {
        method: "POST",
        body: JSON.stringify({ name, every_seconds: Number(seconds || 300), instruction })
      });
      showToast(`${t("toast.scheduled")} ${name}`);
      setName("");
      setInstruction("");
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  return (
    <div className="grid gap-4">
      <div className="grid gap-3">
        <Labeled label={t("cron.name")}>
          <input className="field" value={name} onChange={(event) => setName(event.target.value)} placeholder={t("cron.namePh")} />
        </Labeled>
        <Labeled label={t("cron.everySeconds")}>
          <input className="field" type="number" min="1" value={seconds} onChange={(event) => setSeconds(Number(event.target.value || 300))} />
        </Labeled>
        <Labeled label={t("cron.instruction")}>
          <input className="field" value={instruction} onChange={(event) => setInstruction(event.target.value)} placeholder={t("cron.instructionPh")} />
        </Labeled>
        <button className="btn btn-primary" onClick={add}>
          <Clock3 className="h-4 w-4" />
          {t("cron.add")}
        </button>
      </div>

      <div className="grid gap-2">
        {jobs.length ? (
          jobs.map((job) => (
            <div key={job.id} className="rounded-xl border border-slate-200 bg-white p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="truncate text-sm font-semibold text-slate-900">{job.name}</div>
                <span className="shrink-0 rounded-md border border-slate-200 bg-slate-50 px-2 py-1 font-mono text-[10px] tabular-nums text-slate-500">{Math.round(job.every_seconds)}s</span>
              </div>
              <div className="mt-1.5 text-xs leading-relaxed text-slate-500">{job.instruction}</div>
            </div>
          ))
        ) : (
          <Empty text={t("cron.none")} />
        )}
      </div>
    </div>
  );
}

function mergeRobot(config) {
  return {
    ...DEFAULT_ROBOT,
    ...config,
    laptop: { ...DEFAULT_ROBOT.laptop, ...(config.laptop || {}) },
    fake: { ...(config.fake || {}) },
    unitree_g1: { ...DEFAULT_ROBOT.unitree_g1, ...(config.unitree_g1 || {}) },
    lekiwi: { ...DEFAULT_ROBOT.lekiwi, ...(config.lekiwi || {}) }
  };
}

function setPath(object, path, value) {
  const clone = structuredClone(object);
  let cursor = clone;
  for (const key of path.slice(0, -1)) {
    cursor[key] = cursor[key] || {};
    cursor = cursor[key];
  }
  cursor[path[path.length - 1]] = value;
  return clone;
}

function InfoGrid({ rows }) {
  return (
    <div className="mt-3 grid gap-1.5">
      {rows.map(([key, value]) => (
        <div key={key} className="grid grid-cols-[92px_minmax(0,1fr)] items-baseline gap-3 text-xs">
          <div className="font-mono text-[10px] uppercase tracking-wide text-slate-400">{key}</div>
          <div className="truncate font-mono text-[12px] text-slate-700">{value}</div>
        </div>
      ))}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <div className="font-mono text-2xl font-bold leading-none tabular-nums tracking-tight text-slate-900">{value}</div>
      <div className="mono-label mt-2">{label}</div>
    </div>
  );
}

function MemoryRow({ item }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <div className="mono-label text-blue-600/70">{item.kind || "memory"}</div>
      <div className="mt-1.5 text-sm leading-snug text-slate-700">{item.summary || item.text || ""}</div>
    </div>
  );
}

function SectionTitle({ title, meta }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-slate-100 pb-2">
      <div className="mono-label text-slate-500">{title}</div>
      {meta ? <div className="mono-label normal-case text-slate-400">{meta}</div> : null}
    </div>
  );
}

function Labeled({ label, children }) {
  return (
    <label className="grid gap-1.5">
      <span className="mono-label text-slate-500">{label}</span>
      {children}
    </label>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="flex items-center justify-between gap-3 text-sm text-slate-700">
      <span>{label}</span>
      <input className="h-4 w-4 accent-blue-600" type="checkbox" checked={Boolean(checked)} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function Empty({ text }) {
  return (
    <div className="flex items-center gap-2.5 rounded-xl border border-dashed border-slate-300 bg-slate-50/60 px-4 py-5 text-sm text-slate-400">
      <span className="h-1.5 w-1.5 shrink-0 rounded-full border border-slate-300" />
      {text}
    </div>
  );
}
