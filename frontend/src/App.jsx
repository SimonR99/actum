import {
  Bot,
  BrainCircuit,
  Camera,
  CheckCircle2,
  Clock3,
  Cpu,
  Eye,
  Map,
  Mic2,
  Network,
  Radio,
  Save,
  Send,
  Settings,
  SlidersHorizontal,
  Terminal,
  Wrench
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

const SETTINGS_TABS = [
  { id: "robot", label: "Robot", icon: Bot },
  { id: "model", label: "Model", icon: Cpu },
  { id: "tools", label: "Tools", icon: Wrench },
  { id: "cron", label: "Cron", icon: Clock3 }
];

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

function useActumState() {
  const [state, setState] = useState(null);
  const [frame, setFrame] = useState("");
  const [wsLive, setWsLive] = useState(false);
  const [cameraLive, setCameraLive] = useState(false);
  const [lastFrameAt, setLastFrameAt] = useState("");
  const [toast, setToast] = useState("");
  const latestFrameId = useRef(0);
  const latestFrameSentAt = useRef(0);
  const frameRaf = useRef(0);

  const showToast = (message) => {
    setToast(message);
    window.clearTimeout(showToast._timer);
    showToast._timer = window.setTimeout(() => setToast(""), 2200);
  };

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
        if (msg.type === "turn" && msg.ignored) {
          showToast(msg.reason || "Passive event ignored");
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

  return { state, frame, wsLive, cameraLive, lastFrameAt, toast, showToast, refresh };
}

export default function App() {
  const { state, frame, wsLive, cameraLive, lastFrameAt, toast, showToast, refresh } = useActumState();
  const [command, setCommand] = useState("");
  const [settingsTab, setSettingsTab] = useState("robot");

  const robotName = state?.personality?.name || state?.robot_name || "dino";
  const backend = state?.backend || "unknown";
  const connected = Boolean(state?.robot_state?.connected);
  const modelSettings = state?.settings?.models || DEFAULT_MODELS;
  const activeProvider = modelSettings.active_provider || "local";

  async function sendTrigger(source) {
    const text = command.trim();
    if (!text && source !== "vision") return;
    try {
      await api(`/trigger/${source}`, {
        method: "POST",
        body: JSON.stringify({
          text,
          force: source !== "vision",
          importance: source === "vision" ? 0.6 : 1
        })
      });
      showToast(`${source} trigger queued`);
      if (source !== "vision") setCommand("");
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  return (
    <div className="min-h-screen bg-slate-100 text-slate-950">
      <Header
        robotName={robotName}
        command={command}
        setCommand={setCommand}
        onTrigger={sendTrigger}
        wsLive={wsLive}
        cameraLive={cameraLive}
        backend={backend}
        connected={connected}
        activeProvider={activeProvider}
      />

      <main className="grid h-[calc(100vh-72px)] min-h-[760px] grid-cols-[320px_minmax(520px,1fr)_420px] grid-rows-[minmax(420px,1fr)_300px] gap-4 p-4 max-[1280px]:h-auto max-[1280px]:grid-cols-2 max-[1280px]:grid-rows-none max-[860px]:grid-cols-1">
        <section className="panel row-span-2 flex flex-col">
          <PanelHeader title="Intent" meta={state?.intent?.status || "idle"} icon={BrainCircuit} />
          <IntentPanel state={state} />
        </section>

        <section className="panel flex flex-col">
          <PanelHeader title="Perception" meta={lastFrameAt || "waiting"} icon={Eye} />
          <PerceptionPanel state={state} frame={frame} />
        </section>

        <section className="panel row-span-2 flex flex-col">
          <PanelHeader title="Settings" meta="operator" icon={Settings} />
          <SettingsPanel
            state={state}
            activeTab={settingsTab}
            setActiveTab={setSettingsTab}
            showToast={showToast}
            refresh={refresh}
          />
        </section>

        <section className="grid min-h-0 grid-cols-2 gap-4 max-[860px]:grid-cols-1">
          <div className="panel flex min-h-0 flex-col">
            <PanelHeader title="Tool Graph" meta={`${state?.tool_graph?.length || 0} calls`} icon={Network} />
            <ToolGraph nodes={state?.tool_graph || []} />
          </div>
          <div className="panel flex min-h-0 flex-col">
            <PanelHeader title="Map And Timeline" meta="memory" icon={Map} />
            <MapTimeline state={state} />
          </div>
        </section>
      </main>

      <div
        className={cx(
          "fixed bottom-4 right-4 z-50 max-w-md rounded-lg bg-slate-950 px-4 py-3 text-sm font-medium text-white shadow-xl transition",
          toast ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-2 opacity-0"
        )}
      >
        {toast}
      </div>
    </div>
  );
}

function Header({
  robotName,
  command,
  setCommand,
  onTrigger,
  wsLive,
  cameraLive,
  backend,
  connected,
  activeProvider
}) {
  return (
    <header className="grid min-h-[72px] grid-cols-[300px_minmax(380px,1fr)_auto] items-center gap-4 border-b border-slate-200 bg-white px-4 max-[1180px]:grid-cols-1 max-[1180px]:py-3">
      <div className="flex items-center gap-3">
        <span className={cx("h-3 w-3 rounded-full", wsLive ? "bg-emerald-500 shadow-[0_0_0_5px_rgba(16,185,129,.14)]" : "bg-slate-300")} />
        <div className="min-w-0">
          <div className="truncate text-lg font-bold leading-6">{robotName}</div>
          <div className="truncate text-xs text-slate-500">Always-on autonomy console</div>
        </div>
      </div>

      <div className="grid grid-cols-[minmax(160px,1fr)_auto_auto_auto] gap-2 max-[760px]:grid-cols-1">
        <div className="relative">
          <Terminal className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            className="field pl-9"
            value={command}
            placeholder="Give dino a command"
            onChange={(event) => setCommand(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") onTrigger("chat");
            }}
          />
        </div>
        <TriggerButton icon={Send} label="Chat" onClick={() => onTrigger("chat")} primary />
        <TriggerButton icon={Mic2} label="Language" onClick={() => onTrigger("language")} />
        <TriggerButton icon={Camera} label="Vision" onClick={() => onTrigger("vision")} />
      </div>

      <div className="flex flex-wrap justify-end gap-2 max-[1180px]:justify-start">
        <StatusChip ok={wsLive} label={wsLive ? "ws live" : "ws offline"} />
        <StatusChip ok={cameraLive} label={cameraLive ? "camera live" : "camera waiting"} />
        <StatusChip ok={connected} label={`backend ${backend}`} />
        <span className="chip">model {activeProvider}</span>
      </div>
    </header>
  );
}

function TriggerButton({ icon: Icon, label, onClick, primary = false }) {
  return (
    <button className={cx("btn", primary && "btn-primary")} onClick={onClick}>
      <Icon className="h-4 w-4" />
      {label}
    </button>
  );
}

function StatusChip({ ok, label }) {
  return (
    <span className={cx("chip", ok ? "chip-ok" : "chip-warn")}>
      {ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <Radio className="h-3.5 w-3.5" />}
      {label}
    </span>
  );
}

function PanelHeader({ title, meta, icon: Icon }) {
  return (
    <div className="panel-header">
      <div className="flex min-w-0 items-center gap-2">
        <Icon className="h-4 w-4 text-slate-500" />
        <div className="panel-title">{title}</div>
      </div>
      <div className="panel-meta">{meta}</div>
    </div>
  );
}

function IntentPanel({ state }) {
  const intent = state?.intent || {};
  const behavior = state?.behavior || {};
  const goal = intent.goal || behavior.goal || "No active task";
  const steps = intent.steps || [];
  const nodes = behavior.nodes || [];

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-slate-200 p-4">
        <div className="text-xs font-bold uppercase text-slate-500">Goal</div>
        <div className="mt-2 text-xl font-bold leading-7">{goal}</div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        <SectionTitle title="Plan" meta={`${steps.length} steps`} />
        <div className="mt-3 grid gap-2">
          {steps.length ? (
            steps.map((step, index) => <StepRow key={step.id || index} step={step} index={index} />)
          ) : (
            <Empty text="Waiting for a plan" />
          )}
        </div>

        <div className="mt-6">
          <SectionTitle title="Behavior Tree" meta={`${behavior.tick_count || 0} ticks`} />
          <div className="mt-3 grid gap-2">
            {nodes.length ? nodes.map((node) => <BehaviorNode key={node.id || node.label} node={node} />) : <Empty text="No behavior nodes" />}
          </div>
        </div>
      </div>
    </div>
  );
}

function StepRow({ step, index }) {
  const status = step.status || "pending";
  return (
    <div className={cx("rounded-lg border p-3", statusTone(status))}>
      <div className="flex items-start gap-3">
        <div className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-white text-xs font-bold text-slate-600">
          {status === "done" ? <CheckCircle2 className="h-4 w-4 text-emerald-600" /> : index + 1}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold text-slate-900">{step.label || step.id}</div>
          <div className="mt-0.5 text-xs capitalize text-slate-500">{status}</div>
        </div>
      </div>
    </div>
  );
}

function BehaviorNode({ node }) {
  return (
    <div className={cx("rounded-lg border p-3", statusTone(node.status || "waiting"))}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-slate-900">{node.label || node.id}</div>
          <div className="mt-1 text-xs text-slate-500">
            {node.kind || "node"} · {node.status || "waiting"}
          </div>
        </div>
        <span className="rounded-full bg-white px-2 py-1 text-xs font-medium text-slate-600">{node.id}</span>
      </div>
      {node.detail ? <div className="mt-2 text-xs text-slate-600">{node.detail}</div> : null}
    </div>
  );
}

function statusTone(status) {
  if (status === "active") return "border-blue-200 bg-blue-50";
  if (status === "done") return "border-emerald-200 bg-emerald-50";
  if (status === "blocked" || status === "failed") return "border-red-200 bg-red-50";
  return "border-slate-200 bg-slate-50";
}

function PerceptionPanel({ state, frame }) {
  const body = state?.body || {};
  const memory = state?.memory || {};
  const recent = (memory.recent || []).filter((item) => ["observation", "spatial"].includes(item.kind)).slice(-4).reverse();

  return (
    <div className="grid min-h-0 flex-1 grid-cols-[minmax(360px,1fr)_320px] gap-4 p-4 max-[1180px]:grid-cols-1">
      <div className="min-h-0">
        <div className="relative aspect-[4/3] overflow-hidden rounded-lg border border-slate-200 bg-slate-950">
          {frame ? (
            <img className="h-full w-full object-contain" src={frame} alt="Robot camera feed" />
          ) : (
            <div className="grid h-full place-items-center text-sm text-slate-400">No camera frame</div>
          )}
        </div>
      </div>

      <div className="grid min-h-0 content-start gap-4 overflow-auto">
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <SectionTitle title="Body" meta={body.posture || "unknown"} />
          <div className="mt-3 text-sm font-semibold">{body.summary || "Body state unavailable"}</div>
          <InfoGrid
            rows={[
              ["Holding", body.holding || "nothing"],
              ["Contacts", (body.contacts || []).join(", ") || "none"],
              ["Pose", Object.keys(body.base_pose || {}).length ? JSON.stringify(body.base_pose) : "none"],
              ["Joints", Object.keys(body.joints || {}).length ? `${Object.keys(body.joints).length} values` : "none"]
            ]}
          />
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <SectionTitle title="Recent Perception" meta={`${recent.length} items`} />
          <div className="mt-3 grid gap-2">
            {recent.length ? recent.map((item) => <MemoryRow key={item.id || item.summary} item={item} />) : <Empty text="No observations yet" />}
          </div>
        </div>
      </div>
    </div>
  );
}

function ToolGraph({ nodes }) {
  const visible = nodes.slice(-20);
  return (
    <div className="min-h-0 flex-1 overflow-auto p-4">
      <div className="grid gap-3">
        {visible.length ? (
          visible.map((node, index) => {
            const ok = node.result ? node.result.ok : null;
            return (
              <div key={node.id || index} className="grid grid-cols-[28px_minmax(0,1fr)] gap-3">
                <div className={cx("grid h-7 w-7 place-items-center rounded-full text-xs font-bold", ok === false ? "bg-red-600 text-white" : ok === true ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-600")}>
                  {index + 1}
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold">{node.type || "tool"}</div>
                      <div className="mt-1 text-xs text-slate-500">{node.result ? node.result.message : "queued"}</div>
                    </div>
                    <span className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-500">{node.id}</span>
                  </div>
                  <div className="mt-2 text-xs text-slate-600">{summarizeAction(node.action || {})}</div>
                </div>
              </div>
            );
          })
        ) : (
          <Empty text="No tool calls yet" />
        )}
      </div>
    </div>
  );
}

function summarizeAction(action) {
  const ignored = new Set(["type", "time", "_node_id", "ok", "backend"]);
  return (
    Object.entries(action)
      .filter(([key, value]) => !ignored.has(key) && value !== "" && value !== null && value !== undefined)
      .slice(0, 3)
      .map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`)
      .join(" · ") || "no arguments"
  );
}

function MapTimeline({ state }) {
  const memory = state?.memory || {};
  const map = state?.map || {};
  const events = state?.events || [];
  const observations = map.observations || [];
  const counts = memory.counts || {};

  return (
    <div className="min-h-0 flex-1 overflow-auto p-4">
      <div className="grid grid-cols-2 gap-3">
        <Metric label="Facts" value={counts.facts || 0} />
        <Metric label="Places" value={counts.places || 0} />
        <Metric label="Map" value={observations.length} />
        <Metric label="Events" value={events.length} />
      </div>

      <div className="mt-4">
        <SectionTitle title="Map" meta={`${observations.length} observations`} />
        <div className="mt-3 grid gap-2">
          {observations.slice(-4).reverse().map((item) => (
            <MemoryRow key={item.id} item={{ kind: item.place || "map", summary: item.summary }} />
          ))}
          {!observations.length ? <Empty text="No map observations" /> : null}
        </div>
      </div>

      <div className="mt-4">
        <SectionTitle title="Timeline" meta="latest" />
        <div className="mt-3 grid gap-2">
          {events.slice(-8).reverse().map((event, index) => {
            const type = event.type || "event";
            const data = event.data || {};
            return (
              <div key={`${type}-${index}`} className="rounded-lg border border-slate-200 bg-white p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="truncate text-sm font-semibold">{type}</div>
                  <div className="text-xs text-slate-500">{event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : ""}</div>
                </div>
                <div className="mt-1 text-xs text-slate-500">{short(data.message || data.summary || data.goal || data.step || data.tool || event.message || JSON.stringify(data), 140)}</div>
              </div>
            );
          })}
          {!events.length ? <Empty text="No events yet" /> : null}
        </div>
      </div>
    </div>
  );
}

function SettingsPanel({ state, activeTab, setActiveTab, showToast, refresh }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid grid-cols-4 border-b border-slate-200 p-2">
        {SETTINGS_TABS.map((tab) => {
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
      showToast(payload.message || "Robot settings updated");
      setDirty(false);
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  return (
    <div className="grid gap-4">
      <div className="grid gap-3">
        <Labeled label="Name">
          <input className="field" value={name} onChange={(event) => { setDirty(true); setName(event.target.value); }} />
        </Labeled>
        <Labeled label="Backend">
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
          <Toggle label="Webcam" checked={robot.laptop.webcam} onChange={(value) => update(["laptop", "webcam"], value)} />
          <Toggle label="Microphone" checked={robot.laptop.microphone} onChange={(value) => update(["laptop", "microphone"], value)} />
          <Toggle label="Speaker" checked={robot.laptop.speaker} onChange={(value) => update(["laptop", "speaker"], value)} />
        </div>
      ) : null}

      {backend === "unitree_g1" ? (
        <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <Labeled label="Network interface">
            <input className="field" value={robot.unitree_g1.network_interface} onChange={(event) => update(["unitree_g1", "network_interface"], event.target.value)} />
          </Labeled>
          <div className="grid grid-cols-2 gap-3">
            <Labeled label="Speaker ID">
              <input className="field" type="number" min="0" max="1" value={robot.unitree_g1.speaker_id} onChange={(event) => update(["unitree_g1", "speaker_id"], Number(event.target.value || 0))} />
            </Labeled>
            <Labeled label="Volume">
              <input className="field" type="number" min="0" max="100" value={robot.unitree_g1.volume} onChange={(event) => update(["unitree_g1", "volume"], Number(event.target.value || 80))} />
            </Labeled>
          </div>
        </div>
      ) : null}

      {backend === "lekiwi" ? (
        <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <Labeled label="Remote IP">
            <input className="field" value={robot.lekiwi.remote_ip} onChange={(event) => update(["lekiwi", "remote_ip"], event.target.value)} />
          </Labeled>
          <div className="grid grid-cols-2 gap-3">
            <Labeled label="Port">
              <input className="field" type="number" min="1" max="65535" value={robot.lekiwi.port} onChange={(event) => update(["lekiwi", "port"], Number(event.target.value || 5555))} />
            </Labeled>
            <Labeled label="ID">
              <input className="field" value={robot.lekiwi.id} onChange={(event) => update(["lekiwi", "id"], event.target.value)} />
            </Labeled>
          </div>
        </div>
      ) : null}

      <Toggle label="Save to config.json" checked={persist} onChange={setPersist} />
      <button className="btn btn-primary w-full" onClick={apply}>
        <Save className="h-4 w-4" />
        Apply Robot
      </button>
    </div>
  );
}

function ModelSettings({ state, showToast, refresh }) {
  const models = state?.settings?.models || DEFAULT_MODELS;
  const providers = models.providers || DEFAULT_MODELS.providers;
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

  async function apply() {
    try {
      const payload = await api("/settings/model", {
        method: "POST",
        body: JSON.stringify({ provider, model, api_key: apiKey, enabled, persist, persist_secret: persistSecret })
      });
      showToast(payload.message || "Model settings updated");
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  const configured = Boolean(providers[provider]?.api_key_configured);

  return (
    <div className="grid gap-4">
      <Labeled label="Provider">
        <select className="field" value={provider} onChange={(event) => setProvider(event.target.value)}>
          <option value="local">Local</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
        </select>
      </Labeled>
      <Labeled label="Model">
        <input className="field" value={model} onChange={(event) => setModel(event.target.value)} placeholder="Model name" />
      </Labeled>
      <Labeled label={configured ? "API key configured" : "API key"}>
        <input className="field" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder={configured ? "Leave blank to keep existing" : "API key"} />
      </Labeled>
      <div className="grid gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
        <Toggle label="Provider enabled" checked={enabled} onChange={setEnabled} />
        <Toggle label="Save settings" checked={persist} onChange={setPersist} />
        <Toggle label="Save key to config.json" checked={persistSecret} onChange={setPersistSecret} />
      </div>
      <button className="btn btn-primary w-full" onClick={apply}>
        <SlidersHorizontal className="h-4 w-4" />
        Apply Model
      </button>
    </div>
  );
}

function ToolSettings({ state, showToast, refresh }) {
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
      showToast(payload.message || "Tool updated");
      await refresh();
    } catch (error) {
      showToast(error.message);
    }
  }

  return (
    <div className="grid gap-4">
      <Toggle label="Save tool selection" checked={persist} onChange={setPersist} />
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
      showToast(`Scheduled ${name}`);
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
        <Labeled label="Name">
          <input className="field" value={name} onChange={(event) => setName(event.target.value)} placeholder="Room check" />
        </Labeled>
        <Labeled label="Every seconds">
          <input className="field" type="number" min="1" value={seconds} onChange={(event) => setSeconds(Number(event.target.value || 300))} />
        </Labeled>
        <Labeled label="Instruction">
          <input className="field" value={instruction} onChange={(event) => setInstruction(event.target.value)} placeholder="Review the room and update the map" />
        </Labeled>
        <button className="btn btn-primary" onClick={add}>
          <Clock3 className="h-4 w-4" />
          Add Job
        </button>
      </div>

      <div className="grid gap-2">
        {jobs.length ? (
          jobs.map((job) => (
            <div key={job.id} className="rounded-lg border border-slate-200 bg-white p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="truncate text-sm font-semibold">{job.name}</div>
                <span className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-500">{Math.round(job.every_seconds)}s</span>
              </div>
              <div className="mt-1 text-xs text-slate-500">{job.instruction}</div>
            </div>
          ))
        ) : (
          <Empty text="No scheduled jobs" />
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
        <div key={key} className="grid grid-cols-[92px_minmax(0,1fr)] gap-3 text-xs">
          <div className="text-slate-500">{key}</div>
          <div className="truncate text-slate-700">{value}</div>
        </div>
      ))}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="text-lg font-bold">{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  );
}

function MemoryRow({ item }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="text-xs font-bold uppercase text-slate-500">{item.kind || "memory"}</div>
      <div className="mt-1 text-sm text-slate-700">{item.summary || item.text || ""}</div>
    </div>
  );
}

function SectionTitle({ title, meta }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="text-xs font-bold uppercase text-slate-500">{title}</div>
      {meta ? <div className="text-xs text-slate-400">{meta}</div> : null}
    </div>
  );
}

function Labeled({ label, children }) {
  return (
    <label className="grid gap-1.5">
      <span className="text-xs font-bold uppercase text-slate-500">{label}</span>
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
    <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-400">
      {text}
    </div>
  );
}
