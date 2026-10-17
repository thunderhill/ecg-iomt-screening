import { useEffect, useState } from "react";
import { api, usePoll } from "../api";
import { Empty, Panel } from "../components/ui";

interface Config {
  sample_rate_hz: number;
  rolling_window_s: number;
  retention_days: number;
  selected_model: string;
  api_host: string;
  api_port: number;
  preprocess: { bandpass_low_hz: number; bandpass_high_hz: number; filter_order: number; powerline_hz: number | null };
  screening: { brady_bpm: number; tachy_bpm: number; min_duration_s: number; min_valid_rr: number; analysis_window_s: number };
  quality: { flatline_std_mv: number; clip_fraction_max: number; noise_ratio_max: number; min_window_s: number };
}

type Field = { key: string; label: string; unit?: string; step?: number; help?: string };
const SECTIONS: { title: string; group: "top" | "preprocess" | "screening" | "quality"; fields: Field[]; note?: string }[] = [
  { title: "Acquisition", group: "top", note: "Applies at the next device connection. The device must use the same rate.",
    fields: [{ key: "sample_rate_hz", label: "Sampling rate", unit: "Hz" }, { key: "rolling_window_s", label: "Live window", unit: "s" }] },
  { title: "Filtering", group: "preprocess", note: "Applies at the next device connection. The high cut-off must stay below half the sampling rate.",
    fields: [{ key: "bandpass_low_hz", label: "Low cut-off", unit: "Hz", step: 0.1 }, { key: "bandpass_high_hz", label: "High cut-off", unit: "Hz" },
      { key: "filter_order", label: "Filter order", step: 1 }] },
  { title: "Heart-rate screening thresholds", group: "screening",
    note: "Research parameters, not clinical thresholds. Changes apply immediately.",
    fields: [{ key: "brady_bpm", label: "Low heart rate below", unit: "bpm" }, { key: "tachy_bpm", label: "Elevated heart rate above", unit: "bpm" },
      { key: "min_duration_s", label: "Must persist for", unit: "s" }, { key: "min_valid_rr", label: "Minimum valid intervals", step: 1 },
      { key: "analysis_window_s", label: "Analysis window", unit: "s" }] },
  { title: "Signal-quality thresholds", group: "quality", note: "Heuristic limits. Changes apply immediately.",
    fields: [{ key: "flatline_std_mv", label: "Flat-line below", unit: "mV", step: 0.001 }, { key: "clip_fraction_max", label: "Clipped samples above", unit: "fraction", step: 0.01 },
      { key: "noise_ratio_max", label: "Noise ratio above", step: 0.05 }, { key: "min_window_s", label: "Minimum data before judging", unit: "s" }] },
];

export function SettingsPage() {
  const { data, refresh } = usePoll<{ config: Config }>("/configuration", 0);
  const models = usePoll<{ available: string[]; selected: string }>("/models", 0);
  const devices = usePoll<{ items: { device_id: string; kind: string; source: string; last_seen: number | null; revoked: boolean }[] }>("/devices", 10000);
  const me = usePoll<{ tls: boolean }>("/auth/me", 0);
  const [draft, setDraft] = useState<Record<string, Record<string, string>>>({});
  const [model, setModel] = useState("rules");
  const [powerline, setPowerline] = useState("off");
  const [retention, setRetention] = useState("30");
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    if (!data) return;
    const c = data.config;
    const pick = (o: object, keys: string[]) => Object.fromEntries(keys.map((k) => [k, String((o as Record<string, unknown>)[k])]));
    setDraft({
      top: pick(c, SECTIONS[0].fields.map((f) => f.key)),
      preprocess: pick(c.preprocess, SECTIONS[1].fields.map((f) => f.key)),
      screening: pick(c.screening, SECTIONS[2].fields.map((f) => f.key)),
      quality: pick(c.quality, SECTIONS[3].fields.map((f) => f.key)),
    });
    setModel(c.selected_model);
    setPowerline(c.preprocess.powerline_hz ? String(c.preprocess.powerline_hz) : "off");
    setRetention(String(c.retention_days));
  }, [data]);

  if (!data) return <Panel title="Settings"><Empty>Loading settings.</Empty></Panel>;
  const c = data.config;

  const validate = (): string | null => {
    for (const s of SECTIONS) for (const f of s.fields) {
      const v = Number(draft[s.group]?.[f.key]);
      if (!Number.isFinite(v) || v <= 0) return `${f.label} must be a positive number.`;
    }
    const fs = Number(draft.top.sample_rate_hz);
    if (Number(draft.preprocess.bandpass_high_hz) >= fs / 2) return `High cut-off must be below ${fs / 2} Hz (half the sampling rate).`;
    if (Number(draft.preprocess.bandpass_low_hz) >= Number(draft.preprocess.bandpass_high_hz)) return "Low cut-off must be below the high cut-off.";
    if (Number(draft.screening.brady_bpm) >= Number(draft.screening.tachy_bpm)) return "The low heart-rate threshold must be below the elevated one.";
    return null;
  };

  const save = async () => {
    const problem = validate();
    if (problem) return setMessage({ ok: false, text: problem });
    const num = (g: string) => Object.fromEntries(Object.entries(draft[g]).map(([k, v]) => [k, Number(v)]));
    const body = {
      ...num("top"),
      retention_days: Number(retention),
      selected_model: model,
      preprocess: { ...num("preprocess"), powerline_hz: powerline === "off" ? 0 : Number(powerline) },
      screening: num("screening"),
      quality: num("quality"),
    };
    try {
      const res = await api<{ applies: string }>("/configuration", { method: "PUT", body: JSON.stringify(body) });
      setMessage({ ok: true, text: `Settings saved. They apply ${res.applies === "now" ? "now" : `at the ${res.applies}`}.` });
      refresh();
    } catch (e) {
      setMessage({ ok: false, text: `Not saved: ${(e as Error).message}` });
    }
  };

  return (
    <form className="settings" onSubmit={(e) => { e.preventDefault(); save(); }}>
      {SECTIONS.map((s) => (
        <Panel title={s.title} key={s.title}>
          {s.note && <p className="fine">{s.note}</p>}
          <div className="fields">
            {s.fields.map((f) => (
              <label key={f.key}>
                <span>{f.label}{f.unit ? ` (${f.unit})` : ""}</span>
                <input type="number" inputMode="decimal" step={f.step ?? "any"} value={draft[s.group]?.[f.key] ?? ""}
                  onChange={(e) => setDraft({ ...draft, [s.group]: { ...draft[s.group], [f.key]: e.target.value } })} />
              </label>
            ))}
            {s.group === "preprocess" && (
              <label>
                <span>Mains interference filter</span>
                <select value={powerline} onChange={(e) => setPowerline(e.target.value)}>
                  <option value="off">Off</option><option value="50">50 Hz</option><option value="60">60 Hz</option>
                </select>
              </label>
            )}
          </div>
        </Panel>
      ))}

      <Panel title="Screening model and data">
        <div className="fields">
          <label>
            <span>Beat model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {(models.data?.available ?? ["rules"]).map((m) => (
                <option key={m} value={m}>{{ rules: "Rules only", rf: "Random Forest", xgb: "XGBoost", cnn: "1D-CNN", hybrid: "CNN + XGBoost" }[m] ?? m}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Keep recordings for (days)</span>
            <input type="number" min={1} step={1} value={retention} onChange={(e) => setRetention(e.target.value)} />
          </label>
        </div>
        <p className="fine">Only trained models are listed. Heart-rate rules always run alongside the selected model.</p>
      </Panel>

      <Panel title="Device connection">
        <dl className="facts">
          <dt>Device stream address</dt>
          <dd>{me.data?.tls ? "wss" : "ws"}://{c.api_host === "0.0.0.0" ? "<this computer's address>" : c.api_host}:{c.api_port}/ws/ingest</dd>
          <dt>Encryption (TLS)</dt>
          <dd>{me.data?.tls ? "On" : "Off: this computer only (127.0.0.1)"}</dd>
          <dt>Device sign-in</dt><dd>Per-device token required</dd>
        </dl>
        <h4>Registered devices</h4>
        {devices.data && devices.data.items.length > 0 ? (
          <div className="table-scroll">
            <table className="compare">
              <thead><tr><th scope="col">Device</th><th scope="col">Kind</th><th scope="col">Shown as</th><th scope="col">Last seen</th><th scope="col">Status</th></tr></thead>
              <tbody>
                {devices.data.items.map((d) => (
                  <tr key={d.device_id}>
                    <th scope="row">{d.device_id}</th><td>{d.kind}</td>
                    <td><span className={`src src-${d.source.toLowerCase()}`}>{d.source}</span></td>
                    <td>{d.last_seen ? new Date(d.last_seen * 1000).toLocaleString() : "never"}</td>
                    <td>{d.revoked ? "revoked" : "active"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="fine">No devices registered yet.</p>
        )}
        <p className="fine">
          Register or revoke devices on the gateway with scripts/security_setup.py. To accept an ESP32 over Wi-Fi, start the
          gateway with --host 0.0.0.0 --tls. See docs/security.md and docs/wiring.md.
        </p>
      </Panel>

      <div className="actions sticky">
        <button className="primary" type="submit">Save settings</button>
        {message && <span className={message.ok ? "ok" : "error"} role="status">{message.text}</span>}
      </div>
    </form>
  );
}
