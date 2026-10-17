import { useMemo, useState } from "react";
import { api, fmt, type Live } from "../api";
import { EcgStrip } from "../components/EcgStrip";
import { Plot, palette } from "../components/Plot";
import { Empty, Panel, QualityPill, Stat, Warnings } from "../components/ui";

export function LivePage({ live, error, noDevice }: { live: Live | null; error: Error | null; noDevice: boolean }) {
  const [view, setView] = useState<"filtered" | "raw">("filtered");
  const [saved, setSaved] = useState<string | null>(null);

  const wf = live?.waveform;
  // On a phone, 10 s of calibrated paper would be a few pixels tall; show the last 4 s instead.
  const narrow = typeof window !== "undefined" && window.matchMedia("(max-width: 760px)").matches;
  const strip = useMemo(() => {
    if (!wf || wf.time_s.length === 0) return null;
    const total = wf.filtered_mv.length;
    const keep = narrow && live ? Math.min(total, Math.round(4 * live.fs)) : total;
    const first = total - keep;
    const time = wf.time_s.slice(first);
    const mv = view === "filtered" ? wf.filtered_mv.slice(first) : wf.raw_mv.slice(-total).slice(first);
    return {
      time,
      mv,
      peaks: wf.r_peak_offsets.filter((o) => o >= first && o < total).map((o) => ({ t: wf.time_s[o] })),
    };
  }, [wf, view, narrow, live]);

  const rrPlot = useMemo(() => {
    const rr = live?.rr_recent_s ?? [];
    return [{ x: rr.map((_, i) => i - rr.length + 1), y: rr.map((v) => v * 1000), type: "scatter", mode: "lines+markers",
      line: { color: palette().ink }, marker: { size: 6 }, hovertemplate: "%{y:.0f} ms<extra></extra>" }];
  }, [live?.rr_recent_s]);

  if (!live) {
    return (
      <Panel title="Live monitoring">
        <Empty>
          {noDevice || (error && "status" in error && (error as { status: number }).status === 404)
            ? "No device is streaming yet. Start the simulator with python scripts/run_simulator.py, or power on the ESP32."
            : error
              ? `Cannot reach the server: ${error.message}. Start it with python scripts/serve.py.`
              : "Connecting to the server."}
        </Empty>
      </Panel>
    );
  }

  const hr = live.heart_rate;
  const ps = live.packet_stats;
  const saveRecording = async () => {
    const res = await api<{ id: string; duration_s: number }>("/recordings", { method: "POST", body: JSON.stringify({}) });
    setSaved(`Saved ${res.duration_s.toFixed(0)} s as ${res.id}`);
  };

  return (
    <>
      <Panel
        title="ECG"
        aside={
          <div className="segmented" role="group" aria-label="Signal shown">
            {(["filtered", "raw"] as const).map((v) => (
              <button key={v} aria-pressed={view === v} onClick={() => setView(v)}>
                {v === "filtered" ? "Filtered" : "Raw"}
              </button>
            ))}
          </div>
        }
      >
        {strip ? (
          <EcgStrip time={strip.time} mv={strip.mv} peaks={view === "filtered" ? strip.peaks : []}
            label={`${view} ECG, last ${strip.time.length / live.fs} seconds`} />
        ) : (
          <Empty>Waiting for the first samples.</Empty>
        )}
        <p className="fine">
          Filtered view uses causal filtering, so it lags the raw signal slightly. R-peak markers appear about one second after a
          beat, once it is confirmed.
        </p>
      </Panel>

      <div className="vitals">
        <Stat label="Heart rate now" value={fmt(hr.current_bpm)} unit="bpm" note="from the latest valid RR interval" />
        <Stat label="Average" value={fmt(hr.mean_bpm)} unit="bpm" note={`${hr.valid_rr_count} valid intervals`} />
        <Stat label="Range" value={hr.min_bpm === null ? "—" : `${fmt(hr.min_bpm)}–${fmt(hr.max_bpm)}`} unit="bpm" />
        <Stat label="Since last beat" value={fmt(hr.seconds_since_last_beat, 1)} unit="s" />
        <div className="stat">
          <span className="stat-label">Signal quality</span>
          <span className="stat-value"><QualityPill status={live.quality.status} /></span>
          <span className="stat-note">{live.quality.reasons[0]}</span>
        </div>
      </div>
      <Warnings items={[...hr.warnings, ...(live.quality.status === "poor" ? live.quality.reasons : [])]} title="Measurement notes" />

      <div className="grid-2">
        <Panel title="Recent RR intervals">
          {live.rr_recent_s.length ? (
            <Plot data={rrPlot} height={200} label="Recent RR intervals in milliseconds"
              layout={{ xaxis: { title: { text: "beats ago" }, dtick: 1 }, yaxis: { title: { text: "ms" } } }} />
          ) : (
            <Empty>No RR intervals yet.</Empty>
          )}
          {hr.flagged_rr_count > 0 && <p className="fine">{hr.flagged_rr_count} implausible interval(s) flagged and excluded.</p>}
        </Panel>

        <Panel title="Connection">
          <dl className="facts">
            <dt>Sampling rate</dt><dd>{live.fs} Hz</dd>
            <dt>Last packet</dt><dd>{live.seconds_since_last_packet === null ? "—" : `${live.seconds_since_last_packet.toFixed(1)} s ago`}</dd>
            <dt>Packets accepted</dt><dd>{ps.accepted.toLocaleString()} of {ps.received.toLocaleString()}</dd>
            <dt>Lost packets</dt><dd>{ps.lost_packets} (about {ps.lost_samples_estimated} samples, estimated)</dd>
            <dt>Duplicates</dt><dd>{ps.duplicates}</dd>
            <dt>Malformed</dt><dd>{ps.malformed}</dd>
            <dt>Late, discarded</dt><dd>{ps.out_of_order}</dd>
            <dt>Device restarts</dt><dd>{ps.session_restarts}</dd>
            <dt>Electrodes</dt>
            <dd>{live.lead_off.lo_plus || live.lead_off.lo_minus ? "Lead off reported" : "Attached (as reported)"}</dd>
          </dl>
          {(live.gaps > 0 || live.analysis_restarts > 0) && (
            <p className="fine">
              {live.gaps > 0 && `${live.gaps} gap(s) in the data. Missing samples are never filled in; heart-rate intervals across a gap are excluded, and the waveform shows only data since the last gap. `}
              {live.analysis_restarts > 0 && `The device restarted ${live.analysis_restarts} time(s); analysis began again each time.`}
            </p>
          )}
          <div className="actions">
            <button className="primary" onClick={saveRecording}>Save recording</button>
            <span className="fine">Saves the buffered raw samples, up to the last 10 minutes.</span>
            {saved && <span className="fine" role="status">{saved}</span>}
          </div>
        </Panel>
      </div>
    </>
  );
}
