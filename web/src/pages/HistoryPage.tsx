import { useEffect, useMemo, useState } from "react";
import { fmt, usePoll, type Source } from "../api";
import { EcgStrip } from "../components/EcgStrip";
import { Plot, palette } from "../components/Plot";
import { Empty, Panel, Warnings } from "../components/ui";

interface RecordingItem {
  id: string;
  created_at: number;
  source: Source;
  device_id: string | null;
  fs: number;
  duration_s: number;
  units: string;
  notes: string | null;
}

interface Analysis {
  source: Source;
  duration_s: number;
  filtering: string;
  r_peaks_s: number[];
  rr: { times_s: number[]; rr_s: number[]; flags: string[][]; flagged_fraction: number };
  heart_rate: { mean_bpm: number | null; min_bpm: number | null; max_bpm: number | null; valid_rr_count: number; flagged_rr_count: number; warnings: string[] };
  hrv: { available: boolean; mean_nn_ms: number | null; sdnn_ms: number | null; rmssd_ms: number | null; pnn50_pct: number | null; nn_count: number; warnings: string[] };
  quality: { overall: string; segments: { start_s: number; status: string }[] };
  rules: { title: string; evidence: string[] };
  model?: { kind: string; beat_times_s: number[]; predicted: string[]; class_counts: Record<string, number> };
}

interface Waveform {
  time_s: number[];
  mv: number[];
  start_s: number;
}

const WINDOW_S = 10;

export function HistoryPage({ selected, onSelect }: { selected: string | null; onSelect: (id: string) => void }) {
  const list = usePoll<{ items: RecordingItem[] }>("/recordings", 10000);
  const [start, setStart] = useState(0);
  useEffect(() => setStart(0), [selected]);
  const analysis = usePoll<Analysis>(selected ? `/recordings/${selected}/analysis` : null, 0);
  const wave = usePoll<Waveform>(selected ? `/recordings/${selected}/waveform?start_s=${start}&duration_s=${WINDOW_S}` : null, 0);

  const items = list.data?.items ?? [];
  const current = items.find((r) => r.id === selected) ?? null;
  const a = analysis.data;

  const peaks = useMemo(() => {
    if (!a || !wave.data) return [];
    const t0 = wave.data.start_s;
    const labels = new Map<number, string>();
    a.model?.beat_times_s.forEach((t, i) => labels.set(Math.round(t * 50), a.model!.predicted[i]));
    return a.r_peaks_s
      .filter((t) => t >= t0 && t < t0 + WINDOW_S)
      .map((t) => {
        // Model beats come from causal processing; match within 20 ms of the offline peak.
        const lab = labels.get(Math.round(t * 50)) ?? labels.get(Math.round(t * 50) - 1) ?? labels.get(Math.round(t * 50) + 1);
        return { t, label: lab && lab !== "N" ? lab : undefined };
      });
  }, [a, wave.data]);

  const hrTrend = useMemo(() => {
    if (!a) return [];
    const ok = a.rr.flags.map((f) => f.length === 0);
    return [
      { x: a.rr.times_s.filter((_, i) => ok[i]), y: a.rr.rr_s.filter((_, i) => ok[i]).map((v) => 60 / v), type: "scattergl",
        mode: "markers", marker: { size: 3, color: palette().ink }, hovertemplate: "%{x:.1f} s: %{y:.0f} bpm<extra></extra>" },
      { x: a.rr.times_s.filter((_, i) => !ok[i]), y: a.rr.rr_s.filter((_, i) => !ok[i]).map((v) => 60 / v), type: "scattergl",
        mode: "markers", marker: { size: 5, color: "#B7791F", symbol: "x" }, hovertemplate: "flagged %{x:.1f} s<extra></extra>" },
    ];
  }, [a]);

  return (
    <>
      <Panel title="Saved recordings">
        {items.length === 0 ? (
          <Empty>
            No recordings yet. Save one from the Live page, or import a MIT-BIH record with
            python scripts/import_recording.py --mitdb 208 --minutes 5
          </Empty>
        ) : (
          <table className="recordings">
            <thead>
              <tr><th scope="col">Recording</th><th scope="col">Source</th><th scope="col">Length</th><th scope="col">Notes</th></tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.id} aria-selected={r.id === selected}>
                  <td>
                    <button className="link" onClick={() => onSelect(r.id)}>
                      {new Date(r.created_at * 1000).toLocaleString()}
                    </button>
                  </td>
                  <td><span className={`src src-${r.source.toLowerCase()}`}>{r.source}</span></td>
                  <td>{(r.duration_s / 60).toFixed(1)} min</td>
                  <td>{r.notes ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>

      {current && (
        <>
          <Panel
            title={`Waveform, ${fmt(start, 0)}–${fmt(Math.min(start + WINDOW_S, current.duration_s), 0)} s`}
            aside={
              <span className="actions">
                <a href={`/recordings/${current.id}/export.csv`} download>Export samples (CSV)</a>
                <a href={`/recordings/${current.id}/metrics.csv`} download>Export RR and heart rate (CSV)</a>
              </span>
            }
          >
            <label className="scrub">
              <span>Position in recording</span>
              <input type="range" min={0} max={Math.max(0, current.duration_s - WINDOW_S)} step={1} value={start}
                onChange={(e) => setStart(Number(e.target.value))} />
            </label>
            {wave.data && wave.data.mv.length > 0 ? (
              <EcgStrip time={wave.data.time_s} mv={wave.data.mv} peaks={peaks} label={`Recorded ECG from ${start} seconds`} />
            ) : (
              <Empty>Loading waveform.</Empty>
            )}
            <p className="fine">
              {current.source === "SIMULATED" && "This recording came from the simulated device. "}
              Waveform shown unfiltered in millivolts. R-peak markers come from offline analysis; letters mark beats the selected
              model classified as S, V or F.
            </p>
          </Panel>

          {analysis.error && <Warnings items={[analysis.error.message]} title="Analysis failed" />}
          {a && (
            <div className="grid-2">
              <Panel title="Heart-rate trend">
                <Plot data={hrTrend} height={220} label="Instantaneous heart rate across the recording"
                  layout={{ xaxis: { title: { text: "s" } }, yaxis: { title: { text: "bpm" } } }} />
                <p className="fine">
                  Mean {fmt(a.heart_rate.mean_bpm)} bpm, range {fmt(a.heart_rate.min_bpm)}–{fmt(a.heart_rate.max_bpm)} bpm over{" "}
                  {a.heart_rate.valid_rr_count} valid intervals. {a.heart_rate.flagged_rr_count} flagged intervals are marked with a cross
                  and excluded.
                </p>
              </Panel>
              <Panel title="RR intervals and HRV">
                <Plot height={170} label="RR interval tachogram"
                  data={[{ x: a.rr.times_s, y: a.rr.rr_s.map((v) => v * 1000), type: "scattergl", mode: "lines",
                    line: { color: palette().slate, width: 1 }, hovertemplate: "%{y:.0f} ms<extra></extra>" }]}
                  layout={{ xaxis: { title: { text: "s" } }, yaxis: { title: { text: "RR (ms)" } } }} />
                {a.hrv.available ? (
                  <dl className="facts">
                    <dt>Mean NN</dt><dd>{fmt(a.hrv.mean_nn_ms)} ms</dd>
                    <dt>SDNN</dt><dd>{fmt(a.hrv.sdnn_ms, 1)} ms</dd>
                    <dt>RMSSD</dt><dd>{fmt(a.hrv.rmssd_ms, 1)} ms</dd>
                    <dt>pNN50</dt><dd>{fmt(a.hrv.pnn50_pct, 1)} %</dd>
                    <dt>NN intervals</dt><dd>{a.hrv.nn_count}</dd>
                  </dl>
                ) : null}
                <Warnings items={a.hrv.warnings} title="HRV notes" />
              </Panel>
            </div>
          )}
          {a && (
            <Panel title="Recording summary">
              <dl className="facts">
                <dt>Overall signal quality</dt><dd>{a.quality.overall}</dd>
                <dt>Heart-rate rule (final window)</dt><dd>{a.rules.title}</dd>
                {a.model && <><dt>Beat model ({a.model.kind})</dt>
                  <dd>{Object.entries(a.model.class_counts).map(([c, n]) => `${c} ${n}`).join(", ")}</dd></>}
                <dt>Processing</dt><dd>{a.filtering}</dd>
              </dl>
            </Panel>
          )}
        </>
      )}
    </>
  );
}
