import { useMemo } from "react";
import { fmt, usePoll, type Contribution, type Live, type ModelPart, type Quality, type RuleResult } from "../api";
import { EcgStrip } from "../components/EcgStrip";
import { Plot, palette } from "../components/Plot";
import { Empty, Panel, Warnings } from "../components/ui";

interface ExplainResponse {
  rules: { result: RuleResult; thresholds: Record<string, number>; rr_plausibility: Record<string, number>; note: string };
  model: ModelPart;
  global_importance: null | {
    method: string;
    permutation: { feature: string; mean: number; std: number }[];
    note: string;
    baseline_macro_f1: number;
  };
  quality: Quality;
  cnn_saliency?: null | { t: number[]; mv: number[]; weight: number[]; predicted: string; note: string };
}

const FEATURE_TEXT: Record<string, string> = {
  rr_prev_s: "interval before this beat",
  rr_prev2_s: "interval before that",
  rr_local_mean_s: "average of the previous 8 intervals",
  rr_ratio: "prematurity (interval ÷ recent average)",
  rr_local_std_s: "spread of recent intervals",
  rr_local_rmssd_s: "beat-to-beat change of recent intervals",
  hr_local_bpm: "recent heart rate",
  r_amp_mv: "R-wave height",
  q_amp_mv: "Q-wave depth",
  s_amp_mv: "S-wave depth",
  qrs_width_s: "QRS width",
  qrs_energy: "QRS energy",
  post_r_max_mv: "early T-wave height",
  template_corr: "similarity to recent beats' shape",
  local_noise_mv: "local noise level",
  waveform_shape: "waveform shape (whole beat window)",
  cnn_embedding: "CNN waveform embedding",
};

function unitOf(method: string) {
  if (method.includes("log-odds")) return "contribution (log-odds)";
  if (method.includes("logit")) return "attribution (logit)";
  return "change in probability";
}

const NO_VALUE = new Set(["waveform_shape", "cnn_embedding"]);

function ContributionChart({ items, method }: { items: Contribution[]; method: string }) {
  const sorted = [...items].reverse();
  return (
    <Plot height={60 + items.length * 30} label="Feature contributions to this beat's predicted class"
      data={[{
        type: "bar", orientation: "h",
        x: sorted.map((c) => c.contribution),
        y: sorted.map((c) => (NO_VALUE.has(c.feature) ? FEATURE_TEXT[c.feature] : `${FEATURE_TEXT[c.feature] ?? c.feature} (${c.value.toFixed(3)})`)),
        marker: { color: sorted.map((c) => (c.contribution >= 0 ? palette().accent : palette().amber)) },
        hovertemplate: "%{x:.3f}<extra></extra>",
      }]}
      layout={{ margin: { l: 8, r: 16, t: 4, b: 44 }, yaxis: { automargin: true },
        xaxis: { title: { text: unitOf(method) }, zeroline: true, zerolinecolor: palette().slate } }} />
  );
}

export function ExplainPage({ live }: { live: Live | null }) {
  const { data } = usePoll<ExplainResponse>(live ? "/explain/latest" : null, 3000);

  const beatStrip = useMemo(() => {
    const b = data?.model.explanation?.beat;
    if (!b) return null;
    const t = b.waveform_mv.map((_, i) => i / b.fs);
    return { t, mv: b.waveform_mv, peak: [{ t: b.r_offset / b.fs, label: data?.model.explanation?.predicted_class }] };
  }, [data?.model.explanation]);

  if (!live || !data) return <Panel title="Explainability"><Empty>Explanations appear once a device is streaming.</Empty></Panel>;
  const th = data.rules.thresholds;
  const hr = live.heart_rate;
  const exp = data.model.explanation;

  return (
    <>
      <p className="lede">
        Explanations describe how the system reached its output. Feature contributions show association, not cause, and they do
        not show that the model's reasoning is medically valid.
      </p>

      <Panel title="Why the heart-rate rule gave this result">
        <table className="compare">
          <thead><tr><th scope="col">Measure</th><th scope="col">Measured</th><th scope="col">Configured threshold</th></tr></thead>
          <tbody>
            <tr><th scope="row">Average heart rate</th><td>{fmt(hr.mean_bpm)} bpm</td><td>below {th.brady_bpm} or above {th.tachy_bpm} bpm</td></tr>
            <tr><th scope="row">Valid RR intervals</th><td>{hr.valid_rr_count}</td><td>at least {th.min_valid_rr}</td></tr>
            <tr><th scope="row">Sustained for</th><td>see evidence</td><td>{th.min_duration_s} s</td></tr>
            <tr><th scope="row">Analysis window</th><td>last {th.analysis_window_s} s</td><td></td></tr>
          </tbody>
        </table>
        <h4>Result: {data.rules.result.title}</h4>
        <ul>{data.rules.result.evidence.map((e) => <li key={e}>{e}</li>)}</ul>
        <p className="fine">{data.rules.note}</p>
      </Panel>

      <div className="grid-2">
        <Panel title="One beat, explained">
          {exp && beatStrip ? (
            <>
              <p>
                The model classified this beat as <strong>{exp.predicted_class}</strong> with probability {exp.probability.toFixed(2)}
                {" "}({exp.beat?.reason_selected}, {exp.beat?.seconds_ago.toFixed(1)} s ago).
              </p>
              <EcgStrip time={beatStrip.t} mv={beatStrip.mv} peaks={beatStrip.peak} minSpanMv={2} label="Waveform of the explained beat" />
              <h4>What pushed the model toward or away from {exp.predicted_class}</h4>
              <ContributionChart items={exp.contributions} method={exp.method} />
              <p className="fine">Method: {exp.method}. {exp.note}</p>
            </>
          ) : (
            <Empty>{data.model.available ? "Waiting for enough classified beats." : "Select a beat model in Settings to see beat-level explanations."}</Empty>
          )}
          <Warnings items={data.model.warnings} title="Reliability of this explanation" />
        </Panel>

        <Panel title="What the model relies on overall">
          {data.global_importance ? (
            <>
              <Plot height={60 + data.global_importance.permutation.length * 24} label="Permutation importance of each feature"
                data={[{
                  type: "bar", orientation: "h",
                  x: [...data.global_importance.permutation].reverse().map((f) => f.mean),
                  y: [...data.global_importance.permutation].reverse().map((f) => FEATURE_TEXT[f.feature] ?? f.feature),
                  error_x: { type: "data", array: [...data.global_importance.permutation].reverse().map((f) => f.std), color: palette().muted },
                  marker: { color: palette().slate },
                  hovertemplate: "%{x:.3f}<extra></extra>",
                }]}
                layout={{ margin: { l: 8, r: 16, t: 4, b: 44 }, yaxis: { automargin: true },
                  xaxis: { title: { text: "drop in macro-F1 when shuffled" } } }} />
              <p className="fine">
                {data.global_importance.method}. Baseline macro-F1 {data.global_importance.baseline_macro_f1.toFixed(3)}.{" "}
                {data.global_importance.note}
              </p>
            </>
          ) : (
            <Empty>
              {data.model.kind === "cnn" || data.model.kind === "hybrid"
                ? "Global permutation importance is computed for Random Forest and XGBoost only. For the CNN models, use the per-beat attribution below."
                : "Global importance is computed when models are trained."}
            </Empty>
          )}
        </Panel>
      </div>

      <Panel title="CNN attribution over the waveform">
        {data.cnn_saliency ? (
          <>
            <EcgStrip time={data.cnn_saliency.t} mv={data.cnn_saliency.mv} minSpanMv={2}
              highlight={{ t: data.cnn_saliency.t, weight: data.cnn_saliency.weight }}
              label="Integrated-gradients attribution shaded over the beat waveform" />
            <p className="fine">{data.cnn_saliency.note}</p>
          </>
        ) : (
          <Empty>Select the CNN model in Settings to see where on the waveform it focused. Saliency maps are exploratory.</Empty>
        )}
      </Panel>
    </>
  );
}
