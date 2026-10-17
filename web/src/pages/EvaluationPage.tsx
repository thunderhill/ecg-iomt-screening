import { fmt, usePoll } from "../api";
import { Plot, palette } from "../components/Plot";
import { Empty, Panel } from "../components/ui";

interface PerClass { precision: number; sensitivity: number; specificity: number | null; f1: number; support: number; roc_auc_ovr?: number }
interface ModelEval {
  available: boolean;
  reason?: string;
  test: { accuracy: number; balanced_accuracy: number; macro_f1: number; per_class: Record<string, PerClass>; confusion_matrix: number[][];
    classes: string[]; roc_auc_ovr_macro: number | null; n: number };
  cv_macro_f1: { mean: number | null; std: number | null; folds: number[]; note?: string };
  caveat?: string;
  latency: { single_beat_median_ms: number; batch_per_beat_us: number; note: string };
  size_bytes: number;
  params: Record<string, unknown>;
  n_train: number;
  n_test: number;
}
interface Summary {
  dataset: { name: string; split: string; fs: number; lead: string; classes: string[]; excluded_note: string;
    train_counts: Record<string, number>; test_counts: Record<string, number>;
    excluded_test: Record<string, number> };
  rpeak_detection: Record<string, { sensitivity: number; ppv: number; f1: number; tp: number; fp: number; fn: number; mode: string }>;
  models: Record<string, ModelEval>;
  rules: { description: string; windows: number; agreement: number; note: string; thresholds: Record<string, number> };
  separation_note: string;
  seed_variance?: { seeds: number[]; note: string;
    models: Record<string, { macro_f1_mean: number; macro_f1_std: number; runs: { seed: number; macro_f1: number }[] }> };
}

const ORDER = ["N", "S", "V", "F", "Q"];
const counts = (c: Record<string, number>) =>
  Object.entries(c).sort(([a], [b]) => ORDER.indexOf(a) - ORDER.indexOf(b)).map(([k, n]) => `${k} ${n.toLocaleString()}`).join(", ");

const NAMES: Record<string, string> = { rf: "Random Forest", xgb: "XGBoost", cnn: "1D-CNN", hybrid: "CNN + XGBoost" };

function Confusion({ m }: { m: ModelEval }) {
  const cm = m.test.confusion_matrix;
  // Row-normalised so rare classes are readable; counts shown in each cell.
  const z = cm.map((row) => { const s = row.reduce((a, b) => a + b, 0) || 1; return row.map((v) => v / s); });
  return (
    <Plot height={300} label="Confusion matrix, rows are true classes"
      data={[{ type: "heatmap", z, x: m.test.classes, y: m.test.classes, colorscale: [[0, palette().surface], [1, palette().accent]],
        showscale: false, text: cm.map((r) => r.map(String)), texttemplate: "%{text}", hovertemplate: "true %{y}, predicted %{x}: %{text}<extra></extra>" }]}
      layout={{ xaxis: { title: { text: "predicted" }, side: "bottom" }, yaxis: { title: { text: "true" }, autorange: "reversed" },
        margin: { l: 60, r: 8, t: 8, b: 48 } }} />
  );
}

export function EvaluationPage() {
  const { data, error } = usePoll<Summary>("/evaluation", 0);
  if (error) return <Panel title="Model evaluation"><Empty>No evaluation is available yet. Run python scripts/train_models.py.</Empty></Panel>;
  if (!data) return <Panel title="Model evaluation"><Empty>Loading evaluation.</Empty></Panel>;
  const models = Object.entries(data.models).filter(([, m]) => m.available);

  return (
    <>
      <p className="lede">{data.separation_note} No synthetic data is used in any figure on this page.</p>

      <Panel title="Evaluation data">
        <dl className="facts">
          <dt>Dataset</dt><dd>{data.dataset.name}, lead {data.dataset.lead}, resampled to {data.dataset.fs} Hz</dd>
          <dt>Split</dt><dd>{data.dataset.split}</dd>
          <dt>Training beats</dt><dd>{counts(data.dataset.train_counts)}</dd>
          <dt>Test beats</dt><dd>{counts(data.dataset.test_counts)}</dd>
          <dt>Exclusions</dt><dd>{data.dataset.excluded_note} Test beats our detector found with no matching reference beat: {data.dataset.excluded_test.unmatched_detection}.</dd>
          <dt>Label level</dt><dd>Beat-level AAMI classes. These are not rhythm or recording-level diagnoses.</dd>
        </dl>
      </Panel>

      <Panel title="Model comparison on held-out patients">
        <div className="table-scroll">
          <table className="compare">
            <thead>
              <tr><th scope="col">Model</th><th scope="col">Macro-F1</th><th scope="col">Cross-validation macro-F1</th><th scope="col">Accuracy</th>
                <th scope="col">Balanced accuracy</th><th scope="col">ROC-AUC (one-vs-rest)</th><th scope="col">Latency per beat</th><th scope="col">Size</th></tr>
            </thead>
            <tbody>
              {models.map(([k, m]) => (
                <tr key={k}>
                  <th scope="row">{NAMES[k] ?? k}</th>
                  <td>{m.test.macro_f1.toFixed(3)}</td>
                  <td>{m.cv_macro_f1.mean === null ? "not run" : `${m.cv_macro_f1.mean.toFixed(3)} ± ${(m.cv_macro_f1.std ?? 0).toFixed(3)}`}</td>
                  <td>{m.test.accuracy.toFixed(3)}</td>
                  <td>{m.test.balanced_accuracy.toFixed(3)}</td>
                  <td>{fmt(m.test.roc_auc_ovr_macro, 3)}</td>
                  <td>{m.latency.single_beat_median_ms.toFixed(2)} ms</td>
                  <td>{(m.size_bytes / 1e6).toFixed(1)} MB</td>
                </tr>
              ))}
              {Object.entries(data.models).filter(([, m]) => !m.available).map(([k, m]) => (
                <tr key={k}><th scope="row">{NAMES[k] ?? k}</th><td colSpan={7}>Not available: {m.reason}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="fine">
          Accuracy is dominated by the N class (about 89% of test beats); macro-F1 and balanced accuracy weigh every class equally.
          Cross-validation is grouped by record on the training set; it was not run for the CNN-based models.
          Latency is per beat, single-threaded, on the laptop that trained the models. Models run on the laptop, not on the ESP32.
        </p>
      </Panel>

      {models.map(([k, m]) => (
        <div className="grid-2" key={k}>
          <Panel title={`${NAMES[k] ?? k}: per-class results`}>
            <div className="table-scroll">
              <table className="compare">
                <thead><tr><th scope="col">Class</th><th scope="col">Sensitivity</th><th scope="col">Specificity</th><th scope="col">Precision</th><th scope="col">F1</th><th scope="col">Test beats</th></tr></thead>
                <tbody>
                  {Object.entries(m.test.per_class).map(([c, p]) => (
                    <tr key={c} className={p.sensitivity < 0.5 ? "weak" : undefined}>
                      <th scope="row">{c}</th><td>{p.sensitivity.toFixed(3)}</td><td>{fmt(p.specificity, 3)}</td>
                      <td>{p.precision.toFixed(3)}</td><td>{p.f1.toFixed(3)}</td><td>{p.support.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="fine">Rows shaded amber have sensitivity below 0.5: most of those beats are missed.</p>
          </Panel>
          <Panel title={`${NAMES[k] ?? k}: confusion matrix`}>
            <Confusion m={m} />
            <p className="fine">Shading is row-normalised; numbers are beat counts.</p>
          </Panel>
        </div>
      ))}

      {data.seed_variance && (
        <Panel title="Run-to-run variation of the CNN-based models">
          <table className="compare">
            <thead><tr><th scope="col">Model</th><th scope="col">Macro-F1, mean ± SD</th><th scope="col">Per training seed</th></tr></thead>
            <tbody>
              {Object.entries(data.seed_variance.models).map(([k, v]) => (
                <tr key={k}>
                  <th scope="row">{NAMES[k] ?? k}</th>
                  <td>{v.macro_f1_mean.toFixed(3)} ± {v.macro_f1_std.toFixed(3)}</td>
                  <td>{v.runs.map((r) => `seed ${r.seed}: ${r.macro_f1.toFixed(3)}`).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="fine">
            {data.seed_variance.note} Differences between models smaller than this spread should not be read as one model
            being better.
          </p>
        </Panel>
      )}
      {models.filter(([, m]) => m.caveat).map(([k, m]) => (
        <p className="fine" key={k}>{NAMES[k] ?? k}: {m.caveat}</p>
      ))}

      <div className="grid-2">
        <Panel title="R-peak detection">
          <table className="compare">
            <thead><tr><th scope="col">Records</th><th scope="col">Sensitivity</th><th scope="col">Positive predictive value</th><th scope="col">F1</th></tr></thead>
            <tbody>
              {Object.entries(data.rpeak_detection).map(([s, d]) => (
                <tr key={s}><th scope="row">{s}</th><td>{d.sensitivity.toFixed(4)}</td><td>{d.ppv.toFixed(4)}</td><td>{d.f1.toFixed(4)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="fine">{Object.values(data.rpeak_detection)[0]?.mode}.</p>
        </Panel>
        <Panel title="Heart-rate rules">
          <p>
            Agreement {(data.rules.agreement * 100).toFixed(1)}% across {data.rules.windows} one-minute windows of the test records.
          </p>
          <p className="fine">{data.rules.description}. {data.rules.note}</p>
        </Panel>
      </div>
    </>
  );
}
