import { CATEGORY_LABEL, type Live, type ModelPart, type RuleResult } from "../api";
import { Plot, palette } from "../components/Plot";
import { Empty, Panel, QualityPill, Warnings } from "../components/ui";

const CATEGORIES = Object.entries(CATEGORY_LABEL);

function Report({ heading, result }: { heading: string; result: RuleResult | (ModelPart & { title: string }) }) {
  return (
    <article className="report" aria-label={heading}>
      <p className="report-model">{heading}</p>
      <h3 className={`report-result cat-${result.category}`}>{result.title}</h3>
      <h4>Evidence</h4>
      {result.evidence && result.evidence.length ? (
        <ul>{result.evidence.map((e) => <li key={e}>{e}</li>)}</ul>
      ) : (
        <p className="fine">No supporting measurements for this result.</p>
      )}
      <h4>Interpretation</h4>
      <p>{result.interpretation}</p>
      <h4>Limitations</h4>
      <ul className="limitations">{(result.limitations ?? []).map((l) => <li key={l}>{l}</li>)}</ul>
    </article>
  );
}

export function ScreeningPage({ live }: { live: Live | null }) {
  if (!live) return <Panel title="Rhythm screening"><Empty>Screening starts when a device is streaming.</Empty></Panel>;
  const m = live.model;
  const probs = m.available && m.mean_probability ? Object.entries(m.mean_probability) : [];

  return (
    <>
      <p className="lede">
        Screening flags are preliminary research signals for review. They are not diagnoses, and a recording with no flag is
        not shown to be normal.
      </p>
      <div className="grid-2">
        <Panel title="Heart-rate rules" aside={<QualityPill status={live.quality.status} />}>
          <Report heading="Model A: configurable heart-rate thresholds" result={live.screening} />
          <Warnings items={live.screening.warnings} />
        </Panel>

        <Panel title={m.available ? `Beat model (${m.kind?.toUpperCase()})` : "Beat model"}>
          {m.available && m.title ? (
            <>
              <Report heading="Classifies each beat as N, S, V or F, then summarises the window" result={{ ...m, title: m.title }} />
              {probs.length > 0 && (
                <>
                  <h4>Mean model probability per class</h4>
                  <Plot height={170} label="Mean model probability for each beat class"
                    data={[{ type: "bar", orientation: "h", x: probs.map(([, p]) => p), y: probs.map(([c]) => c),
                      marker: { color: palette().slate }, hovertemplate: "%{y}: %{x:.2f}<extra></extra>" }]}
                    layout={{ xaxis: { range: [0, 1], title: { text: "probability" } }, yaxis: { autorange: "reversed" },
                      margin: { l: 40, r: 16, t: 4, b: 40 } }} />
                  <p className="fine">{m.probability_note}</p>
                </>
              )}
            </>
          ) : (
            <Empty>
              No beat model is selected. Choose Random Forest or XGBoost in Settings after training them with
              python scripts/train_models.py.
            </Empty>
          )}
          <Warnings items={m.warnings} title="Reliability of this result" />
        </Panel>
      </div>

      <Panel title="Possible result categories">
        <ul className="categories">
          {CATEGORIES.map(([key, label]) => (
            <li key={key} aria-current={key === live.screening.category || key === m.category ? "true" : undefined}>
              {label}
            </li>
          ))}
        </ul>
        <p className="fine">
          The beat model supports AAMI classes N, S, V and F only. Q beats (unclassifiable and paced) are not supported:
          MIT-BIH contains too few of them to train or test on.
        </p>
      </Panel>
    </>
  );
}
