import Plotly from "plotly.js-dist-min";
import { useEffect, useRef, useState } from "react";
import { palette } from "./Plot";

/**
 * ECG drawn on standard calibrated paper: 25 mm/s and 10 mm/mV.
 * One large box = 0.2 s horizontally = 0.5 mV vertically, so boxes are square
 * and intervals and amplitudes can be read straight off the grid.
 */
const MM_PER_S = 25;
const MM_PER_MV = 10;
const MAX_PLOT_HEIGHT = 380; // short strips shrink in width instead of growing taller

export interface Marker {
  t: number;
  label?: string;
}

export function EcgStrip({
  time,
  mv,
  peaks = [],
  highlight,
  label,
  minSpanMv = 3,
}: {
  time: number[];
  mv: number[];
  peaks?: Marker[];
  highlight?: { t: number[]; weight: number[] };
  label: string;
  minSpanMv?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const figRef = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(0);

  // Measure the figure, not the plot: the plot shrinks to fit short strips, so
  // measuring it would feed its own width back into the calculation.
  useEffect(() => {
    if (!figRef.current) return;
    setWidth(figRef.current.clientWidth);
    const ro = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width));
    ro.observe(figRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || time.length === 0 || width === 0) return;
    const t0 = time[0];
    const t1 = time[time.length - 1];
    const finite = mv.filter((v) => Number.isFinite(v));
    let lo = Math.min(...finite);
    let hi = Math.max(...finite);
    const span = Math.max(minSpanMv, hi - lo + 0.4);
    const mid = (hi + lo) / 2;
    lo = Math.floor((mid - span / 2) / 0.5) * 0.5;
    hi = Math.ceil((mid + span / 2) / 0.5) * 0.5;

    // Height follows from the calibration: the plot keeps the paper's proportions.
    const margin = { l: 44, r: 8, t: 8, b: 32 };
    // One scale for both axes keeps the paper's proportions; cap it so short strips stay a sensible height.
    const available = Math.max(120, width - margin.l - margin.r);
    const pxPerMm = Math.min(available / ((t1 - t0) * MM_PER_S), MAX_PLOT_HEIGHT / ((hi - lo) * MM_PER_MV));
    const plotW = (t1 - t0) * MM_PER_S * pxPerMm;
    const plotH = (hi - lo) * MM_PER_MV * pxPerMm;

    const traces: object[] = [
      { x: time, y: mv, type: "scattergl", mode: "lines", line: { color: "#1B2A41", width: 1.4 }, hoverinfo: "x+y", name: "ECG" },
    ];
    if (highlight) {
      traces.unshift({
        x: highlight.t,
        y: highlight.t.map(() => hi),
        type: "bar",
        base: lo,
        marker: { color: highlight.weight.map((w) => `rgba(43,108,176,${Math.min(0.55, w * 0.55)})`) },
        width: highlight.t.length > 1 ? highlight.t[1] - highlight.t[0] : 0.004,
        hoverinfo: "skip",
        name: "attribution",
      });
    }
    if (peaks.length) {
      const yAt = (t: number) => {
        let best = 0;
        for (let i = 0; i < time.length; i++) if (Math.abs(time[i] - t) < Math.abs(time[best] - t)) best = i;
        return mv[best];
      };
      traces.push({
        x: peaks.map((p) => p.t),
        y: peaks.map((p) => yAt(p.t) + 0.25),
        text: peaks.map((p) => p.label ?? ""),
        type: "scatter",
        mode: peaks.some((p) => p.label) ? "markers+text" : "markers",
        textposition: "top center",
        textfont: { color: "#9B2C2C", size: 12 },
        marker: { symbol: "triangle-down", size: 9, color: "#9B2C2C" },
        hovertemplate: "R peak %{x:.2f} s<extra></extra>",
        name: "R peaks",
      });
    }

    // Major grid every 0.2 s, but only whole seconds are labelled so the axis stays readable.
    const tickvals: number[] = [];
    for (let v = Math.ceil(t0 / 0.2 - 1e-9) * 0.2; v <= t1 + 1e-9; v += 0.2) tickvals.push(Number(v.toFixed(3)));
    const ticktext = tickvals.map((v) => (Math.abs(v - Math.round(v)) < 1e-6 ? `${Math.round(v)} s` : ""));
    const axis = { showline: false, zeroline: false, ticks: "", gridcolor: "#E89A9A", gridwidth: 1 };
    Plotly.react(
      el,
      traces,
      {
        width: plotW + margin.l + margin.r,
        height: plotH + margin.t + margin.b,
        margin,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#FFFBFB",
        font: { family: "Atkinson Hyperlegible, system-ui, sans-serif", size: 12, color: palette().slate },
        showlegend: false,
        barmode: "overlay",
        xaxis: {
          ...axis,
          range: [t0, t1],
          tickmode: "array",
          tickvals,
          ticktext,
          minor: { dtick: 0.04, gridcolor: "#F6C9C9", gridwidth: 0.6, showgrid: true },
          fixedrange: false,
        },
        yaxis: {
          ...axis,
          range: [lo, hi],
          dtick: 0.5,
          ticksuffix: " mV",
          minor: { dtick: 0.1, gridcolor: "#F6C9C9", gridwidth: 0.6, showgrid: true },
          fixedrange: true,
        },
      },
      { displaylogo: false, responsive: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] },
    );
  }, [time, mv, peaks, highlight, width, minSpanMv]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return (
    <figure className="strip" ref={figRef}>
      <div ref={ref} role="img" aria-label={label} />
      <figcaption>Standard paper speed 25 mm/s and gain 10 mm/mV: one large box is 0.2 s wide and 0.5 mV tall.</figcaption>
    </figure>
  );
}
