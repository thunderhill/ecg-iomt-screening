import Plotly from "plotly.js-dist-min";
import { useEffect, useRef, useState } from "react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Current theme colours, read from the CSS tokens so charts follow light and dark mode. */
export function palette() {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return { ink: v("--ink"), slate: v("--slate"), muted: v("--muted"), line: v("--line"), amber: v("--amber"),
    accent: v("--offline"), surface: v("--surface") };
}

/** Thin wrapper around Plotly.react: re-renders in place, resizes with its container. */
export function Plot({ data, layout, height = 260, label }: { data: Any[]; layout?: Any; height?: number; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    if (!ref.current) return;
    setWidth(ref.current.clientWidth);
    const ro = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!ref.current || width === 0) return;
    const c = palette();
    const axis = { gridcolor: c.line, linecolor: c.line, zerolinecolor: c.line };
    Plotly.react(
      ref.current,
      data,
      {
        width,
        height,
        margin: { l: 56, r: 16, t: 12, b: 44 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Atkinson Hyperlegible, system-ui, sans-serif", size: 13, color: c.slate },
        showlegend: false,
        ...layout,
        xaxis: { ...axis, ...(layout?.xaxis ?? {}) },
        yaxis: { ...axis, ...(layout?.yaxis ?? {}) },
      },
      { displaylogo: false, responsive: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] },
    );
  }, [data, layout, height, width]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} role="img" aria-label={label} className="plot" />;
}
