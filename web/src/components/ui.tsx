import type { ReactNode } from "react";
import type { Source } from "../api";

export function SourceBand({ source, connected, device }: { source: Source; connected: boolean; device?: string | null }) {
  const text: Record<Source, string> = {
    SIMULATED: "Simulated device. Software-generated ECG, not a person.",
    LIVE: "Live hardware stream.",
    OFFLINE: "Offline recording.",
    NONE: "No device connected.",
  };
  return (
    <div className={`band band-${source.toLowerCase()}`} role="status">
      <strong className="band-source">{source === "NONE" ? "Waiting" : source}</strong>
      <span>{text[source]}</span>
      {(source === "LIVE" || source === "SIMULATED") && (
        <span className="band-conn">
          <span className={`dot ${connected ? "dot-on" : "dot-off"}`} aria-hidden="true" />
          {connected ? "Connected" : "Disconnected"}
          {device ? ` (${device})` : ""}
        </span>
      )}
    </div>
  );
}

export function Stat({ label, value, unit, note }: { label: string; value: string; unit?: string; note?: string }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">
        {value}
        {unit && value !== "—" && <span className="stat-unit"> {unit}</span>}
      </span>
      {note && <span className="stat-note">{note}</span>}
    </div>
  );
}

export function Warnings({ items, title = "Read before relying on this" }: { items: string[]; title?: string }) {
  if (!items.length) return null;
  return (
    <aside className="warnings" aria-label={title}>
      <h3>{title}</h3>
      <ul>
        {items.map((w) => (
          <li key={w}>{w}</li>
        ))}
      </ul>
    </aside>
  );
}

export function Panel({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{title}</h2>
        {aside}
      </header>
      {children}
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>;
}

export function QualityPill({ status }: { status: string }) {
  const label: Record<string, string> = {
    good: "Good",
    fair: "Fair",
    poor: "Poor",
    insufficient_data: "Not enough data",
  };
  return <span className={`pill pill-${status}`}>{label[status] ?? status}</span>;
}
