import { useCallback, useEffect, useRef, useState } from "react";

export type Source = "LIVE" | "SIMULATED" | "OFFLINE" | "NONE";

export interface PacketStats {
  received: number;
  accepted: number;
  malformed: number;
  fs_mismatch: number;
  duplicates: number;
  lost_packets: number;
  lost_samples_estimated: number;
  session_restarts: number;
  out_of_order: number;
}

export interface Status {
  mode: Source;
  connected: boolean;
  message?: string;
  device_id?: string | null;
  fs?: number;
  seconds_since_last_packet?: number | null;
  packet_stats?: PacketStats;
  selected_model?: string;
  model_error?: string | null;
}

export interface Quality {
  status: "good" | "fair" | "poor" | "insufficient_data";
  reasons: string[];
  std_mv: number;
  clip_fraction: number;
  noise_ratio: number;
  lead_off: boolean;
}

export interface RuleResult {
  model: string;
  category: string;
  title: string;
  evidence: string[];
  interpretation: string;
  limitations: string[];
  warnings: string[];
}

export interface Contribution {
  feature: string;
  value: number;
  contribution: number;
  direction: string;
}

export interface Explanation {
  predicted_class: string;
  probability: number;
  probabilities: Record<string, number>;
  method: string;
  baseline: number | null;
  contributions: Contribution[];
  note: string;
  beat?: { seconds_ago: number; reason_selected: string; waveform_mv: number[]; r_offset: number; fs: number };
}

export interface ModelPart {
  kind: string | null;
  available: boolean;
  category?: string;
  title?: string;
  evidence?: string[];
  interpretation?: string;
  limitations?: string[];
  class_counts?: Record<string, number>;
  mean_probability?: Record<string, number>;
  probability_note?: string;
  warnings: string[];
  explanation?: Explanation | null;
}

export interface Live {
  source: Source;
  device_id: string | null;
  simulated: boolean | null;
  fs: number;
  connected: boolean;
  seconds_since_last_packet: number | null;
  total_samples: number;
  timeline_s: number;
  segment_s: number;
  gaps: number;
  analysis_restarts: number;
  lead_off: { lo_plus: boolean; lo_minus: boolean };
  packet_stats: PacketStats;
  last_error: string | null;
  heart_rate: {
    current_bpm: number | null;
    mean_bpm: number | null;
    min_bpm: number | null;
    max_bpm: number | null;
    valid_rr_count: number;
    flagged_rr_count: number;
    seconds_since_last_beat: number | null;
    warnings: string[];
  };
  rr_recent_s: number[];
  hrv: {
    available: boolean;
    mean_nn_ms: number | null;
    sdnn_ms: number | null;
    rmssd_ms: number | null;
    pnn50_pct: number | null;
    nn_count: number;
    warnings: string[];
  };
  quality: Quality;
  screening: RuleResult;
  model: ModelPart;
  waveform: { time_s: number[]; raw_mv: number[]; filtered_mv: number[]; r_peak_offsets: number[] };
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** Fired when the server says the session is missing or expired. */
export const SIGNED_OUT_EVENT = "ecg:signed-out";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, credentials: "same-origin", ...init });
  if (res.status === 401 && path !== "/auth/login") {
    window.dispatchEvent(new Event(SIGNED_OUT_EVENT));
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

/** Poll an endpoint. Polling pauses while the tab is hidden. */
export function usePoll<T>(path: string | null, intervalMs: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [loading, setLoading] = useState(path !== null);
  const pathRef = useRef(path);
  pathRef.current = path;

  const refresh = useCallback(async () => {
    const current = pathRef.current;
    if (current === null) return;
    try {
      const value = await api<T>(current);
      if (pathRef.current === current) {
        setData(value);
        setError(null);
      }
    } catch (e) {
      if (pathRef.current === current) setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (path === null) return;
    setLoading(true);
    refresh();
    if (intervalMs <= 0) return;
    const id = window.setInterval(() => {
      if (!document.hidden) refresh();
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [path, intervalMs, refresh]);

  return { data, error, loading, refresh };
}

export const fmt = (v: number | null | undefined, digits = 0, unit = "") =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(digits)}${unit ? ` ${unit}` : ""}`;

export const CATEGORY_LABEL: Record<string, string> = {
  no_configured_flag: "No configured screening flag detected",
  possible_elevated_heart_rate: "Possible elevated heart rate",
  possible_low_heart_rate: "Possible low heart rate",
  possible_arrhythmia_pattern: "Possible arrhythmia pattern",
  insufficient_signal_quality: "Insufficient signal quality",
  insufficient_data: "Insufficient data",
  model_unavailable: "Model unavailable",
};
