import { useEffect, useState } from "react";
import { api, SIGNED_OUT_EVENT, usePoll, type Live, type Status } from "./api";
import { SourceBand } from "./components/ui";
import { EvaluationPage } from "./pages/EvaluationPage";
import { ExplainPage } from "./pages/ExplainPage";
import { HistoryPage } from "./pages/HistoryPage";
import { LoginPage } from "./pages/LoginPage";
import { LivePage } from "./pages/LivePage";
import { ScreeningPage } from "./pages/ScreeningPage";
import { SettingsPage } from "./pages/SettingsPage";

const PAGES = [
  { id: "live", label: "Live monitoring" },
  { id: "screening", label: "Rhythm screening" },
  { id: "explain", label: "Explainability" },
  { id: "history", label: "ECG history" },
  { id: "evaluation", label: "Model evaluation" },
  { id: "settings", label: "Settings" },
] as const;
type PageId = (typeof PAGES)[number]["id"];

function parseHash(): { page: PageId; detail: string | null } {
  const [head, ...rest] = window.location.hash.replace("#", "").split("/");
  const page = (PAGES.find((p) => p.id === head)?.id ?? "live") as PageId;
  return { page, detail: rest.length ? decodeURIComponent(rest.join("/")) : null };
}

export function App() {
  const [auth, setAuth] = useState<"checking" | "signed-in" | "signed-out">("checking");

  useEffect(() => {
    api("/auth/me").then(() => setAuth("signed-in"), () => setAuth("signed-out"));
    const onSignedOut = () => setAuth("signed-out");
    window.addEventListener(SIGNED_OUT_EVENT, onSignedOut);
    return () => window.removeEventListener(SIGNED_OUT_EVENT, onSignedOut);
  }, []);

  if (auth === "checking") return null;
  if (auth === "signed-out") return <LoginPage onSignedIn={() => setAuth("signed-in")} />;
  return <Dashboard onSignOut={async () => { await api("/auth/logout", { method: "POST" }).catch(() => undefined); setAuth("signed-out"); }} />;
}

function Dashboard({ onSignOut }: { onSignOut: () => void }) {
  const [{ page, detail }, setRoute] = useState(parseHash);
  useEffect(() => {
    const onHash = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const status = usePoll<Status>("/status", 2000);
  const streaming = status.data?.mode && status.data.mode !== "NONE";
  const needsLive = page === "live" || page === "screening" || page === "explain";
  const live = usePoll<Live>(streaming && needsLive ? "/live?seconds=10" : null, 1000);

  const source = page === "history" ? "OFFLINE" : status.data?.mode ?? "NONE";
  const title = PAGES.find((p) => p.id === page)!.label;

  return (
    <div className="shell">
      <nav className="nav" aria-label="Pages">
        <p className="brand">IoMT ECG screening lab</p>
        <ul>
          {PAGES.map((p) => (
            <li key={p.id}>
              <a href={`#${p.id}`} aria-current={page === p.id ? "page" : undefined}>{p.label}</a>
            </li>
          ))}
        </ul>
        <div className="nav-foot">
          <button className="link" onClick={onSignOut}>Sign out</button>
          <p>Research prototype. Not a medical device.</p>
        </div>
      </nav>
      <div className="main">
        <SourceBand source={source} connected={page === "history" ? true : !!status.data?.connected}
          device={page === "history" ? null : status.data?.device_id} />
        {status.error && <p className="banner-error" role="alert">Server unreachable: {status.error.message}. Start it with python scripts/serve.py.</p>}
        <main>
          <h1>{title}</h1>
          {page === "live" && <LivePage live={live.data} error={live.error ?? status.error} noDevice={status.data?.mode === "NONE"} />}
          {page === "screening" && <ScreeningPage live={live.data} />}
          {page === "explain" && <ExplainPage live={live.data} />}
          {page === "history" && (
            <HistoryPage selected={detail} onSelect={(id) => { window.location.hash = `history/${encodeURIComponent(id)}`; }} />
          )}
          {page === "evaluation" && <EvaluationPage />}
          {page === "settings" && <SettingsPage />}
        </main>
      </div>
    </div>
  );
}
