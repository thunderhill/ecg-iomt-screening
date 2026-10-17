import { useState } from "react";
import { api } from "../api";

export function LoginPage({ onSignedIn }: { onSignedIn: () => void }) {
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await api("/auth/login", { method: "POST", body: JSON.stringify({ token: token.trim() }) });
      setToken("");
      onSignedIn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <form className="panel login-card" onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <p className="brand">IoMT ECG screening lab</p>
        <h1>Sign in</h1>
        <label className="fields-single">
          <span>Access token</span>
          <input type="password" autoComplete="current-password" value={token} autoFocus
            onChange={(e) => setToken(e.target.value)} aria-describedby="token-help" />
        </label>
        <p className="fine" id="token-help">
          The token printed by <code>scripts/security_setup.py init</code>. Lost it? Run
          {" "}<code>security_setup.py rotate-admin</code> on the gateway to issue a new one.
        </p>
        {error && <p className="error" role="alert">{error}</p>}
        <div className="actions">
          <button className="primary" type="submit" disabled={busy || !token.trim()}>Sign in</button>
        </div>
        <p className="fine">
          {window.location.protocol === "https:"
            ? "This connection is encrypted."
            : "This connection is not encrypted. That is only acceptable on this computer (127.0.0.1)."}
          {" "}Research prototype, not a medical device.
        </p>
      </form>
    </div>
  );
}
