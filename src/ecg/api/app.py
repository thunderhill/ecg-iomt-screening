"""FastAPI gateway of the IoMT system: device ingest, analysis, storage, configuration.

Run with:  python scripts/serve.py [--tls] [--host ...]   (see docs/security.md)

Security (docs/security.md):
* Devices authenticate with a per-device bearer token on the WebSocket upgrade;
  the LIVE/SIMULATED label comes from the device registry.
* Every HTTP route except /health and sign-in needs a session cookie
  (HttpOnly, SameSite=Strict, Secure under TLS).
* Cross-site state-changing requests are refused; security headers are added.
* TLS itself is terminated by uvicorn (scripts/serve.py), which refuses to
  listen on the network without it.
Logs never contain tokens, cookies or waveform samples.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from .. import __version__
from ..analysis import analyze_recording
from ..config import AppConfig, ConfigError
from ..ingest import PacketError, parse_hello
from ..ml.inference import ModelUnavailable, available_models, load_bundle
from ..pipeline import LivePipeline
from ..security import LoginThrottle, SecurityStore
from ..storage import Storage
from .schemas import ConfigPatch, HealthResponse, LoginRequest, SaveRecordingRequest

log = logging.getLogger("ecg.api")
EVENT_LOG_EVERY_S = 5.0
ANALYSIS_ONLY_KEYS = {"screening", "quality", "selected_model", "retention_days"}
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
PUBLIC_PATHS = {"/health", "/auth/login", "/auth/logout"}
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _bearer(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None


def _same_origin(origin: str, request: Request) -> bool:
    return origin.rstrip("/") == f"{request.url.scheme}://{request.url.netloc}"


class AppState:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.storage = Storage(cfg.data_dir)
        self.security = SecurityStore(self.storage.db_path)
        self.throttle = LoginThrottle()
        self.pipeline: LivePipeline | None = None
        self._bundles: dict[str, dict] = {}
        self.model_error: str | None = None
        self._last_logged: tuple[str, str | None] | None = None

    def models_dir(self) -> Path:
        return Path(self.cfg.models_dir)

    def bundle_for(self, kind: str) -> dict | None:
        """Load (and cache) the bundle for ``kind``. Returns None for rules or when unavailable."""
        self.model_error = None
        if kind == "rules":
            return None
        if kind not in self._bundles:
            try:
                bundle = load_bundle(self.models_dir() / f"{kind}.joblib")
            except ModelUnavailable as exc:
                self.model_error = str(exc)
                return None
            inner = getattr(bundle["model"], "model", bundle["model"])
            if hasattr(inner, "n_jobs"):
                inner.n_jobs = 1  # single-beat latency is better without thread fan-out
            self._bundles[kind] = bundle
        return self._bundles[kind]

    def attach_model(self, pipe: LivePipeline) -> None:
        kind = self.cfg.selected_model
        bundle = self.bundle_for(kind)
        pipe.set_model(kind if bundle is not None else None, bundle)

    def maybe_log_event(self, snap: dict) -> None:
        """Store a screening event when the rule or model category changes."""
        rule_cat = snap["screening"]["category"]
        model_cat = snap["model"].get("category") if snap["model"].get("available") else None
        key = (rule_cat, model_cat)
        if key != self._last_logged:
            self._last_logged = key
            self.storage.log_screening(None, snap["source"], "rules", rule_cat, snap["screening"])
            if model_cat is not None:
                payload = {k: v for k, v in snap["model"].items() if k != "explanation"}
                self.storage.log_screening(None, snap["source"], snap["model"]["kind"], model_cat, payload)


def create_app(cfg: AppConfig | None = None, *, restore_settings: bool = True) -> FastAPI:
    state = AppState(cfg or AppConfig())
    if restore_settings:
        _restore_saved_settings(state)
    app = FastAPI(
        title="ECG Screening API (research prototype)",
        version=__version__,
        description=(
            "Research use only. Not a medical device. Data labelled SIMULATED comes from a "
            "software ESP32 emulator and must not be treated as patient data."
        ),
    )
    app.state.ecg = state
    sec = state.cfg.security

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """Session check, cross-site request refusal and security headers for every HTTP request."""
        path = request.url.path
        origin = request.headers.get("origin")
        public = path in PUBLIC_PATHS or path == "/ui" or path.startswith("/ui/")
        if request.method in UNSAFE_METHODS and origin and not _same_origin(origin, request):
            response: Response = JSONResponse({"detail": "cross-site request refused"}, status_code=403)
        elif not public and not state.security.session_valid(request.cookies.get(sec.cookie_name)):
            response = JSONResponse({"detail": "sign in required"}, status_code=401)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
        if sec.tls:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    def _require_pipeline() -> LivePipeline:
        if state.pipeline is None:
            raise HTTPException(status_code=404, detail="no device stream connected yet")
        return state.pipeline

    def _require_recording(rec_id: str):
        info = state.storage.get_recording(rec_id)
        if info is None:
            raise HTTPException(status_code=404, detail="recording not found")
        return info

    # ----- authentication -----

    @app.post("/auth/login")
    def login(body: LoginRequest, request: Request, response: Response) -> dict:
        client = request.client.host if request.client else "unknown"
        if state.throttle.blocked(client):
            raise HTTPException(status_code=429, detail="Too many failed sign-ins. Try again in 15 minutes.")
        if not state.security.verify_admin(body.token):
            state.throttle.record_failure(client)
            log.warning("failed sign-in")
            raise HTTPException(status_code=401, detail="That access token is not valid.")
        state.throttle.reset(client)
        session = state.security.create_session(sec.session_ttl_s)
        response.set_cookie(sec.cookie_name, session, max_age=int(sec.session_ttl_s), path="/",
                            httponly=True, samesite="strict", secure=sec.tls)
        return {"authenticated": True}

    @app.post("/auth/logout")
    def logout(request: Request, response: Response) -> dict:
        state.security.end_session(request.cookies.get(sec.cookie_name))
        response.delete_cookie(sec.cookie_name, path="/", httponly=True, samesite="strict", secure=sec.tls)
        return {"authenticated": False}

    @app.get("/auth/me")
    def me() -> dict:
        return {"authenticated": True, "tls": sec.tls}

    @app.get("/devices")
    def devices() -> dict:
        return {"items": [
            {"device_id": d.device_id, "kind": d.kind, "source": d.source_label, "created_at": d.created_at,
             "last_seen": d.last_seen, "revoked": d.revoked}
            for d in state.security.list_devices()
        ], "note": "Add or revoke devices with scripts/security_setup.py."}

    # ----- status and live data -----

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/status")
    def status() -> dict:
        if state.pipeline is None:
            return {"mode": "NONE", "connected": False,
                    "message": "No device has connected. Start the simulator or the ESP32."}
        snap = state.pipeline.snapshot(recent_waveform_s=1.0)
        return {
            "mode": snap["source"],
            "device_id": snap["device_id"],
            "connected": snap["connected"],
            "fs": snap["fs"],
            "total_samples": snap["total_samples"],
            "seconds_since_last_packet": snap["seconds_since_last_packet"],
            "lead_off": snap["lead_off"],
            "packet_stats": snap["packet_stats"],
            "analysis_restarts": snap["analysis_restarts"],
            "last_error": snap["last_error"],
            "selected_model": state.cfg.selected_model,
            "model_error": state.model_error,
        }

    @app.get("/live")
    def live(seconds: float = Query(10.0, gt=0, le=60)) -> dict:
        """Everything the Live page needs in one request (one consistent snapshot)."""
        snap = _require_pipeline().snapshot(recent_waveform_s=seconds)
        state.maybe_log_event(snap)
        return snap

    @app.get("/ecg/latest")
    def ecg_latest(seconds: float = Query(10.0, gt=0, le=60)) -> dict:
        snap = _require_pipeline().snapshot(recent_waveform_s=seconds)
        return {"source": snap["source"], "fs": snap["fs"], **snap["waveform"]}

    @app.get("/metrics/latest")
    def metrics_latest() -> dict:
        snap = _require_pipeline().snapshot(recent_waveform_s=1.0)
        return {"source": snap["source"], "heart_rate": snap["heart_rate"], "hrv": snap["hrv"],
                "rr_recent_s": snap["rr_recent_s"], "quality": snap["quality"]}

    @app.get("/screening/latest")
    def screening_latest() -> dict:
        snap = _require_pipeline().snapshot(recent_waveform_s=1.0)
        state.maybe_log_event(snap)
        model = {k: v for k, v in snap["model"].items() if k != "explanation"}
        return {"source": snap["source"], "rules": snap["screening"], "model": model, "quality": snap["quality"]}

    @app.get("/explain/latest")
    def explain_latest() -> dict:
        snap = _require_pipeline().snapshot(recent_waveform_s=1.0)
        kind = snap["model"].get("kind")
        importance = None
        if kind:
            path = state.models_dir() / f"importance_{kind}.json"
            importance = json.loads(path.read_text()) if path.exists() else None
        return {
            "source": snap["source"],
            "rules": {
                "result": snap["screening"],
                "thresholds": state.cfg.screening.__dict__,
                "rr_plausibility": state.cfg.rr.__dict__,
                "note": "Thresholds are research parameters, not clinical standards.",
            },
            "model": snap["model"],
            "global_importance": importance,
            "quality": snap["quality"],
            "cnn_saliency": _saliency_payload(snap["model"]),
        }

    @app.get("/screening/history")
    def screening_history(limit: int = Query(50, ge=1, le=500)) -> dict:
        return {"items": state.storage.recent_screening(limit)}

    # ----- recordings -----

    @app.post("/recordings", status_code=201)
    def save_recording(body: SaveRecordingRequest) -> dict:
        pipe = _require_pipeline()
        counts = pipe.buffered_counts()
        if counts.size == 0:
            raise HTTPException(status_code=409, detail="no samples buffered yet")
        info = state.storage.save_recording(
            counts, source=pipe.source, fs=pipe.cfg.sample_rate_hz, device_id=pipe.receiver.device_id,
            config=pipe.cfg.to_dict(), model_version=pipe.model_kind, notes=body.notes,
        )
        return {"id": info.id, "source": info.source, "n_samples": info.n_samples, "duration_s": info.duration_s}

    @app.get("/recordings")
    def list_recordings(limit: int = Query(100, ge=1, le=500)) -> dict:
        return {"items": [
            {"id": r.id, "created_at": r.created_at, "source": r.source, "device_id": r.device_id, "fs": r.fs,
             "n_samples": r.n_samples, "duration_s": r.duration_s, "units": r.units, "notes": r.notes}
            for r in state.storage.list_recordings(limit)
        ]}

    @app.get("/recordings/{rec_id}/waveform")
    def recording_waveform(rec_id: str, start_s: float = Query(0.0, ge=0),
                           duration_s: float = Query(30.0, gt=0, le=600)) -> dict:
        _require_recording(rec_id)
        info, mv = state.storage.load_mv(rec_id, state.cfg.adc)
        lo = min(int(start_s * info.fs), mv.size)
        hi = min(mv.size, lo + int(duration_s * info.fs))
        seg = mv[lo:hi]
        return {"id": rec_id, "source": info.source, "fs": info.fs, "start_s": lo / info.fs,
                "time_s": ((np.arange(seg.size) + lo) / info.fs).round(4).tolist(), "mv": np.round(seg, 4).tolist()}

    @app.get("/recordings/{rec_id}/analysis")
    def recording_analysis(rec_id: str) -> dict:
        _require_recording(rec_id)
        info, mv = state.storage.load_mv(rec_id, state.cfg.adc)
        kind = state.cfg.selected_model
        bundle = state.bundle_for(kind)
        cfg = state.cfg if info.fs == state.cfg.sample_rate_hz else state.cfg.with_updates(sample_rate_hz=info.fs)
        try:
            result = analyze_recording(mv, info.fs, cfg, bundle=bundle, model_kind=kind if bundle else None,
                                       source=info.source)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["recording"] = {"id": info.id, "source": info.source, "notes": info.notes, "units": info.units}
        return result

    @app.get("/recordings/{rec_id}/export.csv")
    def export_recording(rec_id: str) -> Response:
        _require_recording(rec_id)
        info, raw = state.storage.load_raw(rec_id)
        mv = state.cfg.adc.counts_to_mv(raw) if info.units == "adc_counts" else raw.astype(float)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["sample_index", "time_s", "raw_value", "raw_units", "mv", "source"])
        for i, (r, m) in enumerate(zip(raw.tolist(), mv.tolist())):
            writer.writerow([i, f"{i / info.fs:.6f}", r, info.units, f"{m:.6f}", info.source])
        return Response(content=buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{rec_id}.csv"'})

    @app.get("/recordings/{rec_id}/metrics.csv")
    def export_metrics(rec_id: str) -> Response:
        """Derived metrics per RR interval (beat time, RR, instantaneous HR, flags)."""
        _require_recording(rec_id)
        info, mv = state.storage.load_mv(rec_id, state.cfg.adc)
        cfg = state.cfg if info.fs == state.cfg.sample_rate_hz else state.cfg.with_updates(sample_rate_hz=info.fs)
        result = analyze_recording(mv, info.fs, cfg, source=info.source)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["beat_time_s", "rr_s", "hr_bpm", "flags", "source"])
        rr = result["rr"]
        for t, v, f in zip(rr["times_s"], rr["rr_s"], rr["flags"]):
            writer.writerow([t, v, f"{60.0 / v:.2f}", "|".join(f), info.source])
        return Response(content=buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{rec_id}-metrics.csv"'})

    # ----- models and evaluation -----

    @app.get("/models")
    def models() -> dict:
        on_disk = available_models(state.models_dir())
        return {
            "selected": state.cfg.selected_model,
            "available": ["rules"] + [k for k in ("rf", "xgb", "cnn", "hybrid") if k in on_disk],
            "registry": state.storage.list_models(),
            "model_error": state.model_error,
            "supported_classes": ["N", "S", "V", "F"],
            "label_level": "beat-level AAMI classes; not rhythm or recording-level diagnoses",
        }

    @app.get("/evaluation")
    def evaluation() -> dict:
        path = state.models_dir() / "evaluation_summary.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="no evaluation yet; run scripts/train_models.py")
        return {
            **json.loads(path.read_text()),
            "separation_note": "Offline test metrics on held-out MIT-BIH records. They are not live screening output.",
        }

    # ----- configuration -----

    @app.get("/configuration")
    def get_configuration() -> dict:
        return {"config": state.cfg.to_dict(),
                "research_note": "Thresholds are research parameters, not clinical standards."}

    @app.put("/configuration")
    def put_configuration(patch: ConfigPatch) -> dict:
        changes = patch.model_dump(exclude_none=True)
        try:
            new_cfg = _apply_patch(state.cfg, changes)
        except ConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if new_cfg.selected_model != "rules" and state.bundle_for(new_cfg.selected_model) is None:
            raise HTTPException(status_code=422, detail=f"model '{new_cfg.selected_model}' unavailable: {state.model_error}")
        state.cfg = new_cfg
        state.storage.log_config(new_cfg.to_dict())
        applies = "now"
        if state.pipeline is not None:
            if set(changes) <= ANALYSIS_ONLY_KEYS:
                state.pipeline.cfg = new_cfg
                state.attach_model(state.pipeline)
            else:
                applies = "next device connection (acquisition or filter settings changed)"
        return {"config": new_cfg.to_dict(), "applied": True, "applies": applies}

    # ----- device ingest -----

    @app.websocket("/ws/ingest")
    async def ingest(ws: WebSocket) -> None:
        device = state.security.verify_device(_bearer(ws.headers.get("authorization")))
        await ws.accept()

        async def reject(reason: str, detail: str) -> None:
            await ws.send_json({"type": "error", "reason": reason, "detail": detail})
            await ws.close(code=1008)

        try:
            if device is None:
                log.warning("device connection refused: unauthorized")
                await reject("unauthorized", "missing, unknown or revoked device token")
                return
            try:
                hello = parse_hello(await ws.receive_text())
            except PacketError as exc:
                await reject(exc.reason, str(exc))
                return
            if hello.device_id != device.device_id:
                log.warning("device connection refused: device_mismatch")
                await reject("device_mismatch", "hello.device_id does not match the device token")
                return
            if hello.simulated != (device.kind == "simulator"):
                log.warning("device connection refused: source_mismatch")
                await reject("source_mismatch", f"device is registered as {device.kind}")
                return
            if abs(hello.fs - state.cfg.sample_rate_hz) > 1e-9:
                await reject("fs_mismatch", f"configured {state.cfg.sample_rate_hz} Hz")
                return
            # The label comes from the registry, never from the device's own claim.
            source = device.source_label
            state.security.touch_device(device.device_id)
            pipe = LivePipeline(state.cfg, source=source)
            pipe.receiver.on_hello(hello)
            state.attach_model(pipe)
            state.pipeline = pipe
            log.info("device connected source=%s", source)
            await ws.send_json({"type": "ack", "source": source})
            # Events are checked every EVENT_LOG_EVERY_S of SIGNAL time (samples received),
            # so the log reflects how much data was analysed, not delivery speed.
            log_every = int(EVENT_LOG_EVERY_S * pipe.cfg.sample_rate_hz)
            next_log = log_every
            while True:
                pipe.ingest_text(await ws.receive_text())
                if pipe.receiver.total_samples >= next_log:
                    next_log = pipe.receiver.total_samples + log_every
                    state.maybe_log_event(pipe.snapshot(recent_waveform_s=0.1))
                    state.security.touch_device(device.device_id)
        except WebSocketDisconnect:
            log.info("device disconnected")
        except Exception:  # keep the server alive; details go to the log only
            log.exception("ingest stream failed")

    if WEB_DIST.exists():
        # Built dashboard (npm run build in web/). Served at /ui so API paths stay at the root.
        from fastapi.staticfiles import StaticFiles

        app.mount("/ui", StaticFiles(directory=WEB_DIST, html=True), name="ui")

    return app


def _saliency_payload(model: dict) -> dict | None:
    exp = model.get("explanation") or {}
    win = exp.get("saliency_window")
    if not win:
        return None
    fs = win["fs"]
    return {
        "t": [round(i / fs, 4) for i in range(len(win["mv"]))],
        "mv": win["mv"],
        "weight": win["weight"],
        "predicted": exp["predicted_class"],
        "note": exp["note"],
    }


def _restore_saved_settings(state: AppState) -> None:
    """Apply the last settings saved from the dashboard, keeping deployment fields.

    The data directory, model directory and network binding always come from the
    config the server was started with, never from the database.
    """
    saved = state.storage.latest_config()
    if not saved:
        return
    keep = {k: getattr(state.cfg, k) for k in ("data_dir", "models_dir", "api_host", "api_port", "security")}
    try:
        restored = AppConfig.from_dict({**saved, **keep})
    except (ConfigError, TypeError) as exc:
        log.warning("saved settings ignored: %s", exc)
        return
    if restored.selected_model != "rules" and state.bundle_for(restored.selected_model) is None:
        restored = restored.with_updates(selected_model="rules")
    state.cfg = restored


def _apply_patch(cfg: AppConfig, changes: dict[str, Any]) -> AppConfig:
    """Apply a validated partial update and return a new, fully validated config."""
    new = cfg
    if changes.get("preprocess"):
        pre = dict(changes["preprocess"])
        if pre.get("powerline_hz") == 0:
            pre["powerline_hz"] = None  # 0 in the API means "notch off"
        new = new.with_updates(preprocess=replace(new.preprocess, **pre))
    if changes.get("screening"):
        new = new.with_updates(screening=replace(new.screening, **changes["screening"]))
    if changes.get("quality"):
        new = new.with_updates(quality=replace(new.quality, **changes["quality"]))
    top = {k: v for k, v in changes.items() if k not in ("preprocess", "screening", "quality")}
    if top:
        new = new.with_updates(**top)
    return new
