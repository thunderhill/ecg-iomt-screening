import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from _auth import signed_in
from ecg.api.app import create_app
from ecg.config import AppConfig, PreprocessConfig, RPeakConfig
from ecg.ingest import PacketError, StreamReceiver, parse_hello, parse_packet
from ecg.pipeline import LivePipeline
from ecg.simulator import Faults, apply_faults, build_packets, hello_message, simulated_counts
from ecg.synthetic import generate_ecg


def _msg(seq, samples, fs=250.0, device="dev1", t_us=0, lo_plus=False, lo_minus=False):
    return json.dumps(
        {
            "v": 1, "type": "samples", "device_id": device, "seq": seq, "t_us": t_us, "fs": fs,
            "lo_plus": lo_plus, "lo_minus": lo_minus, "samples": samples,
        }
    )


# ---------- parsing ----------

def test_parse_valid_packet():
    p = parse_packet(_msg(1, [1, 2, 3]), expected_fs=250.0)
    assert p.seq == 1 and p.samples.tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[1,2,3]",
        json.dumps({"v": 2, "type": "samples"}),
        _msg(1, [1.5]),  # float sample
        _msg(1, [True]),  # bool is not an ADC count
        _msg(1, [5000]),  # above 12-bit range
        _msg(1, [-1]),
        _msg(-1, [1]),
    ],
)
def test_malformed_packets_rejected(text):
    with pytest.raises(PacketError) as exc:
        parse_packet(text, expected_fs=250.0)
    assert exc.value.reason == "malformed"


def test_missing_field_rejected():
    obj = json.loads(_msg(1, [1]))
    del obj["seq"]
    with pytest.raises(PacketError, match="seq"):
        parse_packet(json.dumps(obj))


def test_fs_mismatch_rejected():
    with pytest.raises(PacketError) as exc:
        parse_packet(_msg(1, [1], fs=500.0), expected_fs=250.0)
    assert exc.value.reason == "fs_mismatch"


def test_hello_requires_simulated_flag():
    obj = {"v": 1, "type": "hello", "device_id": "d", "fs": 250}
    with pytest.raises(PacketError):
        parse_hello(json.dumps(obj))


def test_hello_parses_simulated_flag():
    h = parse_hello(hello_message("sim", 250.0, 10))
    assert h.simulated is True and h.device_id == "sim"


# ---------- integrity tracking ----------

def test_duplicates_and_gaps_counted():
    rx = StreamReceiver(fs=250.0, max_samples=10_000)
    rx.ingest_text(_msg(0, [1] * 10))
    rx.ingest_text(_msg(0, [1] * 10))  # duplicate
    rx.ingest_text(_msg(3, [1] * 10))  # seq 1 and 2 missing
    c = rx.counters
    assert c.duplicates == 1
    assert c.lost_packets == 2
    assert c.lost_samples_estimated == 20
    assert c.accepted == 2


def test_late_packet_is_not_double_counted():
    rx = StreamReceiver(fs=250.0, max_samples=10_000)
    rx.ingest_text(_msg(0, [1] * 10))
    rx.ingest_text(_msg(2, [1] * 10))
    rx.ingest_text(_msg(1, [1] * 10))  # arrives late
    assert rx.counters.out_of_order == 1
    assert rx.total_samples == 20  # the late samples are not placed out of order


def test_device_restart_detected():
    rx = StreamReceiver(fs=250.0, max_samples=10_000)
    rx.ingest_text(_msg(5000, [1] * 10))
    rx.ingest_text(_msg(0, [1] * 10))
    assert rx.counters.session_restarts == 1
    assert rx.counters.accepted == 2


def test_keepalive_advances_no_samples():
    rx = StreamReceiver(fs=250.0, max_samples=10_000)
    rx.ingest_text(_msg(0, [1] * 10))
    rx.ingest_text(_msg(1, []))
    assert rx.total_samples == 10


def test_buffer_is_bounded():
    rx = StreamReceiver(fs=250.0, max_samples=50)
    for seq in range(20):
        rx.ingest_text(_msg(seq, list(range(10))))
    assert rx.latest_counts().size == 50
    assert rx.buffer_start_index == rx.total_samples - 50


def test_lead_off_flags_reported():
    rx = StreamReceiver(fs=250.0, max_samples=100)
    rx.ingest_text(_msg(0, [1], lo_plus=True))
    assert rx.lead_off == (True, False)


def test_malformed_message_counted_not_raised():
    rx = StreamReceiver(fs=250.0, max_samples=100)
    assert rx.ingest_text("garbage") is False
    assert rx.counters.malformed == 1


# ---------- simulator ----------

def test_build_packets_cover_all_samples_in_order():
    counts = np.arange(95)
    msgs = list(build_packets(counts, device_id="d", fs=250.0, batch_size=10))
    assert len(msgs) == 10
    joined = np.concatenate([m["samples"] for m in msgs])
    np.testing.assert_array_equal(joined, counts)
    assert msgs[1]["t_us"] == round(10 * 1e6 / 250.0)


def test_faults_drop_and_duplicate_are_applied():
    msgs = [{"seq": i} for i in range(200)]
    rng = np.random.default_rng(0)
    out = apply_faults(msgs, Faults(drop=0.2, duplicate=0.2), rng)
    seqs = [m["seq"] for m in out]
    assert len(out) != len(msgs)
    assert len(set(seqs)) < len(seqs) or len(set(seqs)) < len(msgs)


def test_invalid_fault_probability_rejected():
    with pytest.raises(ValueError):
        Faults(drop=1.5).validate()


def test_simulated_counts_are_valid_adc_range():
    counts = simulated_counts(5.0, 250.0, AppConfig().adc, seed=1)
    assert counts.min() >= 0 and counts.max() <= 4095


# ---------- pipeline ----------

def _stream_into(pipe: LivePipeline, counts: np.ndarray, batch: int = 10, drop_seqs=()):
    for msg in build_packets(counts, device_id="d", fs=pipe.cfg.sample_rate_hz, batch_size=batch):
        if msg["seq"] in drop_seqs:
            continue
        pipe.ingest_text(json.dumps(msg))


def test_pipeline_detects_heart_rate_on_simulated_stream():
    cfg = AppConfig(data_dir="unused")
    pipe = LivePipeline(cfg, source="SIMULATED")
    rec = generate_ecg(40.0, fs=250.0, hr_bpm=75.0, hr_std_bpm=1.0, seed=3)
    counts = cfg.adc.mv_to_counts(rec.signal)
    _stream_into(pipe, counts)
    snap = pipe.snapshot()
    assert snap["source"] == "SIMULATED"
    assert snap["heart_rate"]["mean_bpm"] is not None
    assert abs(snap["heart_rate"]["mean_bpm"] - 75.0) < 8.0
    assert snap["quality"]["status"] in ("good", "fair")


def test_gap_keeps_history_and_flags_spanning_interval():
    cfg = AppConfig(data_dir="unused")
    pipe = LivePipeline(cfg, source="SIMULATED")
    rec = generate_ecg(60.0, fs=250.0, hr_bpm=70.0, hr_std_bpm=1.0, seed=4)
    counts = cfg.adc.mv_to_counts(rec.signal)
    _stream_into(pipe, counts, drop_seqs={750, 751, 752})  # 1.2 s missing at t = 30 s
    snap = pipe.snapshot()
    assert snap["packet_stats"]["lost_packets"] == 3
    assert snap["gaps"] == 1 and snap["analysis_restarts"] == 0
    # The device timeline includes the missing samples, so it matches the recording length.
    assert snap["timeline_s"] == pytest.approx(60.0, abs=0.05)
    # History from before the gap is kept: far more valid intervals than the 30 s after it.
    assert snap["heart_rate"]["valid_rr_count"] > 50
    assert abs(snap["heart_rate"]["mean_bpm"] - 70.0) < 5.0
    assert snap["heart_rate"]["flagged_rr_count"] >= 1  # the interval across the gap
    # The display and saved recordings only contain the contiguous segment after the gap.
    assert snap["segment_s"] < 31.0
    assert pipe.buffered_counts().size == int(round(snap["segment_s"] * 250))


def test_device_restart_resets_timeline():
    cfg = AppConfig(data_dir="unused")
    pipe = LivePipeline(cfg, source="SIMULATED")
    counts = cfg.adc.mv_to_counts(generate_ecg(20.0, fs=250.0, seed=5).signal)
    msgs = list(build_packets(counts, device_id="d", fs=250.0, batch_size=10, start_seq=5000))
    for m in msgs:
        pipe.ingest_text(json.dumps(m))
    for m in build_packets(counts[:500], device_id="d", fs=250.0, batch_size=10, start_seq=0):
        pipe.ingest_text(json.dumps(m))
    snap = pipe.snapshot()
    assert snap["analysis_restarts"] == 1
    assert snap["heart_rate"]["valid_rr_count"] == 0  # old timeline's beats are gone


def test_detected_r_peaks_align_with_displayed_signal():
    """Regression: causal detection once timestamped beats late (internal filter delay)."""
    cfg = AppConfig(data_dir="unused")
    pipe = LivePipeline(cfg, source="SIMULATED")
    rec = generate_ecg(20.0, fs=250.0, noise_std_mv=0.005, seed=6)
    _stream_into(pipe, cfg.adc.mv_to_counts(rec.signal))
    wf = pipe.snapshot()["waveform"]
    y = np.asarray(wf["filtered_mv"])
    offsets = [int(np.argmax(y[o - 10 : o + 11])) - 10 for o in wf["r_peak_offsets"] if 10 <= o < y.size - 11]
    assert offsets and max(abs(o) for o in offsets) <= 1


def test_pipeline_never_labels_sim_as_live():
    pipe = LivePipeline(AppConfig(data_dir="unused"), source="SIMULATED")
    assert pipe.snapshot()["source"] == "SIMULATED"


def test_pipeline_rejects_unknown_source():
    with pytest.raises(ValueError):
        LivePipeline(AppConfig(data_dir="unused"), source="MAYBE")


def test_insufficient_data_before_enough_beats():
    cfg = AppConfig(data_dir="unused")
    pipe = LivePipeline(cfg, source="SIMULATED")
    rec = generate_ecg(3.0, fs=250.0, seed=5)
    _stream_into(pipe, cfg.adc.mv_to_counts(rec.signal))
    assert pipe.snapshot()["screening"]["category"] in ("insufficient_data", "insufficient_signal_quality", "no_configured_flag")


# ---------- API ----------

@pytest.fixture()
def client(tmp_path):
    cfg = AppConfig(data_dir=str(tmp_path))
    return signed_in(cfg)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_endpoints_before_any_device(client):
    assert client.get("/status").json()["mode"] == "NONE"
    assert client.get("/ecg/latest").status_code == 404


def test_websocket_stream_is_labelled_simulated(client):
    cfg = AppConfig()
    rec = generate_ecg(15.0, fs=250.0, seed=6)
    counts = cfg.adc.mv_to_counts(rec.signal)
    with client.websocket_connect("/ws/ingest", headers=client.device_headers("ecg-sim-01")) as ws:
        ws.send_text(hello_message("ecg-sim-01", 250.0, 10))
        assert json.loads(ws.receive_text())["source"] == "SIMULATED"
        for msg in build_packets(counts, device_id="ecg-sim-01", fs=250.0, batch_size=10):
            ws.send_text(json.dumps(msg))
    status = client.get("/status").json()
    assert status["mode"] == "SIMULATED"
    assert status["packet_stats"]["accepted"] > 0
    assert client.get("/screening/latest").json()["source"] == "SIMULATED"


def test_websocket_rejects_wrong_sample_rate(client):
    with client.websocket_connect("/ws/ingest", headers=client.device_headers("d")) as ws:
        ws.send_text(hello_message("d", 500.0, 10))
        reply = json.loads(ws.receive_text())
        assert reply["type"] == "error" and reply["reason"] == "fs_mismatch"


def test_configuration_update_validated(client):
    r = client.put("/configuration", json={"preprocess": {"bandpass_high_hz": 200.0}})
    assert r.status_code == 422
    ok = client.put("/configuration", json={"screening": {"tachy_bpm": 110.0}})
    assert ok.status_code == 200
    assert ok.json()["config"]["screening"]["tachy_bpm"] == 110.0


def test_configuration_rejects_unknown_field(client):
    r = client.put("/configuration", json={"nonsense": 1})
    assert r.status_code == 200  # extra fields are ignored by the Pydantic model
    assert "nonsense" not in r.json()["config"]


def test_recording_save_and_export(client, tmp_path):
    cfg = AppConfig()
    rec = generate_ecg(12.0, fs=250.0, seed=7)
    counts = cfg.adc.mv_to_counts(rec.signal)
    with client.websocket_connect("/ws/ingest", headers=client.device_headers("d")) as ws:
        ws.send_text(hello_message("d", 250.0, 10))
        ws.receive_text()
        for msg in build_packets(counts, device_id="d", fs=250.0, batch_size=10):
            ws.send_text(json.dumps(msg))
    saved = client.post("/recordings", json={"notes": "test"})
    assert saved.status_code == 201
    rec_id = saved.json()["id"]
    listing = client.get("/recordings").json()["items"]
    assert any(item["id"] == rec_id and item["source"] == "SIMULATED" for item in listing)
    csv_text = client.get(f"/recordings/{rec_id}/export.csv").text
    assert csv_text.splitlines()[0] == "sample_index,time_s,raw_value,raw_units,mv,source"
    assert "SIMULATED" in csv_text.splitlines()[1]


def test_unknown_recording_is_404(client):
    assert client.get("/recordings/does-not-exist/waveform").status_code == 404


# ---------- offline analysis, models, explanations ----------

from pathlib import Path  # noqa: E402

REPO_MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture()
def model_client(tmp_path):
    cfg = AppConfig(data_dir=str(tmp_path), models_dir=str(REPO_MODELS))
    return signed_in(cfg), cfg


def _stream(client, seconds=40.0, seed=8):
    cfg = AppConfig()
    rec = generate_ecg(seconds, fs=250.0, seed=seed)
    with client.websocket_connect("/ws/ingest", headers=client.device_headers("d")) as ws:
        ws.send_text(hello_message("d", 250.0, 10))
        ws.receive_text()
        for msg in build_packets(cfg.adc.mv_to_counts(rec.signal), device_id="d", fs=250.0, batch_size=10):
            ws.send_text(json.dumps(msg))


def test_offline_analysis_of_mv_recording(model_client):
    client, cfg = model_client
    storage = client.app.state.ecg.storage
    rec = generate_ecg(90.0, fs=250.0, hr_bpm=65, seed=12)
    info = storage.save_recording(rec.signal, source="OFFLINE", fs=250.0, device_id=None,
                                  config=cfg.to_dict(), units="mv")
    res = client.get(f"/recordings/{info.id}/analysis").json()
    assert res["source"] == "OFFLINE"
    assert abs(res["heart_rate"]["mean_bpm"] - 65) < 5
    assert res["hrv"]["available"] is True
    metrics_csv = client.get(f"/recordings/{info.id}/metrics.csv").text
    assert metrics_csv.splitlines()[0] == "beat_time_s,rr_s,hr_bpm,flags,source"


def test_rules_only_model_reports_unavailable(model_client):
    client, _ = model_client
    _stream(client)
    live = client.get("/live").json()
    assert live["model"]["available"] is False
    assert any("No trained model" in w for w in live["model"]["warnings"])


@pytest.mark.skipif(not (REPO_MODELS / "xgb.joblib").exists(), reason="models not trained")
def test_select_model_and_explain(model_client):
    client, _ = model_client
    r = client.put("/configuration", json={"selected_model": "xgb"})
    assert r.status_code == 200
    _stream(client, seconds=70.0)
    exp = client.get("/explain/latest").json()
    assert exp["model"]["available"] is True
    assert exp["model"]["explanation"]["contributions"]
    assert exp["global_importance"]["permutation"]
    assert any("SIMULATED" in w for w in exp["model"]["warnings"])
    assert client.get("/screening/history").json()["items"]


def test_selecting_missing_model_is_rejected(tmp_path):
    cfg = AppConfig(data_dir=str(tmp_path), models_dir=str(tmp_path / "none"))
    client = signed_in(cfg)
    r = client.put("/configuration", json={"selected_model": "rf"})
    assert r.status_code == 422


@pytest.mark.skipif(not (REPO_MODELS / "evaluation_summary.json").exists(), reason="models not trained")
def test_evaluation_endpoint_separates_offline_metrics(model_client):
    client, _ = model_client
    ev = client.get("/evaluation").json()
    assert "not live screening output" in ev["separation_note"]
    assert ev["models"]["rf"]["test"]["per_class"]["V"]["support"] > 0


def test_models_endpoint_lists_rules(model_client):
    client, _ = model_client
    assert "rules" in client.get("/models").json()["available"]


def test_storage_migrates_old_schema(tmp_path):
    import sqlite3

    from ecg.storage import Storage

    con = sqlite3.connect(tmp_path / "ecg.sqlite3")
    con.execute("CREATE TABLE recordings (id TEXT PRIMARY KEY, created_at REAL NOT NULL, source TEXT NOT NULL, "
                "device_id TEXT, fs REAL NOT NULL, n_samples INTEGER NOT NULL, waveform_path TEXT NOT NULL, "
                "config_json TEXT NOT NULL, model_version TEXT, notes TEXT)")
    con.commit()
    con.close()
    storage = Storage(tmp_path)
    info = storage.save_recording(np.ones(10), source="OFFLINE", fs=250.0, device_id=None, config={}, units="mv")
    assert storage.get_recording(info.id).units == "mv"


def test_config_round_trips_through_dict():
    cfg = AppConfig().with_updates(selected_model="rules", retention_days=7)
    assert AppConfig.from_dict(cfg.to_dict()) == cfg


def test_settings_survive_restart(tmp_path):
    cfg = AppConfig(data_dir=str(tmp_path), models_dir=str(tmp_path / "none"))
    first = signed_in(cfg)
    assert first.put("/configuration", json={"screening": {"tachy_bpm": 115.0}}).status_code == 200
    second = signed_in(cfg, admin_token=first.admin_token)
    assert second.get("/configuration").json()["config"]["screening"]["tachy_bpm"] == 115.0
    # Deployment fields come from the start-up config, not the database.
    assert second.get("/configuration").json()["config"]["data_dir"] == str(tmp_path)
