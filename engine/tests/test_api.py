"""HTTP surface: health, catalog, and the analyze contract.

The analyze tests deliberately point the engine at an empty model directory, so
they assert the *degraded* path — which is the one that must never break. An
engine with no weights still has to return a schema-valid result carrying a
retry, because "refusing to judge is a first-class outcome" (docs/architecture.md §1).
"""

import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from openschwa_engine.config import Settings
from openschwa_engine.schemas.analysis import AnalysisResult
from openschwa_engine.server import create_app

EXERCISE_ID = "en.seg.dh-z.this"


@pytest.fixture
def client(tmp_path):
    """An engine whose model cache is empty, whatever the developer has downloaded."""
    return TestClient(create_app(Settings(model_dir=tmp_path / "models")))


def recording(duration_s: float = 1.2, rate: int = 48_000, amp: float = 0.4) -> bytes:
    t = np.arange(int(rate * duration_s)) / rate
    signal = np.random.RandomState(0).normal(0, 0.0015, t.shape)
    speaking = (t > 0.2) & (t < duration_s - 0.2)
    for harmonic in range(6):
        signal[speaking] += (
            amp / (harmonic + 1) * np.sin(2 * np.pi * 140.0 * (harmonic + 1) * t[speaking])
        )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def post_analysis(client: TestClient, exercise_id: str = EXERCISE_ID, wav: bytes | None = None):
    return client.post(
        "/v1/analyze",
        files={"audio": ("recording.wav", wav if wav is not None else recording(), "audio/wav")},
        data={"exercise_id": exercise_id},
    )


# -- health + catalog ---------------------------------------------------------------


def test_health_reports_versions_and_analysis_readiness(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["schema_version"] == "1.0"
    assert body["alignment_model"] == "wav2vec2-espeak-cv-ft"
    assert body["analysis_available"] is False  # empty model dir


def test_models_are_pinned_to_a_commit_not_a_branch(client):
    """An upstream retrain must not silently change what learners are scored against."""
    for model in client.get("/v1/models").json()["models"]:
        assert len(model["revision"]) == 40, f"{model['id']} is not pinned to a commit sha"
        assert model["state"] == "missing"


def test_lists_exercises(client):
    body = client.get("/v1/exercises").json()
    ids = [e["id"] for e in body["exercises"]]
    assert EXERCISE_ID in ids
    assert next(e for e in body["exercises"] if e["id"] == EXERCISE_ID)["focus_phone"] == "ð"


def test_exercise_detail_carries_what_the_ui_needs_before_recording(client):
    body = client.get(f"/v1/exercises/{EXERCISE_ID}").json()
    assert body["text"] == "this"
    assert body["ipa"] == "ðɪs"
    assert [p["ph"] for p in body["phones"]] == ["ð", "ɪ", "s"]
    assert body["phones"][0]["focus"] is True
    assert body["has_reference_audio"] is False


def test_unknown_exercise_is_a_404(client):
    assert client.get("/v1/exercises/nope").status_code == 404
    assert client.get("/v1/exercises/nope/reference-audio").status_code == 404


def test_reference_audio_404s_until_it_is_recorded(client):
    assert client.get(f"/v1/exercises/{EXERCISE_ID}/reference-audio").status_code == 404


# -- analyze ------------------------------------------------------------------------


def test_analysis_without_a_model_is_a_valid_retry_not_an_error(client):
    response = post_analysis(client)
    assert response.status_code == 200

    result = AnalysisResult.model_validate(response.json())
    assert result.alignment.status == "failed"
    assert result.alignment.phones == []
    assert [item.kind for item in result.feedback] == ["retry"]


def test_audio_measurement_works_without_a_model(client):
    """Duration, speech interval, quality, and F0 are engine-side arithmetic —
    they must not depend on the acoustic model being present."""
    body = post_analysis(client).json()
    assert body["audio"]["duration_s"] == pytest.approx(1.2, abs=0.02)
    assert body["audio"]["sample_rate"] == 48_000
    start, end = body["audio"]["speech_interval_s"]
    assert start == pytest.approx(0.2, abs=0.1) and end == pytest.approx(1.0, abs=0.1)
    assert body["prosody"]["f0"]["median_hz"] == pytest.approx(140.0, abs=3.0)


def test_m0_emits_no_ungated_judgements(client):
    """No feedback type has cleared the shipping bar (`eval/README.md`) yet, so the engine
    reports evidence and stays silent on verdicts."""
    body = post_analysis(client).json()
    assert body["contrasts"] == []
    assert body["annotations"] == []
    assert all(item["kind"] == "retry" for item in body["feedback"])


def test_a_quiet_recording_is_still_analysed(client):
    """Level is reported for advice; it must not become a refusal."""
    body = post_analysis(client, wav=recording(amp=0.005)).json()
    quality = body["audio"]["quality"]
    assert quality["too_quiet"] is False
    assert quality["speech_level_dbfs"] < -40  # low enough to have tripped the old gate
    assert body["audio"]["speech_interval_s"] is not None


def test_silence_produces_a_specific_retry_message(client):
    silence = np.random.RandomState(2).normal(0, 0.0005, 48_000).astype(np.float32)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes((silence * 32767).astype("<i2").tobytes())

    body = post_analysis(client, wav=buffer.getvalue()).json()
    assert body["audio"]["speech_interval_s"] is None
    assert "couldn't hear any speech" in body["feedback"][0]["message"]


def test_response_matches_the_committed_json_schema(client):
    """The UI's types are generated from this file; drifting from it breaks them."""
    import jsonschema

    schema_path = Settings().content_schema_path.parents[2] / "schemas"
    schema = json.loads((schema_path / "analysis_result.v1.schema.json").read_text())
    jsonschema.validate(post_analysis(client).json(), schema)


@pytest.mark.parametrize(
    ("exercise_id", "payload", "expected"),
    [
        ("nope", None, 404),
        (EXERCISE_ID, b"not a wav at all", 400),
    ],
)
def test_client_errors(client, exercise_id, payload, expected):
    assert post_analysis(client, exercise_id, payload).status_code == expected


def test_rejects_an_over_long_recording(client):
    assert post_analysis(client, wav=recording(duration_s=31.0, rate=16_000)).status_code == 400
