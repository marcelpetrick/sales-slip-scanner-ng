"""Offline tests for the local hot-folder receipt workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import receipt_ocr
import salesSlipScanner as scanner


def make_image(path: Path, color=(200, 100, 50)) -> Path:
    """Create a small valid receipt-like image fixture."""
    Image.new("RGB", (100, 80), color=color).save(path)
    return path


def model_list(*names: str) -> MagicMock:
    models = [SimpleNamespace(model=name) for name in names]
    return MagicMock(return_value=SimpleNamespace(models=models))


def chat_response(content: str) -> MagicMock:
    return MagicMock(
        return_value=SimpleNamespace(message=SimpleNamespace(content=content))
    )


def receipt(file_name: str, digest: str, cents: int) -> dict:
    return {
        "file": file_name,
        "sha256": digest,
        "price_cents": cents,
        "model": scanner.DEFAULT_MODEL,
        "response": f"{cents // 100},{cents % 100:02d}",
        "processed_at": "2026-07-13T12:00:00+00:00",
    }


def test_default_is_measured_local_model():
    assert scanner.DEFAULT_MODEL == "qwen3.5:4b"


def test_list_local_models_returns_exact_tags():
    with patch.object(scanner.ollama, "list", model_list("qwen3.5:4b", "gemma3:4b")):
        assert scanner.list_local_models() == ["qwen3.5:4b", "gemma3:4b"]


def test_unreachable_ollama_becomes_clear_runtime_error():
    with patch.object(scanner.ollama, "list", side_effect=ConnectionError("offline")):
        with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
            scanner.list_local_models()


def test_missing_model_exits_with_pull_command(capsys):
    with patch.object(scanner.ollama, "list", model_list("other:latest")):
        with pytest.raises(SystemExit):
            scanner.ensure_model_available(scanner.DEFAULT_MODEL)
    assert f"ollama pull {scanner.DEFAULT_MODEL}" in capsys.readouterr().err


def test_warm_model_uses_requested_keep_alive():
    with patch.object(scanner.ollama, "generate") as generate:
        scanner.warm_model("model", "15m")
    assert generate.call_args.kwargs["keep_alive"] == "15m"
    assert generate.call_args.kwargs["options"]["num_ctx"] == 4096


def test_warm_failure_has_model_context():
    with patch.object(scanner.ollama, "generate", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="Could not warm.*model"):
            scanner.warm_model("model", "10m")


def test_query_model_uses_deterministic_warm_options():
    chat = chat_response(" 79,49 ")
    with patch.object(receipt_ocr.ollama, "chat", chat):
        assert receipt_ocr.query_model("model", "image", "12m") == "79,49"
    options = chat.call_args.kwargs
    assert options["think"] is False
    assert options["keep_alive"] == "12m"
    assert options["options"] == {
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 64,
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("79,49", 7949), ("10.93", 1093), ("€28,41", None), ("10,00 20,00", None)],
)
def test_strict_price_parsing(raw, expected):
    assert receipt_ocr.parse_price(raw) == expected


def test_image_encoding_resizes_large_input(tmp_path):
    image = tmp_path / "large.png"
    Image.new("RGB", (3000, 1500), color="white").save(image)
    assert isinstance(receipt_ocr.encode_image(image), str)


def test_file_hash_changes_with_content(tmp_path):
    path = tmp_path / "receipt.jpg"
    path.write_bytes(b"first")
    first = scanner.file_sha256(path)
    path.write_bytes(b"second")
    assert scanner.file_sha256(path) != first


def test_missing_state_is_empty(tmp_path):
    assert scanner.load_state(tmp_path) == scanner.empty_state()


def test_state_round_trip_is_atomic(tmp_path):
    state = scanner.empty_state()
    state["receipts"].append(receipt("a.jpg", "a" * 64, 1093))
    scanner.save_state(tmp_path, state)
    assert scanner.load_state(tmp_path) == state
    assert not list(tmp_path.glob(f"{scanner.STATE_NAME}.*"))


def test_obsolete_rename_manifest_is_rejected(tmp_path):
    (tmp_path / scanner.STATE_NAME).write_text(
        json.dumps({"version": 1, "receipts": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="obsolete"):
        scanner.load_state(tmp_path)


def test_collect_pending_filters_extensions_and_content_duplicates(tmp_path):
    first = make_image(tmp_path / "b.jpg")
    duplicate = tmp_path / "a.jpg"
    duplicate.write_bytes(first.read_bytes())
    make_image(tmp_path / "c.png", color=(1, 2, 3))
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    pending = scanner.collect_pending(tmp_path, set())
    assert [path.name for path, _ in pending] == ["a.jpg", "c.png"]


def test_collect_pending_skips_successful_fingerprint(tmp_path):
    image = make_image(tmp_path / "receipt.jpg")
    assert scanner.collect_pending(tmp_path, {scanner.file_sha256(image)}) == []


def test_markdown_report_contains_receipts_total_and_failures():
    state = scanner.empty_state()
    state["receipts"] = [
        receipt("shop|one.jpg", "a" * 64, 7949),
        receipt("fuel.jpg", "b" * 64, 2841),
    ]
    state["last_failures"] = [{"file": "bad.jpg", "error": "not readable"}]
    with patch.object(scanner, "utc_now", return_value="NOW"):
        report = scanner.render_report(state, scanner.DEFAULT_MODEL)
    assert "shop\\|one.jpg" in report
    assert "79,49 €" in report
    assert "## Grand total: 107,90 €" in report
    assert "bad.jpg" in report
    assert scanner.DEFAULT_MODEL in report


def test_empty_markdown_report_has_zero_total():
    with patch.object(scanner, "utc_now", return_value="NOW"):
        report = scanner.render_report(scanner.empty_state(), scanner.DEFAULT_MODEL)
    assert "No successful receipts" in report
    assert "Grand total: 0,00 €" in report


def test_process_file_success_does_not_mutate_source(tmp_path):
    image = make_image(tmp_path / "receipt.jpg")
    original = image.read_bytes()
    with (
        patch.object(scanner, "encode_image", return_value="encoded"),
        patch.object(scanner, "query_model", return_value="28,41") as query,
        patch.object(scanner, "utc_now", return_value="NOW"),
    ):
        result = scanner.process_file(image, "a" * 64, "model", "10m")
    assert result["price_cents"] == 2841
    assert image.read_bytes() == original
    query.assert_called_once_with("model", "encoded", keep_alive="10m")


def test_process_file_rejects_unparseable_response(tmp_path):
    image = make_image(tmp_path / "receipt.jpg")
    with (
        patch.object(scanner, "encode_image", return_value="encoded"),
        patch.object(scanner, "query_model", return_value="EUR 28,41"),
    ):
        result = scanner.process_file(image, "a" * 64, "model", "10m")
    assert "unparseable" in result["error"]
    assert image.exists()


def test_process_file_captures_inference_error(tmp_path):
    image = make_image(tmp_path / "receipt.jpg")
    with patch.object(scanner, "encode_image", side_effect=OSError("broken")):
        result = scanner.process_file(image, "a" * 64, "model", "10m")
    assert result["error"] == "OSError: broken"


def test_empty_hot_folder_is_created_with_report_and_state(tmp_path):
    hot = tmp_path / "new-hot-folder"
    with patch.object(scanner, "ensure_model_available"):
        result = scanner.run(input_dir=hot)
    assert result["receipt_count"] == 0
    assert (hot / scanner.REPORT_NAME).exists()
    assert (hot / scanner.STATE_NAME).exists()


def test_run_processes_sequentially_and_accumulates_total(tmp_path):
    first = make_image(tmp_path / "a.jpg")
    second = make_image(tmp_path / "b.jpg", color=(1, 2, 3))
    first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
    responses = iter(["79,49", "28,41"])
    with (
        patch.object(scanner, "ensure_model_available"),
        patch.object(scanner, "warm_model") as warm,
        patch.object(
            scanner, "query_model", side_effect=lambda *_args, **_kw: next(responses)
        ),
    ):
        result = scanner.run(input_dir=tmp_path, keep_alive="20m")
    assert result["processed"] == 2
    assert result["total_cents"] == 10790
    assert first.read_bytes() == first_bytes
    assert second.read_bytes() == second_bytes
    assert len(scanner.load_state(tmp_path)["receipts"]) == 2
    assert "Grand total: 107,90 €" in (tmp_path / scanner.REPORT_NAME).read_text()
    warm.assert_called_once_with(scanner.DEFAULT_MODEL, "20m")


def test_successful_content_is_not_processed_twice(tmp_path):
    make_image(tmp_path / "receipt.jpg")
    with (
        patch.object(scanner, "ensure_model_available"),
        patch.object(scanner, "warm_model"),
        patch.object(scanner, "query_model", return_value="10,93") as query,
    ):
        first = scanner.run(input_dir=tmp_path)
        second = scanner.run(input_dir=tmp_path)
    assert first["processed"] == 1
    assert second["processed"] == 0
    assert second["total_cents"] == 1093
    assert query.call_count == 1


def test_failed_receipt_is_retried_next_run(tmp_path):
    make_image(tmp_path / "receipt.jpg")
    with (
        patch.object(scanner, "ensure_model_available"),
        patch.object(scanner, "warm_model"),
        patch.object(scanner, "query_model", side_effect=["NaN", "10,93"]),
    ):
        failed = scanner.run(input_dir=tmp_path)
        succeeded = scanner.run(input_dir=tmp_path)
    assert failed["failed"] == 1
    assert succeeded["processed"] == 1
    assert scanner.load_state(tmp_path)["last_failures"] == []


def test_cli_defaults_to_hot_folder_and_winner():
    args = scanner.parse_args([])
    assert args.model == scanner.DEFAULT_MODEL
    assert args.hot_folder == scanner.HOT_FOLDER
    assert args.keep_alive == scanner.KEEP_ALIVE


def test_cli_accepts_paths_and_keep_alive(tmp_path):
    args = scanner.parse_args(
        [
            "--hot-folder",
            str(tmp_path),
            "--report",
            str(tmp_path / "report.md"),
            "--keep-alive",
            "30m",
        ]
    )
    assert args.report == tmp_path / "report.md"
    assert args.keep_alive == "30m"


def test_main_returns_nonzero_for_current_failure():
    with patch.object(scanner, "run", return_value={"failed": 1}):
        assert scanner.main([]) == 1


def test_main_returns_zero_for_success():
    with patch.object(scanner, "run", return_value={"failed": 0}):
        assert scanner.main([]) == 0
