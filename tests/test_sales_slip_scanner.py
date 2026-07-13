"""Unit tests for salesSlipScanner.py.

All external I/O (ollama, filesystem rename) is mocked so the suite runs
offline without any model installed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import salesSlipScanner as sss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_jpeg(path: Path, width: int = 100, height: int = 80) -> Path:
    """Create a minimal solid-colour JPEG at *path* and return *path*."""
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    img.save(path, format="JPEG")
    return path


def fake_ollama_list(*model_ids: str):
    """Return a callable that mimics ``ollama.list()`` for the given IDs."""
    models = [SimpleNamespace(model=m) for m in model_ids]
    return MagicMock(return_value=SimpleNamespace(models=models))


def fake_ollama_chat(content: str):
    """Return a callable that mimics ``ollama.chat()`` returning *content*."""
    msg = SimpleNamespace(content=content)
    resp = SimpleNamespace(message=msg)
    return MagicMock(return_value=resp)


# ===========================================================================
# parse_price
# ===========================================================================


class TestParsePrice:
    def test_standard_comma(self):
        assert sss.parse_price("79,49") == 7949

    def test_dot_decimal(self):
        assert sss.parse_price("79.49") == 7949

    def test_euro_prefix(self):
        assert sss.parse_price("€79,49") is None

    def test_euro_suffix_with_space(self):
        assert sss.parse_price("79,49 €") is None

    def test_embedded_in_sentence(self):
        assert sss.parse_price("Summe: 28,41 EUR") is None

    def test_leading_whitespace(self):
        assert sss.parse_price("  10,93  ") == 1093

    def test_one_euro(self):
        assert sss.parse_price("1,00") == 100

    def test_sub_euro(self):
        assert sss.parse_price("0,99") == 99

    def test_three_digit_euros(self):
        assert sss.parse_price("100,00") == 10000

    def test_nan_string(self):
        assert sss.parse_price("NaN") is None

    def test_empty_string(self):
        assert sss.parse_price("") is None

    def test_no_digits(self):
        assert sss.parse_price("no number here") is None

    def test_only_one_decimal_digit(self):
        # "7,4" has only 1 decimal digit — must not match (\d{2} required)
        assert sss.parse_price("7,4") is None

    def test_three_decimal_digits_not_matched(self):
        assert sss.parse_price("7,449") is None

    def test_multiple_amounts_rejected(self):
        assert sss.parse_price("12,00 79,49") is None


# ===========================================================================
# build_renamed_path
# ===========================================================================


class TestBuildRenamedPath:
    def test_jpg(self):
        p = Path("/input/receipt.jpg")
        assert sss.build_renamed_path(p, 7949) == Path("/input/receipt_7949.jpg")

    def test_png(self):
        p = Path("/input/slip.png")
        assert sss.build_renamed_path(p, 2841) == Path("/input/slip_2841.png")

    def test_jpeg_extension_preserved(self):
        p = Path("/input/scan.jpeg")
        assert sss.build_renamed_path(p, 1093) == Path("/input/scan_1093.jpeg")

    def test_zero_cents(self):
        p = Path("/input/free.jpg")
        assert sss.build_renamed_path(p, 0) == Path("/input/free_0.jpg")

    def test_parent_directory_preserved(self):
        p = Path("/some/deep/dir/receipt.jpg")
        result = sss.build_renamed_path(p, 500)
        assert result.parent == Path("/some/deep/dir")

    def test_stem_unchanged(self):
        p = Path("/input/my_receipt.jpg")
        result = sss.build_renamed_path(p, 999)
        assert result.name == "my_receipt_999.jpg"


# ===========================================================================
# manifest
# ===========================================================================


class TestManifest:
    def test_missing_manifest_returns_empty_structure(self, tmp_path):
        assert sss.load_manifest(tmp_path) == {"version": 1, "receipts": []}

    def test_manifest_round_trip(self, tmp_path):
        manifest = {
            "version": 1,
            "receipts": [{"source": "a.jpg", "file": "a_99.jpg"}],
        }
        sss.save_manifest(tmp_path, manifest)
        assert sss.load_manifest(tmp_path) == manifest

    def test_invalid_manifest_is_rejected(self, tmp_path):
        (tmp_path / sss.MANIFEST_NAME).write_text('{"version": 2}', encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid processing manifest"):
            sss.load_manifest(tmp_path)


# ===========================================================================
# collect_input_files
# ===========================================================================


class TestCollectInputFiles:
    def test_empty_directory(self, tmp_path):
        assert sss.collect_input_files(tmp_path) == []

    def test_missing_directory_raises(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            sss.collect_input_files(missing)

    def test_jpg_picked_up(self, tmp_path):
        make_jpeg(tmp_path / "a.jpg")
        files = sss.collect_input_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "a.jpg"

    def test_jpeg_picked_up(self, tmp_path):
        make_jpeg(tmp_path / "b.jpeg")
        assert len(sss.collect_input_files(tmp_path)) == 1

    def test_png_picked_up(self, tmp_path):
        img = Image.new("RGB", (10, 10))
        img.save(tmp_path / "c.png")
        assert len(sss.collect_input_files(tmp_path)) == 1

    def test_unsupported_extensions_ignored(self, tmp_path):
        (tmp_path / "doc.pdf").write_bytes(b"%PDF")
        (tmp_path / "note.txt").write_text("hello")
        assert sss.collect_input_files(tmp_path) == []

    def test_already_processed_excluded(self, tmp_path):
        make_jpeg(tmp_path / "slip_7949.jpg")
        assert sss.collect_input_files(tmp_path, {"slip_7949.jpg"}) == []

    def test_mix_processed_and_unprocessed(self, tmp_path):
        make_jpeg(tmp_path / "new.jpg")
        make_jpeg(tmp_path / "old_7949.jpg")
        files = sss.collect_input_files(tmp_path, {"old_7949.jpg"})
        assert len(files) == 1
        assert files[0].name == "new.jpg"

    def test_numeric_filename_is_not_assumed_processed(self, tmp_path):
        make_jpeg(tmp_path / "receipt_2024.jpg")
        assert sss.collect_input_files(tmp_path) == [tmp_path / "receipt_2024.jpg"]

    def test_sorted_order(self, tmp_path):
        make_jpeg(tmp_path / "c.jpg")
        make_jpeg(tmp_path / "a.jpg")
        make_jpeg(tmp_path / "b.jpg")
        names = [f.name for f in sss.collect_input_files(tmp_path)]
        assert names == ["a.jpg", "b.jpg", "c.jpg"]

    def test_subdirectories_not_traversed(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        make_jpeg(sub / "nested.jpg")
        assert sss.collect_input_files(tmp_path) == []


# ===========================================================================
# load_and_encode_image
# ===========================================================================


class TestLoadAndEncodeImage:
    def test_returns_base64_string(self, tmp_path):
        p = make_jpeg(tmp_path / "img.jpg")
        result = sss.load_and_encode_image(p)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_small_image_not_resized(self, tmp_path):
        p = make_jpeg(tmp_path / "small.jpg", width=100, height=80)
        # Should succeed without error
        result = sss.load_and_encode_image(p)
        assert result

    def test_large_image_encoded(self, tmp_path):
        # Image wider than MAX_SIDE_PX — should still encode (after resize)
        p = make_jpeg(tmp_path / "big.jpg", width=3000, height=2000)
        result = sss.load_and_encode_image(p)
        assert result

    def test_output_is_valid_jpeg_when_decoded(self, tmp_path):
        import base64
        p = make_jpeg(tmp_path / "img.jpg")
        encoded = sss.load_and_encode_image(p)
        raw = base64.b64decode(encoded)
        # JPEG magic bytes
        assert raw[:2] == b"\xff\xd8"


# ===========================================================================
# list_local_models / ensure_model_available
# ===========================================================================


class TestListLocalModels:
    def test_returns_model_ids(self):
        with patch("salesSlipScanner.ollama.list", fake_ollama_list("llava:7b", "moondream")):
            result = sss.list_local_models()
        assert result == ["llava:7b", "moondream"]

    def test_raises_on_connection_error(self):
        with patch("salesSlipScanner.ollama.list", side_effect=Exception("refused")):
            with pytest.raises(RuntimeError, match="Cannot reach ollama"):
                sss.list_local_models()


class TestEnsureModelAvailable:
    def test_exact_match_passes(self):
        with patch("salesSlipScanner.ollama.list", fake_ollama_list("qwen3-vl:4b")):
            sss.ensure_model_available("qwen3-vl:4b")  # must not raise

    def test_different_tag_is_rejected(self):
        with patch("salesSlipScanner.ollama.list", fake_ollama_list("qwen3-vl:4b-fp16")):
            with pytest.raises(SystemExit):
                sss.ensure_model_available("qwen3-vl:4b")

    def test_implicit_latest_tag_passes(self):
        with patch("salesSlipScanner.ollama.list", fake_ollama_list("moondream:latest")):
            sss.ensure_model_available("moondream")

    def test_absent_model_exits(self, capsys):
        with patch("salesSlipScanner.ollama.list", fake_ollama_list("moondream")):
            with pytest.raises(SystemExit) as exc:
                sss.ensure_model_available("llava:7b")
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "ollama pull llava:7b" in err

    def test_no_models_installed(self, capsys):
        with patch("salesSlipScanner.ollama.list", fake_ollama_list()):
            with pytest.raises(SystemExit):
                sss.ensure_model_available("llava:7b")
        err = capsys.readouterr().err
        assert "(none)" in err


# ===========================================================================
# query_ollama
# ===========================================================================


class TestQueryOllama:
    def test_returns_stripped_content(self):
        with patch("salesSlipScanner.ollama.chat", fake_ollama_chat("  79,49  ")):
            result = sss.query_ollama("some-model", "b64data")
        assert result == "79,49"

    def test_uses_benchmarked_deterministic_options(self):
        chat = fake_ollama_chat("79,49")
        with patch("salesSlipScanner.ollama.chat", chat):
            sss.query_ollama("some-model", "b64data")
        assert chat.call_args.kwargs["think"] is False
        assert chat.call_args.kwargs["options"] == {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 64,
        }

    def test_propagates_exception(self):
        with patch("salesSlipScanner.ollama.chat", side_effect=RuntimeError("timeout")):
            with pytest.raises(RuntimeError):
                sss.query_ollama("model", "b64")


# ===========================================================================
# process_file
# ===========================================================================


class TestProcessFile:
    def test_success_renames_file(self, tmp_path):
        img = make_jpeg(tmp_path / "receipt.jpg")
        with patch("salesSlipScanner.load_and_encode_image", return_value="b64"), \
             patch("salesSlipScanner.query_ollama", return_value="79,49"):
            price = sss.process_file(img, "model")
        assert price == 7949
        assert not img.exists()
        assert (tmp_path / "receipt_7949.jpg").exists()

    def test_nan_response_skips_rename(self, tmp_path):
        img = make_jpeg(tmp_path / "unknown.jpg")
        with patch("salesSlipScanner.load_and_encode_image", return_value="b64"), \
             patch("salesSlipScanner.query_ollama", return_value="NaN"):
            price = sss.process_file(img, "model")
        assert price is None
        assert img.exists()  # file must not be renamed

    def test_ocr_exception_returns_none(self, tmp_path):
        img = make_jpeg(tmp_path / "broken.jpg")
        with patch("salesSlipScanner.load_and_encode_image", side_effect=OSError("fail")):
            price = sss.process_file(img, "model")
        assert price is None
        assert img.exists()

    def test_existing_target_is_never_replaced(self, tmp_path):
        img = make_jpeg(tmp_path / "receipt.jpg")
        existing = tmp_path / "receipt_7949.jpg"
        existing.write_bytes(b"keep me")
        with patch("salesSlipScanner.load_and_encode_image", return_value="b64"), \
             patch("salesSlipScanner.query_ollama", return_value="79,49"):
            price = sss.process_file(img, "model")
        assert price is None
        assert img.exists()
        assert existing.read_bytes() == b"keep me"

    def test_correct_price_in_cents(self, tmp_path):
        img = make_jpeg(tmp_path / "slip.jpg")
        with patch("salesSlipScanner.load_and_encode_image", return_value="b64"), \
             patch("salesSlipScanner.query_ollama", return_value="10,93"):
            price = sss.process_file(img, "model")
        assert price == 1093
        assert (tmp_path / "slip_1093.jpg").exists()


# ===========================================================================
# run
# ===========================================================================


class TestRun:
    def _patch_model(self, model_id=sss.DEFAULT_MODEL):
        return patch(
            "salesSlipScanner.ollama.list",
            fake_ollama_list(model_id),
        )

    def test_empty_input_returns_zero_summary(self, tmp_path):
        with self._patch_model():
            summary = sss.run(input_dir=tmp_path)
        assert summary == {"processed": 0, "skipped": 0, "total_cents": 0, "results": []}

    def test_model_not_available_exits(self, tmp_path):
        with patch("salesSlipScanner.ollama.list", fake_ollama_list("other-model")):
            with pytest.raises(SystemExit):
                sss.run(model_id="missing-model", input_dir=tmp_path)

    def test_single_file_processed(self, tmp_path):
        make_jpeg(tmp_path / "a.jpg")
        with self._patch_model(), \
             patch("salesSlipScanner.load_and_encode_image", return_value="b64"), \
             patch("salesSlipScanner.query_ollama", return_value="79,49"):
            summary = sss.run(input_dir=tmp_path)
        assert summary["processed"] == 1
        assert summary["skipped"] == 0
        assert summary["total_cents"] == 7949

    def test_multiple_files_summed(self, tmp_path):
        make_jpeg(tmp_path / "a.jpg")
        make_jpeg(tmp_path / "b.jpg")
        responses = iter(["79,49", "28,41"])
        with self._patch_model(), \
             patch("salesSlipScanner.load_and_encode_image", return_value="b64"), \
             patch("salesSlipScanner.query_ollama", side_effect=responses):
            summary = sss.run(input_dir=tmp_path)
        assert summary["processed"] == 2
        assert summary["total_cents"] == 7949 + 2841

    def test_failed_file_counted_as_skipped(self, tmp_path):
        make_jpeg(tmp_path / "x.jpg")
        with self._patch_model(), \
             patch("salesSlipScanner.load_and_encode_image", return_value="b64"), \
             patch("salesSlipScanner.query_ollama", return_value="NaN"):
            summary = sss.run(input_dir=tmp_path)
        assert summary["processed"] == 0
        assert summary["skipped"] == 1
        assert summary["total_cents"] == 0

    def test_mix_of_success_and_failure(self, tmp_path):
        make_jpeg(tmp_path / "a.jpg")
        make_jpeg(tmp_path / "b.jpg")
        responses = iter(["10,93", "NaN"])
        with self._patch_model(), \
             patch("salesSlipScanner.load_and_encode_image", return_value="b64"), \
             patch("salesSlipScanner.query_ollama", side_effect=responses):
            summary = sss.run(input_dir=tmp_path)
        assert summary["processed"] == 1
        assert summary["skipped"] == 1
        assert summary["total_cents"] == 1093

    def test_success_is_recorded_and_not_reprocessed(self, tmp_path):
        make_jpeg(tmp_path / "a.jpg")
        with self._patch_model(), \
             patch("salesSlipScanner.load_and_encode_image", return_value="b64"), \
             patch("salesSlipScanner.query_ollama", return_value="0,99"):
            first = sss.run(input_dir=tmp_path)
            second = sss.run(input_dir=tmp_path)
        assert first["processed"] == 1
        assert second["processed"] == 0
        manifest = sss.load_manifest(tmp_path)
        assert manifest["receipts"][0]["file"] == "a_99.jpg"

    def test_missing_input_dir_raises(self):
        missing = Path("/tmp/does_not_exist_xyzzy")
        with patch(
            "salesSlipScanner.ollama.list", fake_ollama_list(sss.DEFAULT_MODEL)
        ):
            with pytest.raises(FileNotFoundError):
                sss.run(input_dir=missing)


# ===========================================================================
# parse_args
# ===========================================================================


class TestParseArgs:
    def test_default_model(self):
        args = sss.parse_args([])
        assert args.model == sss.DEFAULT_MODEL

    def test_custom_model_flag(self):
        args = sss.parse_args(["--model", "llama3.2-vision:11b"])
        assert args.model == "llama3.2-vision:11b"

    def test_unknown_flag_exits(self):
        with pytest.raises(SystemExit):
            sss.parse_args(["--unknown-flag"])


class TestMain:
    def test_success_returns_zero(self):
        with patch("salesSlipScanner.run", return_value={"skipped": 0}):
            assert sss.main([]) == 0

    def test_partial_failure_returns_nonzero(self):
        with patch("salesSlipScanner.run", return_value={"skipped": 1}):
            assert sss.main([]) == 1
