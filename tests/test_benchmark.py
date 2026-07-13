"""Offline tests for the resumable benchmark."""

from unittest.mock import patch

import pytest

from localVisionModelTest import benchmark as bm


def result(trials):
    return {"trials": trials}


def trial(image="a.jpg", correct=True, latency=1.0, error=None):
    return {"image": image, "correct": correct, "wall_seconds": latency, "error": error}


def test_errors_remain_in_accuracy_denominator():
    spec = bm.MODELS[0]
    row = bm.summary(spec, result([trial(), trial(correct=False, error="timeout")]))
    assert row["accuracy"] == 0.5
    assert row["errors"] == 1


def test_stable_receipt_requires_every_run_correct():
    spec = bm.MODELS[0]
    row = bm.summary(spec, result([trial(), trial(), trial(correct=False)]))
    assert row["stable"] == 0


def test_recommend_smallest_model_within_speed_margin():
    rows = [
        {
            "model": "fast",
            "name": "Fast",
            "total": 9,
            "accuracy": 1,
            "stable": 3,
            "warm": 1.0,
            "size_gb": 4,
        },
        {
            "model": "small",
            "name": "Small",
            "total": 9,
            "accuracy": 1,
            "stable": 3,
            "warm": 1.4,
            "size_gb": 2,
        },
    ]
    assert bm.recommend(rows)["model"] == "small"


def test_missing_model_never_downloads_without_permission():
    with (
        patch.object(bm, "installed", return_value={}),
        patch.object(bm.ollama, "pull") as pull,
    ):
        with pytest.raises(RuntimeError, match="allow-downloads"):
            bm.pull(bm.MODELS[0], False, 10)
    pull.assert_not_called()


def test_capacity_check_uses_size_plus_reserve():
    with (
        patch.object(bm, "installed", return_value={}),
        patch.object(bm, "free_gb", return_value=11),
    ):
        with pytest.raises(RuntimeError, match="required"):
            bm.pull(bm.MODELS[0], True, 10)


def test_cli_requires_explicit_selection():
    with pytest.raises(SystemExit):
        bm.parse_args([])


def test_catalog_contains_fifteen_local_models():
    assert len(bm.MODELS) == 15
