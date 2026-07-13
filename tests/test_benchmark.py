"""Offline unit tests for benchmark scoring and selection."""

from argparse import Namespace

import pytest

from localVisionModelTest import benchmark as bm


def test_inference_errors_count_as_incorrect():
    result = bm.ModelResult(
        "model",
        "Model",
        "1 GB",
        0,
        image_results=[
            bm.ImageResult("a_100.jpg", "1,00", "1,00", True, 1.0),
            bm.ImageResult("b_200.jpg", "2,00", "", False, 0.0, "timeout"),
        ],
    )
    assert result.accuracy == 0.5
    assert result.correct_count == 1


def test_expected_amount_comes_from_final_filename_suffix():
    assert bm.extract_expected("shop_2024_receipt_1093.jpg") == "10,93"
    assert bm.extract_expected("receipt.jpg") == "?"


def test_model_catalog_is_available_for_repeatable_selection():
    assert len(bm.MODELS) == 13
    assert bm.selected_models(Namespace(all=False, model=["qwen3-vl:4b"])) == [
        next(spec for spec in bm.MODELS if spec["id"] == "qwen3-vl:4b")
    ]


def test_all_and_individual_selection_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        bm.parse_args(["--all", "--model", "moondream"])
