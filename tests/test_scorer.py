"""Tests for scoring module."""
import pytest
from server.scorer.compositional import compute_bbox_iou
from server.scorer.scorer import ForgeScore, RegionScore, Scorer


class TestBboxIoU:
    def test_identical_boxes(self):
        box = [0.1, 0.1, 0.5, 0.5]
        assert compute_bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = [0.0, 0.0, 0.3, 0.3]
        b = [0.7, 0.7, 1.0, 1.0]
        assert compute_bbox_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = [0.0, 0.0, 0.6, 0.6]
        b = [0.4, 0.4, 1.0, 1.0]
        iou = compute_bbox_iou(a, b)
        assert 0.0 < iou < 1.0
        assert iou == pytest.approx(0.04 / 0.68, abs=0.01)

    def test_contained_box(self):
        outer = [0.0, 0.0, 1.0, 1.0]
        inner = [0.25, 0.25, 0.75, 0.75]
        iou = compute_bbox_iou(outer, inner)
        assert iou == pytest.approx(0.25, abs=0.01)

    def test_empty_boxes(self):
        assert compute_bbox_iou([], [0.1, 0.1, 0.5, 0.5]) == 0.0
        assert compute_bbox_iou([0.1, 0.1, 0.5, 0.5], []) == 0.0
        assert compute_bbox_iou([], []) == 0.0


class TestForgeScore:
    def test_convergence(self):
        score = ForgeScore(overall=0.9, converged=True)
        assert score.converged is True

    def test_no_convergence(self):
        score = ForgeScore(overall=0.3, converged=False)
        assert score.converged is False

    def test_diagnosis_method(self):
        score = ForgeScore(overall=0.3, _diagnosis=["style: low match"])
        diag = score.diagnosis()
        assert isinstance(diag, list)
        assert len(diag) == 1
        assert "style" in diag[0]

    def test_diagnosis_empty(self):
        score = ForgeScore(overall=0.9, converged=True)
        assert score.diagnosis() == []

    def test_to_dict(self):
        region = RegionScore(label="car", bbox=[0.1, 0.2, 0.5, 0.6], clip_score=0.7, composite=0.7)
        score = ForgeScore(composition=0.8, style=0.7, subject=0.7, overall=0.75, regions=[region])
        d = score.to_dict()
        assert "overall" in d
        assert d["overall"] == 0.75
        assert "regions" in d
        assert d["regions"][0]["label"] == "car"
        assert d["regions"][0]["clip_score"] == 0.7


class TestRegionScore:
    def test_creation(self):
        r = RegionScore(label="test", bbox=[0.1, 0.2, 0.3, 0.4], clip_score=0.8, composite=0.8)
        assert r.label == "test"
        assert r.clip_score == 0.8
        assert r.composite == 0.8

    def test_diagnosis_string(self):
        r = RegionScore(label="dog", bbox=[], clip_score=0.2, composite=0.2, diagnosis="dog: not detected")
        assert "dog" in r.diagnosis

    def test_default_values(self):
        r = RegionScore(label="car", bbox=[])
        assert r.clip_score == 0.0
        assert r.composite == 0.0
        assert r.diagnosis == ""


class TestNormalization:
    """Test the calibrated piecewise linear normalization."""

    def test_bad_match(self):
        # Raw < 0.15 should map to < 0.15
        assert Scorer.normalize(0.10) < 0.15
        assert Scorer.normalize(0.05) < 0.10

    def test_needs_mutation(self):
        # Raw 0.15-0.22 should map to 0.15-0.50
        score = Scorer.normalize(0.18)
        assert 0.15 <= score <= 0.50

    def test_good_match(self):
        # Raw 0.22-0.30 should map to 0.50-0.85
        score = Scorer.normalize(0.25)
        assert 0.50 <= score <= 0.85

    def test_excellent(self):
        # Raw > 0.30 should map to 0.85-1.0
        score = Scorer.normalize(0.35)
        assert score >= 0.85

    def test_clamped(self):
        assert Scorer.normalize(-0.1) == 0.0
        assert Scorer.normalize(0.50) == 1.0
