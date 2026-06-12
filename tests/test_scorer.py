"""Tests for scoring module."""
import pytest
from server.scorer.compositional import compute_bbox_iou
from server.scorer.scorer import ForgeScore, RegionScore


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
        # Intersection: 0.2*0.2 = 0.04, Union: 0.36+0.36-0.04 = 0.68
        assert iou == pytest.approx(0.04 / 0.68, abs=0.01)
    
    def test_contained_box(self):
        outer = [0.0, 0.0, 1.0, 1.0]
        inner = [0.25, 0.25, 0.75, 0.75]
        iou = compute_bbox_iou(outer, inner)
        # Intersection = inner area = 0.25, Union = outer area = 1.0
        assert iou == pytest.approx(0.25, abs=0.01)
    
    def test_empty_boxes(self):
        assert compute_bbox_iou([], [0.1, 0.1, 0.5, 0.5]) == 0.0
        assert compute_bbox_iou([0.1, 0.1, 0.5, 0.5], []) == 0.0
        assert compute_bbox_iou([], []) == 0.0


class TestForgeScore:
    def test_convergence(self):
        score = ForgeScore(overall=0.9, converged=True, threshold=0.85)
        assert score.converged is True
    
    def test_no_convergence(self):
        score = ForgeScore(overall=0.5, converged=False, threshold=0.85)
        assert score.converged is False
    
    def test_diagnosis_missing_element(self):
        region = RegionScore(element_id="e1", label="car", present=False)
        score = ForgeScore(regions=[region])
        diag = score.diagnosis()
        assert any("missing" in d.lower() for d in diag)
    
    def test_diagnosis_misplaced(self):
        region = RegionScore(element_id="e1", label="car", bbox_iou=0.3, present=True)
        score = ForgeScore(regions=[region])
        diag = score.diagnosis()
        assert any("misplaced" in d.lower() or "iou" in d.lower() for d in diag)
    
    def test_to_dict(self):
        region = RegionScore(element_id="e1", label="car", bbox_iou=0.8, dino_similarity=0.7, clip_score=0.9)
        score = ForgeScore(composition=0.8, style=0.7, subject=0.7, overall=0.75, regions=[region])
        d = score.to_dict()
        assert "overall" in d
        assert "regions" in d
        assert d["regions"][0]["label"] == "car"


class TestRegionScore:
    def test_composite_present(self):
        r = RegionScore(element_id="e1", label="test", bbox_iou=0.8, dino_similarity=0.9, clip_score=0.7, present=True)
        # 0.4*0.8 + 0.4*0.9 + 0.2*0.7 = 0.32 + 0.36 + 0.14 = 0.82
        assert r.composite == pytest.approx(0.82, abs=0.01)
    
    def test_composite_missing(self):
        r = RegionScore(element_id="e1", label="test", present=False)
        assert r.composite == 0.0
