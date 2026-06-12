"""Tests for analyzer module."""
import pytest
from server.analyzer.florence import Element, AnalysisResult, FlorenceAnalyzer
from server.analyzer.palette import extract_palette
from server.analyzer.style import StyleExtractor, STYLE_VOCABULARY
from PIL import Image


class TestElement:
    def test_creation(self):
        e = Element(id="e1", type="obj", label="car", description="red car", bbox=[0.1, 0.2, 0.5, 0.8])
        assert e.id == "e1"
        assert e.type == "obj"
        assert e.bbox == [0.1, 0.2, 0.5, 0.8]


class TestAnalysisResult:
    def test_empty(self):
        r = AnalysisResult()
        assert r.caption == ""
        assert r.elements == []
        assert r.model_used == ""


class TestPalette:
    def test_extract_palette(self):
        # Create a simple test image (red + blue halves)
        img = Image.new("RGB", (100, 100))
        for x in range(50):
            for y in range(100):
                img.putpixel((x, y), (255, 0, 0))
        for x in range(50, 100):
            for y in range(100):
                img.putpixel((x, y), (0, 0, 255))
        
        palette = extract_palette(img, n_colors=3)
        assert len(palette) == 3
        # All should be valid hex colors
        for color in palette:
            assert color.startswith("#")
            assert len(color) == 7


class TestStyleExtractor:
    def test_default_style(self):
        ext = StyleExtractor()
        style = ext._default_style()
        assert "aesthetics" in style
        assert "lighting" in style
        assert "medium" in style
    
    def test_vocabulary_structure(self):
        for field, candidates in STYLE_VOCABULARY.items():
            assert isinstance(candidates, list)
            assert len(candidates) > 0


class TestFlorenceAnalyzer:
    def test_to_ideogram_json(self):
        analyzer = FlorenceAnalyzer()
        analysis = AnalysisResult(
            caption="A red car on a street",
            background="Urban street with buildings",
            elements=[
                Element(id="e1", type="obj", label="car", description="red car", bbox=[0.2, 0.3, 0.8, 0.7]),
            ]
        )
        result = analyzer.to_ideogram_json(analysis)
        assert "caption" in result
        assert "composition" in result
        assert len(result["composition"]["elements"]) == 1
        # Check bbox conversion (normalized to 1024)
        elem = result["composition"]["elements"][0]
        assert elem["bbox"] == [205, 307, 819, 717]
