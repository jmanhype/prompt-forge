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
        assert "high_level_description" in result
        assert result["high_level_description"] == "A red car on a street"
        assert "compositional_decomposition" in result
        assert "background" in result["compositional_decomposition"]
        assert "elements" in result["compositional_decomposition"]
        assert len(result["compositional_decomposition"]["elements"]) == 1
        
        # style_description should NOT be present (cocktailpeanut format)
        assert "style_description" not in result
        
        # Check bbox is in correct Ideogram 4 format: [y_min, x_min, y_max, x_max] in 0-1000
        elem = result["compositional_decomposition"]["elements"][0]
        assert elem["type"] == "obj"
        assert elem["desc"] == "red car"
        assert elem["bbox"] == [300, 200, 700, 800]  # [y1*1000, x1*1000, y2*1000, x2*1000]
