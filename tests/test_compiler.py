"""Tests for compiler module."""
import pytest
from server.compiler.capability import ComfyUICapabilities
from server.compiler.templates import load_template, list_templates, _fallback_workflow
from pathlib import Path


class TestCapabilities:
    def test_default_strategy(self):
        caps = ComfyUICapabilities()
        assert caps.best_strategy == "mega_prompt"
    
    def test_gligen_priority(self):
        caps = ComfyUICapabilities(has_gligen=True, has_attention_couple=True)
        assert caps.best_strategy == "gligen"
    
    def test_attention_couple_not_used(self):
        # attention_couple removed from best_strategy — ideogram4/flux/gligen/mega_prompt
        caps = ComfyUICapabilities(has_attention_couple=True)
        assert caps.best_strategy == "mega_prompt"  # falls through to default
    
    def test_not_connected(self):
        caps = ComfyUICapabilities(connected=False)
        assert caps.best_strategy == "mega_prompt"


class TestTemplates:
    def test_fallback_workflow(self):
        wf = _fallback_workflow()
        assert "3" in wf  # KSampler
        assert "4" in wf  # CheckpointLoader
        assert "6" in wf  # CLIPTextEncode
        assert wf["6"]["inputs"]["text"] == "a beautiful landscape"
    
    def test_list_templates(self):
        templates_dir = Path(__file__).parent.parent / "workflows" / "templates"
        if templates_dir.exists():
            names = list_templates(templates_dir)
            assert "mega_prompt" in names
    
    def test_load_template_copy(self):
        templates_dir = Path(__file__).parent.parent / "workflows" / "templates"
        if not templates_dir.exists():
            pytest.skip("Templates directory not found")
        
        t1 = load_template("mega_prompt", templates_dir)
        t2 = load_template("mega_prompt", templates_dir)
        
        # Mutating one should not affect the other
        t1["6"]["inputs"]["text"] = "modified"
        assert t2["6"]["inputs"]["text"] != "modified"
