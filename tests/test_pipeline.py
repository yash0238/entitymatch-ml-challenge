"""
Unit tests for entity resolution pipeline
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import parse_entity_list, format_entity_list, calculate_f05_score, evaluate_entity_match
from preprocessing import TextNormalizer
from matching_model import EntityMatchingModel


class TestUtils:
    """Test utility functions"""
    
    def test_parse_entity_list(self):
        assert parse_entity_list("S2-123,S3-456") == ["S2-123", "S3-456"]
        assert parse_entity_list("") == []
        assert parse_entity_list("S2-789") == ["S2-789"]
    
    def test_format_entity_list(self):
        assert format_entity_list(["S2-123", "S3-456"]) == "S2-123,S3-456"
        assert format_entity_list([]) == ""
        assert format_entity_list(["S2-789"]) == "S2-789"
    
    def test_calculate_f05_score(self):
        # Perfect match
        assert calculate_f05_score(1.0, 1.0) == 1.0
        
        # No match
        assert calculate_f05_score(0.0, 0.0) == 0.0
        
        # Precision-heavy: precision=0.8, recall=0.6
        f05 = calculate_f05_score(0.8, 0.6)
        assert 0.7 < f05 < 0.8  # Should be closer to precision
    
    def test_evaluate_entity_match(self):
        # Perfect match
        precision, recall, f05 = evaluate_entity_match(
            ["S2-1", "S3-2"], 
            ["S2-1", "S3-2"]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f05 == 1.0
        
        # Correct singleton
        precision, recall, f05 = evaluate_entity_match([], [])
        assert precision == 1.0
        assert recall == 1.0
        assert f05 == 1.0
        
        # False positive
        precision, recall, f05 = evaluate_entity_match(["S2-1"], [])
        assert precision == 0.0
        assert recall == 0.0
        assert f05 == 0.0

    def test_entity_threshold_includes_singleton_credit(self):
        model = EntityMatchingModel()
        score, threshold = model._optimize_entity_threshold(
            np.array([1, 0, 0]),
            np.array([0.9, 0.6, 0.8]),
            np.array(['matched', 'singleton', 'negative']),
            np.array(['matched', 'singleton', 'negative']),
            {'matched': 1, 'singleton': 0, 'negative': 1},
        )
        assert score == pytest.approx(2 / 3)
        assert threshold > 0.8


class TestTextNormalizer:
    """Test text normalization"""
    
    def test_clean_text(self):
        assert TextNormalizer.clean_text("  Hello  World  ") == "hello world"
        assert TextNormalizer.clean_text("ABC@#$123") == "abc123"
        assert TextNormalizer.clean_text("Test & Co.") == "test & co."
        assert TextNormalizer.clean_text("Café Société लिमिटेड") == "café société लिमिटेड"
    
    def test_normalize_business_name(self):
        assert TextNormalizer.normalize_business_name("ABC Corporation") == "abc corp"
        assert TextNormalizer.normalize_business_name("XYZ Inc.") == "xyz inc"
        assert TextNormalizer.normalize_business_name("Test Company") == "test co"
        assert TextNormalizer.normalize_business_name("ACME लिमिटेड") == "acme ltd"
    
    def test_normalize_address(self):
        assert "st" in TextNormalizer.normalize_address("123 Main Street")
        assert "rd" in TextNormalizer.normalize_address("456 Oak Road")
        assert "ave" in TextNormalizer.normalize_address("789 Park Avenue")
    
    def test_extract_tokens(self):
        tokens = TextNormalizer.extract_tokens("hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
    
    def test_extract_numeric_components(self):
        nums = TextNormalizer.extract_numeric_components("123 Main St, Apt 4B, 90210")
        assert "123" in nums
        assert "4" in nums
        assert "90210" in nums


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
