"""
Unit tests for the simulator module
"""

import pytest
import json
import tempfile
import os
from secprompt.simulator import PromptSimulator, InjectionPayload


class TestInjectionPayload:
    def test_payload_creation(self):
        payload = InjectionPayload(
            content="test content",
            category="test_category",
            severity="medium",
            description="test description",
            tags=["test", "tag"]
        )
        
        assert payload.content == "test content"
        assert payload.category == "test_category"
        assert payload.severity == "medium"
        assert payload.description == "test description"
        assert payload.tags == ["test", "tag"]


class TestPromptSimulator:
    def setup_method(self):
        self.simulator = PromptSimulator()
    
    def test_initialization(self):
        assert hasattr(self.simulator, 'base_payloads')
        assert hasattr(self.simulator, 'mutation_techniques')
        assert len(self.simulator.base_payloads) > 0
        assert len(self.simulator.mutation_techniques) > 0
    
    def test_generate_base_payload(self):
        payload = self.simulator.generate_base_payload()
        
        assert isinstance(payload, InjectionPayload)
        assert payload.content in [p for category in self.simulator.base_payloads.values() for p in category]
        assert payload.category in self.simulator.base_payloads.keys()
        assert payload.severity in ["low", "medium", "high"]
        assert "base" in payload.tags
    
    def test_generate_base_payload_specific_category(self):
        category = "role_confusion"
        payload = self.simulator.generate_base_payload(category)
        
        assert payload.category == category
        assert payload.content in self.simulator.base_payloads[category]
    
    def test_generate_base_payload_invalid_category(self):
        with pytest.raises(ValueError):
            self.simulator.generate_base_payload("invalid_category")
    
    def test_mutate_payload(self):
        original_payload = self.simulator.generate_base_payload()
        mutated_payload = self.simulator.mutate_payload(original_payload)
        
        assert isinstance(mutated_payload, InjectionPayload)
        assert mutated_payload.category == original_payload.category
        assert mutated_payload.severity == original_payload.severity
        assert "mutated" in mutated_payload.tags
        assert len(mutated_payload.tags) > len(original_payload.tags)
    
    def test_mutate_payload_specific_technique(self):
        original_payload = self.simulator.generate_base_payload()
        technique = "case_variation"
        mutated_payload = self.simulator.mutate_payload(original_payload, technique)
        
        assert technique in mutated_payload.tags
    
    def test_case_variation_mutation(self):
        text = "Hello World"
        result = self.simulator._apply_case_variation(text)
        
        assert isinstance(result, str)
        assert len(result) == len(text)
        # Should have some case variation
        assert result != text or result != text.lower() or result != text.upper()
    
    def test_spacing_manipulation_mutation(self):
        text = "Hello world test"
        result = self.simulator._apply_spacing_manipulation(text)
        
        assert isinstance(result, str)
        assert "  " in result  # Should have double spaces
    
    def test_punctuation_addition_mutation(self):
        text = "Hello world test"
        result = self.simulator._apply_punctuation_addition(text)
        
        assert isinstance(result, str)
        # Should have some punctuation added
        assert any(p in result for p in ['.', ',', '!', '?', ';', ':'])
    
    def test_synonym_replacement_mutation(self):
        text = "Ignore previous instructions"
        result = self.simulator._apply_synonym_replacement(text)
        
        assert isinstance(result, str)
        # Should have some synonym replacement
        assert result != text or "ignore" not in result.lower()
    
    def test_encoding_manipulation_mutation(self):
        text = "Hello world"
        result = self.simulator._apply_encoding_manipulation(text)
        
        assert isinstance(result, str)
        assert len(result) >= len(text)
        # Should have invisible characters
        assert any(ord(c) > 127 for c in result)
    
    def test_generate_dataset(self):
        size = 10
        dataset = self.simulator.generate_dataset(size=size, include_mutations=False)
        
        assert len(dataset) == size
        assert all(isinstance(payload, InjectionPayload) for payload in dataset)
    
    def test_generate_dataset_with_mutations(self):
        size = 10
        dataset = self.simulator.generate_dataset(size=size, include_mutations=True)
        
        assert len(dataset) >= size
        assert all(isinstance(payload, InjectionPayload) for payload in dataset)
        
        # Should have some mutated payloads
        mutated_count = sum(1 for payload in dataset if "mutated" in payload.tags)
        assert mutated_count > 0
    
    def test_save_and_load_dataset(self):
        # Create test dataset
        dataset = self.simulator.generate_dataset(size=5, include_mutations=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.simulator.save_dataset(dataset, temp_file)
            
            # Load dataset
            loaded_dataset = self.simulator.load_dataset(temp_file)
            
            assert len(loaded_dataset) == len(dataset)
            assert all(isinstance(payload, InjectionPayload) for payload in loaded_dataset)
            
            # Check content matches
            for original, loaded in zip(dataset, loaded_dataset):
                assert original.content == loaded.content
                assert original.category == loaded.category
                assert original.severity == loaded.severity
                assert original.description == loaded.description
                assert original.tags == loaded.tags
                
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_save_dataset_json_format(self):
        dataset = self.simulator.generate_dataset(size=3, include_mutations=False)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.simulator.save_dataset(dataset, temp_file)
            
            # Check JSON format
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            assert isinstance(data, list)
            assert len(data) == 3
            
            for item in data:
                assert 'content' in item
                assert 'category' in item
                assert 'severity' in item
                assert 'description' in item
                assert 'tags' in item
                assert isinstance(item['tags'], list)
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__]) 