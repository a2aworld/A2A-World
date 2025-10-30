"""
A2A World Platform - Narrative XAI Agent Tests

Unit tests for the NarrativeXAIAgent and XAI explanation generation.
"""

import asyncio
import unittest
import uuid
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from agents.xai.narrative_xai_agent import NarrativeXAIAgent, NarrativeXAIConfig


class TestNarrativeXAIAgent(unittest.TestCase):
    """Test cases for NarrativeXAIAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = NarrativeXAIConfig()
        self.agent = NarrativeXAIAgent(config=self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        # Reset agent state
        self.agent.explanation_cache.clear()
        self.agent.narrative_templates.clear()

    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_type, "narrative_xai")
        self.assertIsInstance(self.agent.explanation_cache, dict)
        self.assertIsInstance(self.agent.narrative_templates, dict)
        self.assertEqual(self.agent.explanations_generated, 0)
        self.assertEqual(self.agent.cot_reasoning_performed, 0)

    def test_get_capabilities(self):
        """Test agent capabilities."""
        capabilities = self.agent._get_capabilities()
        expected_caps = [
            "narrative_xai", "explanation_generation", "cot_reasoning",
            "narrative_generation", "multimodal_explanations", "pattern_explanation",
            "validation_explanation", "story_driven_xai", "chain_of_thought",
            "interactive_explanations", "xai_integration", "narrative_templates",
            "explanation_caching"
        ]

        for cap in expected_caps:
            self.assertIn(cap, capabilities)

    def test_cot_observe_target(self):
        """Test CoT observation step."""
        target_data = {
            "name": "Test Pattern",
            "pattern_type": "spatial_clustering",
            "confidence_score": 0.85
        }

        result = self.agent._cot_observe_target("pattern", target_data)

        self.assertEqual(result["step_number"], 1)
        self.assertEqual(result["step_type"], "observation")
        self.assertIn("confidence_level", result)
        self.assertIn("reasoning_content", result)

    def test_cot_analyze_data(self):
        """Test CoT analysis step."""
        target_data = {
            "pattern_components": [
                {"component_id": "1", "relevance_score": 0.9},
                {"component_id": "2", "relevance_score": 0.7}
            ]
        }

        result = self.agent._cot_analyze_data("pattern", target_data)

        self.assertEqual(result["step_number"], 2)
        self.assertEqual(result["step_type"], "analysis")
        self.assertIn("component_count", result["evidence_used"])

    def test_cot_identify_insights(self):
        """Test CoT insight identification step."""
        target_data = {
            "confidence_score": 0.9,
            "statistical_significance": 0.001
        }
        previous_steps = [
            {"step_type": "observation", "confidence_level": 0.8},
            {"step_type": "analysis", "confidence_level": 0.7}
        ]

        result = self.agent._cot_identify_insights("pattern", target_data, previous_steps)

        self.assertEqual(result["step_number"], 3)
        self.assertEqual(result["step_type"], "insight_identification")
        self.assertTrue(result["is_key_insight"])

    def test_cot_consider_alternatives(self):
        """Test CoT alternative consideration step."""
        target_data = {"confidence_score": 0.8}
        insight_step = {"evidence_used": {"insights": ["high confidence"]}}

        result = self.agent._cot_consider_alternatives("pattern", target_data, insight_step)

        self.assertEqual(result["step_number"], 4)
        self.assertEqual(result["step_type"], "alternative_consideration")
        self.assertIn("alternatives", result["evidence_used"])

    def test_cot_validate_reasoning(self):
        """Test CoT validation step."""
        reasoning_steps = [
            {"confidence_level": 0.8},
            {"confidence_level": 0.7},
            {"confidence_level": 0.9}
        ]

        result = self.agent._cot_validate_reasoning(reasoning_steps)

        self.assertEqual(result["step_number"], 5)
        self.assertEqual(result["step_type"], "validation")
        self.assertIn("assessment", result["reasoning_content"])

    def test_calculate_cot_confidence(self):
        """Test CoT confidence calculation."""
        reasoning_steps = [
            {"confidence_level": 0.8},
            {"confidence_level": 0.7},
            {"confidence_level": 0.9},
            {"confidence_level": 0.6},
            {"confidence_level": 0.8}
        ]

        confidence = self.agent._calculate_cot_confidence(reasoning_steps)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_calculate_cot_confidence_empty(self):
        """Test CoT confidence calculation with empty steps."""
        confidence = self.agent._calculate_cot_confidence([])
        self.assertEqual(confidence, 0.0)

    def test_summarize_target_data_pattern(self):
        """Test target data summarization for patterns."""
        target_data = {
            "name": "Ancient Sites Pattern",
            "pattern_type": "spatial_clustering"
        }

        summary = self.agent._summarize_target_data(target_data)
        self.assertIn("Ancient Sites Pattern", summary)
        self.assertIn("spatial_clustering", summary)

    def test_summarize_target_data_validation(self):
        """Test target data summarization for validations."""
        target_data = {
            "validation_result": "approved",
            "validation_score": 0.95
        }

        summary = self.agent._summarize_target_data(target_data)
        self.assertIn("approved", summary)
        self.assertIn("0.95", summary)

    def test_analyze_target_components(self):
        """Test target component analysis."""
        target_data = {
            "pattern_components": [
                {"id": "1"}, {"id": "2"}, {"id": "3"}
            ]
        }

        result = self.agent._analyze_target_components(target_data)
        self.assertEqual(result["component_count"], 3)
        self.assertIn("components identified", result["summary"])

    def test_extract_key_insights(self):
        """Test key insight extraction."""
        target_data = {
            "confidence_score": 0.9,
            "statistical_significance": 0.001
        }
        previous_steps = []

        result = self.agent._extract_key_insights(target_data, previous_steps)
        self.assertIn("insights", result)
        self.assertIn("confidence", result)

    def test_generate_alternative_explanations(self):
        """Test alternative explanation generation."""
        target_data = {"confidence_score": 0.8}
        insight_step = {"evidence_used": {"insights": ["strong pattern"]}}

        result = self.agent._generate_alternative_explanations(target_data, insight_step)
        self.assertIn("alternatives", result)
        self.assertIn("rationale", result)

    def test_validate_reasoning_chain(self):
        """Test reasoning chain validation."""
        reasoning_steps = [
            {"confidence_level": 0.8},
            {"confidence_level": 0.9},
            {"confidence_level": 0.7}
        ]

        result = self.agent._validate_reasoning_chain(reasoning_steps)
        self.assertIn("assessment", result)
        self.assertEqual(result["step_count"], 3)

    def test_calculate_explanation_confidence(self):
        """Test explanation confidence calculation."""
        cot_result = {"confidence_score": 0.8}
        narrative_result = {"title": "Test Explanation"}

        confidence = self.agent._calculate_explanation_confidence(cot_result, narrative_result)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_assess_explanation_quality(self):
        """Test explanation quality assessment."""
        narrative_result = {
            "title": "Test Narrative",
            "word_count": 150
        }
        multimodal_elements = [
            {"element_type": "chart"},
            {"element_type": "diagram"}
        ]

        quality = self.agent._assess_explanation_quality(narrative_result, multimodal_elements)

        self.assertIn("clarity_score", quality)
        self.assertIn("completeness_score", quality)
        self.assertIn("usefulness_score", quality)
        self.assertTrue(quality["multimodal_enhancement"])
        self.assertIn("overall_quality", quality)

    def test_select_narrative_template(self):
        """Test narrative template selection."""
        # Setup templates
        self.agent.narrative_templates = {
            "pattern_intermediate": {"template_name": "pattern_intermediate"},
            "validation_expert": {"template_name": "validation_expert"}
        }

        # Test existing template
        template = self.agent._select_narrative_template("pattern", "intermediate")
        self.assertIsNone(template)  # Should not match due to key format

        # Test with correct key format
        self.agent.narrative_templates["pattern_intermediate"] = {"template_name": "pattern_intermediate"}
        template = self.agent._select_narrative_template("pattern", "intermediate")
        self.assertIsNotNone(template)
        self.assertEqual(template["template_name"], "pattern_intermediate")

    def test_enhance_with_storytelling(self):
        """Test storytelling enhancement."""
        narrative = {
            "title": "Pattern Explanation",
            "body": "This is a pattern explanation."
        }
        target_data = {"pattern_type": "spatial"}

        enhanced = self.agent._enhance_with_storytelling(narrative, target_data)

        self.assertIn("story_elements", enhanced)
        self.assertIn("context", enhanced["story_elements"])

    @patch('agents.xai.narrative_xai_agent.PatternStorage')
    async def test_get_target_data_pattern(self, mock_pattern_storage):
        """Test getting pattern target data."""
        # Setup mock
        mock_storage = AsyncMock()
        mock_storage.get_pattern.return_value = {"id": "test-pattern", "name": "Test Pattern"}
        mock_pattern_storage.return_value = mock_storage

        # Create agent with mocked storage
        agent = NarrativeXAIAgent()
        agent.pattern_storage = mock_storage

        # Test getting pattern data
        result = await agent._get_target_data("pattern", "test-pattern-id")

        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "test-pattern")
        mock_storage.get_pattern.assert_called_once_with("test-pattern-id")

    @patch('agents.xai.narrative_xai_agent.PatternStorage')
    async def test_get_target_data_validation(self, mock_pattern_storage):
        """Test getting validation target data."""
        # Setup mock
        mock_storage = AsyncMock()
        mock_storage.get_validation.return_value = {"id": "test-validation", "result": "approved"}
        mock_pattern_storage.return_value = mock_storage

        # Create agent with mocked storage
        agent = NarrativeXAIAgent()
        agent.pattern_storage = mock_storage

        # Test getting validation data
        result = await agent._get_target_data("validation", "test-validation-id")

        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "test-validation")
        mock_storage.get_validation.assert_called_once_with("test-validation-id")

    async def test_get_target_data_invalid_type(self):
        """Test getting target data with invalid type."""
        agent = NarrativeXAIAgent()

        result = await agent._get_target_data("invalid", "test-id")

        self.assertIsNone(result)

    def test_cleanup_explanation_cache(self):
        """Test explanation cache cleanup."""
        # Add some test cache entries
        old_time = datetime.utcnow().replace(hour=datetime.utcnow().hour - 25)  # 25 hours ago
        new_time = datetime.utcnow()

        self.agent.explanation_cache = {
            "old_key": {"generated_at": old_time.isoformat()},
            "new_key": {"generated_at": new_time.isoformat()},
            "invalid_key": {}  # Missing timestamp
        }

        # Run cleanup (this would be async in real implementation)
        # For testing, we'll simulate the cleanup logic
        current_time = datetime.utcnow()
        cache_ttl_seconds = self.agent.xai_config.cache_ttl_hours * 3600

        keys_to_remove = []
        for key, explanation in self.agent.explanation_cache.items():
            if "generated_at" in explanation:
                try:
                    gen_time = datetime.fromisoformat(explanation["generated_at"])
                    if (current_time - gen_time).total_seconds() > cache_ttl_seconds:
                        keys_to_remove.append(key)
                except (ValueError, KeyError):
                    keys_to_remove.append(key)

        # Should remove old_key and invalid_key
        self.assertIn("old_key", keys_to_remove)
        self.assertIn("invalid_key", keys_to_remove)
        self.assertNotIn("new_key", keys_to_remove)


class TestNarrativeXAIConfig(unittest.TestCase):
    """Test cases for NarrativeXAIConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NarrativeXAIConfig()

        self.assertEqual(config.max_narrative_length, 2000)
        self.assertEqual(config.default_audience_level, "intermediate")
        self.assertEqual(config.cot_max_steps, 10)
        self.assertTrue(config.enable_multimodal)
        self.assertTrue(config.cache_explanations)
        self.assertEqual(config.cache_ttl_hours, 24)
        self.assertEqual(config.generation_timeout_seconds, 30)
        self.assertEqual(config.quality_threshold, 0.7)


if __name__ == '__main__':
    unittest.main()