"""
A2A World Platform - Ethical Validation Framework

Implements human flourishing alignment validation and bias diversity assessment
for ethical pattern validation in multidisciplinary protocols.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import uuid
import json
from dataclasses import dataclass
from enum import Enum

from agents.core.pattern_storage import PatternStorage


class FlourishingDimension(Enum):
    """Dimensions of human flourishing based on positive psychology."""
    EMOTIONAL_WELLBEING = "emotional_wellbeing"
    PSYCHOLOGICAL_WELLBEING = "psychological_wellbeing"
    SOCIAL_WELLBEING = "social_wellbeing"
    PHYSICAL_HEALTH = "physical_health"
    SPIRITUAL_GROWTH = "spiritual_growth"
    ENVIRONMENTAL_HARMONY = "environmental_harmony"


class BiasType(Enum):
    """Types of bias to assess in validation."""
    CULTURAL_BIAS = "cultural_bias"
    GEOGRAPHICAL_BIAS = "geographical_bias"
    TEMPORAL_BIAS = "temporal_bias"
    METHODOLOGICAL_BIAS = "methodological_bias"
    INTERPRETATION_BIAS = "interpretation_bias"
    REPRESENTATION_BIAS = "representation_bias"


@dataclass
class EthicalAssessment:
    """Ethical assessment result."""
    flourishing_score: float
    bias_score: float
    ethical_compliance: bool
    flourishing_dimensions: Dict[str, float]
    identified_biases: List[Dict[str, Any]]
    ethical_concerns: List[str]
    recommendations: List[str]
    assessment_timestamp: str
    assessed_by: str


class HumanFlourishingValidator:
    """
    Validates alignment with human flourishing principles.

    Assesses patterns against dimensions of human wellbeing, positive psychology,
    and ethical considerations for human development.
    """

    def __init__(self):
        """Initialize human flourishing validator."""
        self.logger = logging.getLogger(__name__)

        # Flourishing assessment criteria weights
        self.flourishing_weights = {
            FlourishingDimension.EMOTIONAL_WELLBEING: 0.2,
            FlourishingDimension.PSYCHOLOGICAL_WELLBEING: 0.2,
            FlourishingDimension.SOCIAL_WELLBEING: 0.15,
            FlourishingDimension.PHYSICAL_HEALTH: 0.15,
            FlourishingDimension.SPIRITUAL_GROWTH: 0.15,
            FlourishingDimension.ENVIRONMENTAL_HARMONY: 0.15
        }

        # Keywords and indicators for each flourishing dimension
        self.flourishing_indicators = {
            FlourishingDimension.EMOTIONAL_WELLBEING: [
                "joy", "happiness", "peace", "harmony", "contentment", "emotional",
                "wellbeing", "positive emotions", "resilience", "emotional intelligence"
            ],
            FlourishingDimension.PSYCHOLOGICAL_WELLBEING: [
                "growth", "purpose", "meaning", "autonomy", "competence", "mastery",
                "self-actualization", "psychological", "mental health", "cognitive"
            ],
            FlourishingDimension.SOCIAL_WELLBEING: [
                "community", "relationship", "connection", "social", "belonging",
                "empathy", "compassion", "collaboration", "social harmony", "interpersonal"
            ],
            FlourishingDimension.PHYSICAL_HEALTH: [
                "health", "vitality", "physical", "wellness", "energy", "strength",
                "healing", "restoration", "bodily", "physiological"
            ],
            FlourishingDimension.SPIRITUAL_GROWTH: [
                "spiritual", "transcendence", "meaning", "purpose", "sacred", "divine",
                "consciousness", "enlightenment", "wisdom", "inner peace"
            ],
            FlourishingDimension.ENVIRONMENTAL_HARMONY: [
                "nature", "environment", "earth", "harmony", "sustainability", "balance",
                "ecological", "natural", "planetary", "biosphere"
            ]
        }

        # Ethical concern indicators
        self.ethical_concerns = {
            "manipulation": ["control", "influence", "persuasion", "coercion"],
            "exploitation": ["exploit", "take advantage", "unfair", "unequal"],
            "harm": ["harm", "damage", "injure", "negative impact"],
            "discrimination": ["discriminate", "exclude", "bias", "prejudice"],
            "privacy": ["privacy", "confidentiality", "personal data", "surveillance"],
            "autonomy": ["autonomous", "freedom", "choice", "consent"]
        }

    async def assess_human_flourishing(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess human flourishing alignment of a pattern.

        Args:
            pattern_data: Pattern data to assess

        Returns:
            Human flourishing assessment
        """
        try:
            self.logger.info("Starting human flourishing assessment")

            assessment = {
                "assessment_id": str(uuid.uuid4()),
                "pattern_id": pattern_data.get("id", pattern_data.get("pattern_id")),
                "flourishing_score": 0.0,
                "dimension_scores": {},
                "ethical_concerns": [],
                "flourishing_alignment": {},
                "recommendations": [],
                "assessment_timestamp": datetime.utcnow().isoformat(),
                "assessment_method": "comprehensive_flourishing_analysis"
            }

            # Extract text content for analysis
            text_content = self._extract_text_content(pattern_data)

            # Assess each flourishing dimension
            dimension_scores = {}
            for dimension, indicators in self.flourishing_indicators.items():
                score = self._assess_dimension(text_content, indicators)
                dimension_scores[dimension.value] = score

            assessment["dimension_scores"] = dimension_scores

            # Calculate overall flourishing score
            weighted_sum = sum(
                dimension_scores[dimension.value] * weight
                for dimension, weight in self.flourishing_weights.items()
            )
            assessment["flourishing_score"] = weighted_sum

            # Assess ethical concerns
            ethical_analysis = self._assess_ethical_concerns(text_content)
            assessment["ethical_concerns"] = ethical_analysis["concerns"]

            # Adjust flourishing score based on ethical concerns
            ethical_penalty = len(assessment["ethical_concerns"]) * 0.1
            assessment["flourishing_score"] = max(0.0, assessment["flourishing_score"] - ethical_penalty)

            # Determine flourishing alignment
            assessment["flourishing_alignment"] = self._determine_flourishing_alignment(
                assessment["flourishing_score"], dimension_scores
            )

            # Generate recommendations
            assessment["recommendations"] = self._generate_flourishing_recommendations(
                assessment["flourishing_score"], assessment["ethical_concerns"]
            )

            return assessment

        except Exception as e:
            self.logger.error(f"Human flourishing assessment failed: {e}")
            return {
                "error": str(e),
                "flourishing_score": 0.0,
                "assessment_timestamp": datetime.utcnow().isoformat()
            }

    def _extract_text_content(self, pattern_data: Dict[str, Any]) -> str:
        """Extract text content from pattern data for analysis."""
        text_parts = []

        # Add pattern name and description
        text_parts.append(pattern_data.get("name", ""))
        text_parts.append(pattern_data.get("description", ""))

        # Add component descriptions
        for component in pattern_data.get("pattern_components", []):
            if "description" in component:
                text_parts.append(component["description"])

        # Add metadata descriptions
        metadata = pattern_data.get("metadata", {})
        for key, value in metadata.items():
            if isinstance(value, str):
                text_parts.append(value)

        return " ".join(text_parts).lower()

    def _assess_dimension(self, text_content: str, indicators: List[str]) -> float:
        """Assess a specific flourishing dimension."""
        try:
            matches = sum(1 for indicator in indicators if indicator in text_content)
            score = min(1.0, matches / max(1, len(indicators) * 0.5))  # Normalize to 0-1

            # Boost score for multiple matches
            if matches >= 3:
                score = min(1.0, score * 1.2)
            elif matches >= 2:
                score = min(1.0, score * 1.1)

            return score

        except Exception as e:
            self.logger.error(f"Dimension assessment failed: {e}")
            return 0.0

    def _assess_ethical_concerns(self, text_content: str) -> Dict[str, Any]:
        """Assess ethical concerns in the content."""
        concerns = []

        try:
            for concern_type, keywords in self.ethical_concerns.items():
                matches = [keyword for keyword in keywords if keyword in text_content]
                if matches:
                    concerns.append({
                        "concern_type": concern_type,
                        "severity": "moderate" if len(matches) > 1 else "low",
                        "matched_keywords": matches,
                        "description": f"Potential {concern_type} concerns identified"
                    })

        except Exception as e:
            self.logger.error(f"Ethical concerns assessment failed: {e}")

        return {"concerns": concerns}

    def _determine_flourishing_alignment(
        self,
        overall_score: float,
        dimension_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Determine overall flourishing alignment."""
        alignment = {
            "alignment_level": "unknown",
            "strengths": [],
            "weaknesses": [],
            "primary_dimensions": []
        }

        try:
            # Determine alignment level
            if overall_score >= 0.8:
                alignment["alignment_level"] = "highly_aligned"
            elif overall_score >= 0.6:
                alignment["alignment_level"] = "well_aligned"
            elif overall_score >= 0.4:
                alignment["alignment_level"] = "moderately_aligned"
            elif overall_score >= 0.2:
                alignment["alignment_level"] = "weakly_aligned"
            else:
                alignment["alignment_level"] = "not_aligned"

            # Identify strengths and weaknesses
            for dimension, score in dimension_scores.items():
                if score >= 0.7:
                    alignment["strengths"].append(dimension)
                elif score <= 0.3:
                    alignment["weaknesses"].append(dimension)

            # Identify primary dimensions (highest scores)
            sorted_dimensions = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
            alignment["primary_dimensions"] = [dim for dim, score in sorted_dimensions[:3] if score > 0.5]

        except Exception as e:
            self.logger.error(f"Flourishing alignment determination failed: {e}")

        return alignment

    def _generate_flourishing_recommendations(
        self,
        flourishing_score: float,
        ethical_concerns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate flourishing-based recommendations."""
        recommendations = []

        try:
            # Score-based recommendations
            if flourishing_score >= 0.8:
                recommendations.append("Pattern highly aligned with human flourishing principles")
                recommendations.append("Suitable for applications promoting wellbeing and positive development")
            elif flourishing_score >= 0.6:
                recommendations.append("Pattern shows good alignment with human flourishing")
                recommendations.append("Consider enhancing aspects of social and environmental wellbeing")
            elif flourishing_score >= 0.4:
                recommendations.append("Pattern has moderate flourishing alignment")
                recommendations.append("Focus on strengthening psychological and emotional wellbeing aspects")
            else:
                recommendations.append("Pattern shows limited alignment with human flourishing")
                recommendations.append("Consider redesign to incorporate more wellbeing-promoting elements")

            # Ethical concern recommendations
            if ethical_concerns:
                recommendations.append(f"Address {len(ethical_concerns)} identified ethical concerns")
                for concern in ethical_concerns:
                    concern_type = concern.get("concern_type", "general")
                    if concern_type == "privacy":
                        recommendations.append("Implement privacy protection measures")
                    elif concern_type == "autonomy":
                        recommendations.append("Ensure user autonomy and informed consent")
                    elif concern_type == "harm":
                        recommendations.append("Conduct harm reduction assessment")
                    else:
                        recommendations.append(f"Address {concern_type} ethical considerations")

            # General recommendations
            recommendations.extend([
                "Consider impact on diverse user populations",
                "Document ethical considerations and flourishing alignment",
                "Regular ethical review and impact assessment recommended"
            ])

        except Exception as e:
            self.logger.error(f"Flourishing recommendations generation failed: {e}")
            recommendations.append("Manual ethical review recommended due to processing error")

        return recommendations


class BiasDiversityAssessor:
    """
    Assesses bias and diversity in validation results and pattern data.

    Evaluates representation across cultural, geographical, temporal, and
    methodological dimensions to ensure comprehensive and unbiased validation.
    """

    def __init__(self):
        """Initialize bias diversity assessor."""
        self.logger = logging.getLogger(__name__)

        # Diversity criteria and their importance weights
        self.diversity_criteria = {
            "cultural_diversity": 0.25,
            "geographical_diversity": 0.20,
            "temporal_diversity": 0.15,
            "methodological_diversity": 0.20,
            "stakeholder_diversity": 0.10,
            "data_source_diversity": 0.10
        }

        # Bias detection patterns
        self.bias_patterns = {
            BiasType.CULTURAL_BIAS: {
                "indicators": ["western", "european", "american", "single culture"],
                "description": "Over-reliance on single cultural perspective"
            },
            BiasType.GEOGRAPHICAL_BIAS: {
                "indicators": ["single region", "local", "limited geography"],
                "description": "Geographical concentration limiting generalizability"
            },
            BiasType.TEMPORAL_BIAS: {
                "indicators": ["contemporary", "modern", "single period"],
                "description": "Temporal limitation in data or perspective"
            },
            BiasType.METHODOLOGICAL_BIAS: {
                "indicators": ["single method", "limited approach", "methodological homogeneity"],
                "description": "Limited methodological approaches"
            },
            BiasType.INTERPRETATION_BIAS: {
                "indicators": ["subjective", "biased interpretation", "confirmation bias"],
                "description": "Potential interpretation bias in analysis"
            },
            BiasType.REPRESENTATION_BIAS: {
                "indicators": ["underrepresented", "marginalized", "excluded groups"],
                "description": "Underrepresentation of certain groups or perspectives"
            }
        }

    async def audit_bias_diversity(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit bias and diversity in validation results.

        Args:
            validation_data: Validation results or pattern data to audit

        Returns:
            Bias diversity audit results
        """
        try:
            self.logger.info("Starting bias diversity audit")

            audit = {
                "audit_id": str(uuid.uuid4()),
                "diversity_score": 0.0,
                "bias_score": 1.0,  # Start with no bias assumption
                "diversity_dimensions": {},
                "identified_biases": [],
                "diversity_recommendations": [],
                "audit_timestamp": datetime.utcnow().isoformat(),
                "audit_method": "comprehensive_bias_diversity_assessment"
            }

            # Assess diversity across different dimensions
            diversity_scores = {}
            for criterion, weight in self.diversity_criteria.items():
                score = self._assess_diversity_dimension(validation_data, criterion)
                diversity_scores[criterion] = score

            audit["diversity_dimensions"] = diversity_scores

            # Calculate overall diversity score
            weighted_diversity = sum(
                diversity_scores[criterion] * weight
                for criterion, weight in self.diversity_criteria.items()
            )
            audit["diversity_score"] = weighted_diversity

            # Detect biases
            bias_analysis = self._detect_biases(validation_data)
            audit["identified_biases"] = bias_analysis["biases"]

            # Calculate bias score (inverse of bias severity)
            bias_severity = bias_analysis["overall_severity"]
            audit["bias_score"] = max(0.0, 1.0 - bias_severity)

            # Generate diversity recommendations
            audit["diversity_recommendations"] = self._generate_diversity_recommendations(
                diversity_scores, bias_analysis["biases"]
            )

            # Overall assessment
            audit["overall_assessment"] = self._assess_overall_diversity_bias(
                audit["diversity_score"], audit["bias_score"]
            )

            return audit

        except Exception as e:
            self.logger.error(f"Bias diversity audit failed: {e}")
            return {
                "error": str(e),
                "diversity_score": 0.5,
                "bias_score": 0.5,
                "audit_timestamp": datetime.utcnow().isoformat()
            }

    def _assess_diversity_dimension(self, validation_data: Dict[str, Any], dimension: str) -> float:
        """Assess diversity in a specific dimension."""
        try:
            score = 0.0

            if dimension == "cultural_diversity":
                score = self._assess_cultural_diversity(validation_data)
            elif dimension == "geographical_diversity":
                score = self._assess_geographical_diversity(validation_data)
            elif dimension == "temporal_diversity":
                score = self._assess_temporal_diversity(validation_data)
            elif dimension == "methodological_diversity":
                score = self._assess_methodological_diversity(validation_data)
            elif dimension == "stakeholder_diversity":
                score = self._assess_stakeholder_diversity(validation_data)
            elif dimension == "data_source_diversity":
                score = self._assess_data_source_diversity(validation_data)

            return score

        except Exception as e:
            self.logger.error(f"Diversity dimension assessment failed for {dimension}: {e}")
            return 0.5

    def _assess_cultural_diversity(self, validation_data: Dict[str, Any]) -> float:
        """Assess cultural diversity in validation data."""
        try:
            # Check for multiple cultural traditions
            cultural_elements = []

            # Look in pattern data
            if "cultural_associations" in validation_data:
                cultural_elements.extend(validation_data["cultural_associations"])

            # Look in validation results
            if "layer_results" in validation_data:
                cultural_layer = validation_data["layer_results"].get("cultural", {})
                if "cultural_contexts" in cultural_layer:
                    cultural_elements.extend(cultural_layer["cultural_contexts"])

            # Calculate diversity score
            unique_cultures = len(set(cultural_elements))
            if unique_cultures >= 5:
                return 0.9
            elif unique_cultures >= 3:
                return 0.7
            elif unique_cultures >= 2:
                return 0.5
            elif unique_cultures >= 1:
                return 0.3
            else:
                return 0.1

        except Exception:
            return 0.5

    def _assess_geographical_diversity(self, validation_data: Dict[str, Any]) -> float:
        """Assess geographical diversity."""
        try:
            geographical_elements = []

            # Check pattern region
            if "discovery_region" in validation_data:
                geographical_elements.append("pattern_region")

            # Check component locations
            components = validation_data.get("pattern_components", [])
            locations = set()
            for comp in components:
                if "location" in comp:
                    # Simplified: count unique location references
                    locations.add(str(comp["location"]))

            geographical_elements.extend(locations)

            # Calculate diversity score
            unique_locations = len(geographical_elements)
            if unique_locations >= 10:
                return 0.9
            elif unique_locations >= 5:
                return 0.7
            elif unique_locations >= 3:
                return 0.5
            elif unique_locations >= 1:
                return 0.3
            else:
                return 0.1

        except Exception:
            return 0.5

    def _assess_temporal_diversity(self, validation_data: Dict[str, Any]) -> float:
        """Assess temporal diversity."""
        try:
            temporal_elements = []

            # Check time periods
            if "temporal_coverage" in validation_data:
                temporal_elements.extend(validation_data["temporal_coverage"])

            if "time_period_start" in validation_data or "time_period_end" in validation_data:
                temporal_elements.append("specified_period")

            # Calculate diversity score
            unique_periods = len(set(temporal_elements))
            if unique_periods >= 5:
                return 0.9
            elif unique_periods >= 3:
                return 0.7
            elif unique_periods >= 2:
                return 0.5
            elif unique_periods >= 1:
                return 0.3
            else:
                return 0.1

        except Exception:
            return 0.5

    def _assess_methodological_diversity(self, validation_data: Dict[str, Any]) -> float:
        """Assess methodological diversity."""
        try:
            methods = []

            # Check validation methods used
            if "validation_methods" in validation_data:
                methods.extend(validation_data["validation_methods"])

            # Check statistical methods
            if "statistical_results" in validation_data:
                stat_results = validation_data["statistical_results"]
                if isinstance(stat_results, list):
                    methods.extend([r.get("statistic_name", "unknown") for r in stat_results])

            # Calculate diversity score
            unique_methods = len(set(methods))
            if unique_methods >= 8:
                return 0.9
            elif unique_methods >= 5:
                return 0.7
            elif unique_methods >= 3:
                return 0.5
            elif unique_methods >= 2:
                return 0.3
            else:
                return 0.1

        except Exception:
            return 0.5

    def _assess_stakeholder_diversity(self, validation_data: Dict[str, Any]) -> float:
        """Assess stakeholder diversity."""
        try:
            stakeholders = []

            # Check consensus participants
            if "consensus_results" in validation_data:
                consensus = validation_data["consensus_results"]
                if "participating_agents" in consensus:
                    stakeholders.extend(consensus["participating_agents"])

            # Simplified diversity assessment
            unique_stakeholders = len(set(stakeholders))
            if unique_stakeholders >= 5:
                return 0.9
            elif unique_stakeholders >= 3:
                return 0.7
            elif unique_stakeholders >= 2:
                return 0.5
            else:
                return 0.3

        except Exception:
            return 0.5

    def _assess_data_source_diversity(self, validation_data: Dict[str, Any]) -> float:
        """Assess data source diversity."""
        try:
            sources = []

            # Check data sources mentioned
            if "data_sources" in validation_data:
                sources.extend(validation_data["data_sources"])

            # Check dataset references
            if "dataset_ids" in validation_data:
                sources.extend(validation_data["dataset_ids"])

            # Calculate diversity score
            unique_sources = len(set(sources))
            if unique_sources >= 5:
                return 0.9
            elif unique_sources >= 3:
                return 0.7
            elif unique_sources >= 2:
                return 0.5
            elif unique_sources >= 1:
                return 0.3
            else:
                return 0.1

        except Exception:
            return 0.5

    def _detect_biases(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential biases in validation data."""
        biases = []
        overall_severity = 0.0

        try:
            # Extract text content for bias detection
            text_content = ""
            for key, value in validation_data.items():
                if isinstance(value, str):
                    text_content += value + " "
                elif isinstance(value, list):
                    text_content += " ".join(str(v) for v in value) + " "

            text_content = text_content.lower()

            # Check for each bias type
            for bias_type, bias_info in self.bias_patterns.items():
                indicators_found = []
                for indicator in bias_info["indicators"]:
                    if indicator in text_content:
                        indicators_found.append(indicator)

                if indicators_found:
                    severity = min(1.0, len(indicators_found) * 0.3)
                    overall_severity += severity

                    biases.append({
                        "bias_type": bias_type.value,
                        "severity": severity,
                        "description": bias_info["description"],
                        "indicators_found": indicators_found,
                        "recommendations": self._get_bias_recommendations(bias_type)
                    })

        except Exception as e:
            self.logger.error(f"Bias detection failed: {e}")

        return {
            "biases": biases,
            "overall_severity": min(1.0, overall_severity)
        }

    def _get_bias_recommendations(self, bias_type: BiasType) -> List[str]:
        """Get recommendations for addressing specific bias types."""
        recommendations = {
            BiasType.CULTURAL_BIAS: [
                "Include perspectives from multiple cultural traditions",
                "Consult with diverse cultural experts",
                "Consider cross-cultural validation approaches"
            ],
            BiasType.GEOGRAPHICAL_BIAS: [
                "Expand geographical scope of data collection",
                "Include diverse regional perspectives",
                "Consider global applicability of findings"
            ],
            BiasType.TEMPORAL_BIAS: [
                "Include historical and contemporary perspectives",
                "Consider temporal trends in analysis",
                "Validate across different time periods"
            ],
            BiasType.METHODOLOGICAL_BIAS: [
                "Use multiple analytical methods",
                "Triangulate findings across different approaches",
                "Consider alternative methodological frameworks"
            ],
            BiasType.INTERPRETATION_BIAS: [
                "Use blind review processes",
                "Include diverse interpretation perspectives",
                "Document potential biases in interpretation"
            ],
            BiasType.REPRESENTATION_BIAS: [
                "Ensure diverse stakeholder representation",
                "Include underrepresented groups in validation",
                "Address representation gaps in methodology"
            ]
        }

        return recommendations.get(bias_type, ["Address identified bias through methodological improvements"])

    def _generate_diversity_recommendations(
        self,
        diversity_scores: Dict[str, float],
        identified_biases: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate diversity improvement recommendations."""
        recommendations = []

        try:
            # Recommendations based on low diversity scores
            for dimension, score in diversity_scores.items():
                if score < 0.5:
                    if dimension == "cultural_diversity":
                        recommendations.append("Increase cultural diversity in data and perspectives")
                    elif dimension == "geographical_diversity":
                        recommendations.append("Expand geographical coverage and regional perspectives")
                    elif dimension == "temporal_diversity":
                        recommendations.append("Include broader temporal scope in analysis")
                    elif dimension == "methodological_diversity":
                        recommendations.append("Use more diverse methodological approaches")
                    elif dimension == "stakeholder_diversity":
                        recommendations.append("Include more diverse stakeholders in validation")
                    elif dimension == "data_source_diversity":
                        recommendations.append("Diversify data sources and collection methods")

            # Recommendations based on identified biases
            for bias in identified_biases:
                recommendations.extend(bias.get("recommendations", []))

            # General recommendations
            if not recommendations:
                recommendations.append("Diversity assessment shows good coverage across dimensions")
            else:
                recommendations.append("Regular diversity audits recommended for ongoing validation")

        except Exception as e:
            self.logger.error(f"Diversity recommendations generation failed: {e}")
            recommendations.append("Manual diversity assessment recommended")

        return recommendations

    def _assess_overall_diversity_bias(self, diversity_score: float, bias_score: float) -> Dict[str, Any]:
        """Assess overall diversity and bias situation."""
        assessment = {
            "overall_rating": "unknown",
            "diversity_level": "unknown",
            "bias_level": "unknown",
            "summary": ""
        }

        try:
            # Assess diversity level
            if diversity_score >= 0.8:
                assessment["diversity_level"] = "excellent"
            elif diversity_score >= 0.6:
                assessment["diversity_level"] = "good"
            elif diversity_score >= 0.4:
                assessment["diversity_level"] = "moderate"
            elif diversity_score >= 0.2:
                assessment["diversity_level"] = "limited"
            else:
                assessment["diversity_level"] = "poor"

            # Assess bias level
            if bias_score >= 0.8:
                assessment["bias_level"] = "minimal"
            elif bias_score >= 0.6:
                assessment["bias_level"] = "low"
            elif bias_score >= 0.4:
                assessment["bias_level"] = "moderate"
            elif bias_score >= 0.2:
                assessment["bias_level"] = "high"
            else:
                assessment["bias_level"] = "severe"

            # Overall rating
            combined_score = (diversity_score + bias_score) / 2
            if combined_score >= 0.8:
                assessment["overall_rating"] = "excellent"
                assessment["summary"] = "Strong diversity with minimal bias concerns"
            elif combined_score >= 0.6:
                assessment["overall_rating"] = "good"
                assessment["summary"] = "Good diversity with manageable bias considerations"
            elif combined_score >= 0.4:
                assessment["overall_rating"] = "moderate"
                assessment["summary"] = "Moderate diversity with some bias concerns to address"
            else:
                assessment["overall_rating"] = "needs_improvement"
                assessment["summary"] = "Diversity and bias issues require attention"

        except Exception as e:
            self.logger.error(f"Overall diversity-bias assessment failed: {e}")

        return assessment