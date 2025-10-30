"""
A2A World Platform - Text Processor

Advanced NLP processing for mythological and cultural text data.
Provides tokenization, entity recognition, sentiment analysis, and cross-referencing
with geospatial data for cultural/mythological narratives.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import uuid

try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .quality_checker import QualityChecker, QualityReport
from ..core.messaging import get_nats_client, AgentMessaging, AgentMessage


@dataclass
class TextProcessingResult:
    """Result of text processing operations."""

    success: bool
    text_data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    quality_report: Optional[QualityReport] = None
    entities: List[Dict[str, Any]] = None
    sentiment_analysis: Dict[str, Any] = None
    cross_references: List[Dict[str, Any]] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class MythologicalEntity:
    """Represents a mythological entity extracted from text."""

    name: str
    entity_type: str  # 'deity', 'hero', 'monster', 'location', 'artifact', etc.
    confidence: float
    context: str
    start_pos: int
    end_pos: int
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class CrossReference:
    """Cross-reference between text content and geospatial data."""

    text_entity: str
    geo_feature_id: str
    reference_type: str  # 'location_mention', 'cultural_site', 'mythical_place'
    confidence: float
    context: str
    coordinates: Optional[Tuple[float, float]] = None


class TextProcessor:
    """
    Advanced text processor for mythological and cultural data.

    Features:
    - NLP processing (tokenization, POS tagging, dependency parsing)
    - Named entity recognition for mythological entities
    - Sentiment analysis for cultural texts
    - Cross-referencing with geospatial data
    - Async processing with NATS messaging integration
    - Quality assessment and validation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_checker = QualityChecker()

        # Initialize NLP components
        self.nlp = None
        self.sentiment_analyzer = None
        self.embedding_model = None
        self.nats_client = None
        self.messaging = None

        # Mythological entity patterns
        self.mythological_patterns = self._load_mythological_patterns()

        # Cultural/geographical knowledge base
        self.cultural_kb = self._load_cultural_knowledge_base()

    async def initialize(self) -> bool:
        """Initialize NLP models and NATS connection."""
        try:
            # Initialize spaCy
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Download model if not available
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                    self.nlp = spacy.load("en_core_web_sm")
            else:
                self.nlp = English()
                self.nlp.add_pipe("sentencizer")

            # Initialize NLTK
            if NLTK_AVAILABLE:
                try:
                    nltk.data.find('vader_lexicon')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                self.sentiment_analyzer = SentimentIntensityAnalyzer()

            # Initialize sentence transformers for embeddings
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Initialize NATS client
            self.nats_client = await get_nats_client()
            self.messaging = AgentMessaging(self.nats_client, "text_processor")

            self.logger.info("Text processor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize text processor: {e}")
            return False

    async def process_file(
        self,
        file_path: str,
        extract_entities: bool = True,
        analyze_sentiment: bool = True,
        cross_reference_geo: bool = True,
        generate_quality_report: bool = True
    ) -> TextProcessingResult:
        """
        Process a text file with mythological/cultural content.

        Args:
            file_path: Path to text file
            extract_entities: Whether to extract mythological entities
            analyze_sentiment: Whether to perform sentiment analysis
            cross_reference_geo: Whether to cross-reference with geospatial data
            generate_quality_report: Whether to generate quality report

        Returns:
            TextProcessingResult with processed data and analysis
        """
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                return TextProcessingResult(
                    success=False,
                    text_data=[],
                    metadata={},
                    errors=[f"File not found: {file_path}"]
                )

            self.logger.info(f"Processing text file: {file_path}")

            # Read text content
            try:
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try alternative encodings
                for encoding in ['latin-1', 'cp1252', 'utf-8-sig']:
                    try:
                        with open(file_path_obj, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return TextProcessingResult(
                        success=False,
                        text_data=[],
                        metadata={'file_path': str(file_path_obj)},
                        errors=["Could not decode file with any supported encoding"]
                    )

            # Process the text content
            result = await self._process_text_content(
                content, extract_entities, analyze_sentiment,
                cross_reference_geo, generate_quality_report
            )

            # Add file metadata
            result.metadata.update({
                'file_path': str(file_path_obj),
                'file_size': file_path_obj.stat().st_size,
                'processed_at': datetime.utcnow().isoformat(),
                'processor': 'TextProcessor'
            })

            return result

        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {e}")
            return TextProcessingResult(
                success=False,
                text_data=[],
                metadata={'file_path': file_path},
                errors=[f"Processing failed: {str(e)}"]
            )

    async def process_text_string(
        self,
        text: str,
        source_id: Optional[str] = None,
        extract_entities: bool = True,
        analyze_sentiment: bool = True,
        cross_reference_geo: bool = True,
        generate_quality_report: bool = True
    ) -> TextProcessingResult:
        """
        Process text content from string.

        Args:
            text: Text content to process
            source_id: Optional identifier for the text source
            extract_entities: Whether to extract mythological entities
            analyze_sentiment: Whether to perform sentiment analysis
            cross_reference_geo: Whether to cross-reference with geospatial data
            generate_quality_report: Whether to generate quality report

        Returns:
            TextProcessingResult with processed data and analysis
        """
        try:
            result = await self._process_text_content(
                text, extract_entities, analyze_sentiment,
                cross_reference_geo, generate_quality_report
            )

            result.metadata.update({
                'source': 'string',
                'source_id': source_id,
                'processed_at': datetime.utcnow().isoformat(),
                'processor': 'TextProcessor'
            })

            return result

        except Exception as e:
            self.logger.error(f"Error processing text string: {e}")
            return TextProcessingResult(
                success=False,
                text_data=[],
                metadata={'source': 'string', 'source_id': source_id},
                errors=[f"Processing failed: {str(e)}"]
            )

    async def _process_text_content(
        self,
        content: str,
        extract_entities: bool,
        analyze_sentiment: bool,
        cross_reference_geo: bool,
        generate_quality_report: bool
    ) -> TextProcessingResult:
        """Process text content with NLP analysis."""

        text_data = []
        entities = []
        sentiment_analysis = {}
        cross_references = []
        warnings = []
        errors = []

        try:
            # Basic text segmentation
            sentences = self._segment_text(content)
            tokens = self._tokenize_text(content)

            # Create base text data structure
            text_data = [{
                'content': content,
                'sentences': sentences,
                'tokens': tokens,
                'word_count': len(tokens),
                'sentence_count': len(sentences),
                'character_count': len(content)
            }]

            # Extract mythological entities
            if extract_entities:
                entities = await self._extract_mythological_entities(content)
                text_data[0]['entities'] = [entity.__dict__ for entity in entities]

            # Perform sentiment analysis
            if analyze_sentiment:
                sentiment_analysis = self._analyze_sentiment(content)
                text_data[0]['sentiment'] = sentiment_analysis

            # Cross-reference with geospatial data
            if cross_reference_geo and entities:
                cross_references = await self._cross_reference_geospatial(entities, content)
                text_data[0]['cross_references'] = [ref.__dict__ for ref in cross_references]

            # Generate quality report
            quality_report = None
            if generate_quality_report:
                # Convert to feature-like format for quality checking
                features = self._convert_to_features(text_data, entities)
                quality_report = self.quality_checker.check_dataset_quality(
                    features,
                    "Mythological Text Data"
                )

            return TextProcessingResult(
                success=True,
                text_data=text_data,
                metadata={},
                quality_report=quality_report,
                entities=[entity.__dict__ for entity in entities],
                sentiment_analysis=sentiment_analysis,
                cross_references=[ref.__dict__ for ref in cross_references],
                warnings=warnings,
                errors=errors
            )

        except Exception as e:
            return TextProcessingResult(
                success=False,
                text_data=[],
                metadata={},
                errors=[f"Text content processing failed: {str(e)}"]
            )

    def _segment_text(self, text: str) -> List[str]:
        """Segment text into sentences."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass

        # Fallback sentence segmentation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except:
                pass

        if self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc]

        # Fallback tokenization
        return re.findall(r'\b\w+\b', text)

    async def _extract_mythological_entities(self, text: str) -> List[MythologicalEntity]:
        """Extract mythological entities from text."""
        entities = []

        # Use spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)

            for ent in doc.ents:
                # Check if entity matches mythological patterns
                entity_type = self._classify_entity_type(ent.text, ent.label_)
                if entity_type:
                    entity = MythologicalEntity(
                        name=ent.text,
                        entity_type=entity_type,
                        confidence=0.8,  # spaCy confidence
                        context=text[max(0, ent.start_char-50):ent.end_char+50],
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        properties={
                            'spacy_label': ent.label_,
                            'source': 'spacy_ner'
                        }
                    )
                    entities.append(entity)

        # Use pattern-based extraction
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)

        # Remove duplicates and merge similar entities
        entities = self._deduplicate_entities(entities)

        return entities

    def _classify_entity_type(self, entity_text: str, spacy_label: str) -> Optional[str]:
        """Classify entity type based on text and spaCy label."""
        entity_lower = entity_text.lower()

        # Check against known mythological entities
        for pattern_type, patterns in self.mythological_patterns.items():
            for pattern in patterns:
                if re.search(pattern, entity_lower, re.IGNORECASE):
                    return pattern_type

        # Map spaCy labels to mythological types
        label_mapping = {
            'PERSON': 'character',
            'GPE': 'location',
            'LOC': 'location',
            'ORG': 'group'
        }

        return label_mapping.get(spacy_label)

    def _extract_pattern_entities(self, text: str) -> List[MythologicalEntity]:
        """Extract entities using pattern matching."""
        entities = []

        for entity_type, patterns in self.mythological_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = MythologicalEntity(
                        name=match.group(),
                        entity_type=entity_type,
                        confidence=0.6,  # Pattern matching confidence
                        context=text[max(0, match.start()-30):match.end()+30],
                        start_pos=match.start(),
                        end_pos=match.end(),
                        properties={
                            'source': 'pattern_matching',
                            'pattern': pattern
                        }
                    )
                    entities.append(entity)

        return entities

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        sentiment = {}

        # Use NLTK VADER
        if self.sentiment_analyzer:
            scores = self.sentiment_analyzer.polarity_scores(text)
            sentiment['vader'] = scores

        # Use TextBlob
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            sentiment['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }

        # Calculate overall sentiment
        if sentiment:
            polarities = []
            if 'vader' in sentiment:
                polarities.append(sentiment['vader']['compound'])
            if 'textblob' in sentiment:
                polarities.append(sentiment['textblob']['polarity'])

            if polarities:
                sentiment['overall'] = {
                    'polarity': sum(polarities) / len(polarities),
                    'magnitude': max(abs(p) for p in polarities)
                }

        return sentiment

    async def _cross_reference_geospatial(
        self,
        entities: List[MythologicalEntity],
        text: str
    ) -> List[CrossReference]:
        """Cross-reference mythological entities with geospatial data."""
        cross_references = []

        if not self.messaging:
            return cross_references

        try:
            # Request geospatial data for location entities
            location_entities = [e for e in entities if e.entity_type == 'location']

            for entity in location_entities:
                # Query geospatial service via NATS
                query_message = AgentMessage.create(
                    sender_id="text_processor",
                    message_type="geospatial_query",
                    payload={
                        "entity_name": entity.name,
                        "entity_type": "mythological_location",
                        "context": entity.context
                    }
                )

                try:
                    response = await self.nats_client.request(
                        "agents.geospatial.query",
                        query_message,
                        timeout=10.0
                    )

                    if response and response.payload.get('features'):
                        for feature in response.payload['features']:
                            cross_ref = CrossReference(
                                text_entity=entity.name,
                                geo_feature_id=feature.get('id', str(uuid.uuid4())),
                                reference_type="mythical_location",
                                confidence=0.7,
                                context=entity.context,
                                coordinates=feature.get('coordinates')
                            )
                            cross_references.append(cross_ref)

                except Exception as e:
                    self.logger.warning(f"Geospatial query failed for {entity.name}: {e}")

        except Exception as e:
            self.logger.error(f"Cross-referencing failed: {e}")

        return cross_references

    def _load_mythological_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for mythological entity recognition."""
        return {
            'deity': [
                r'\b(Zeus|Jupiter|Hera|Juno|Poseidon|Neptune|Athena|Minerva|Apollo|Hermes|Mercury|Ares|Mars|Aphrodite|Venus|Hades|Pluto|Hephaestus|Vulcan|Dionysus|Bacchus|Demeter|Ceres|Artemis|Diana|Hestia|Vesta)\b',
                r'\b(Odin|Thor|Loki|Freya|Frey|Heimdall|Balder|Frigg|Tyr|Odin)\b',
                r'\b(Ra|Osiris|Isis|Horus|Anubis|Thoth|Set|Bastet|Hathor)\b',
                r'\b(Indra|Agni|Vayu|Varuna|Surya|Chandra|Yama|Kubera)\b'
            ],
            'hero': [
                r'\b(Hercules|Heracles|Perseus|Theseus|Jason|Achilles|Odysseus|Ulysses|Ajax|Diomedes)\b',
                r'\b(Beowulf|Sigurd|Siegfried|Roland|Arthur|Lancelot|Gawain|Percival)\b',
                r'\b(Arjuna|Krishna|Bheema|Yudhishthira|Nakula|Sahadeva)\b'
            ],
            'monster': [
                r'\b(Medusa|Hydra|Cerberus|Chimera|Sphinx|Minotaur|Cyclops|Harpy|Siren|Gorgon)\b',
                r'\b(Beowulf|Grendel|Dragon|Fafnir|Jormungandr|Fenrir)\b',
                r'\b(Ravana|Kumbhakarna|Surpanakha|Mahiravana)\b'
            ],
            'location': [
                r'\b(Olympus|Mount Olympus|Asgard|Valhalla|Helheim|Niflheim|Muspelheim|Alfheim|Vanaheim|Jotunheim)\b',
                r'\b(Tartarus|Elysium|Hades|Underworld|Acheron|Styx|Lethe|Cocytus|Phlegethon)\b',
                r'\b(Camelot|Avalon|Tintagel|Glastonbury|Stonehenge)\b'
            ],
            'artifact': [
                r'\b(Mjolnir|Gungnir|Draugr|Gram|Excalibur|Holy Grail|Mead of Poetry)\b',
                r'\b(Trident|Thunderbolt|Aegis|Helm of Darkness|Winged Sandals|Caduceus)\b'
            ]
        }

    def _load_cultural_knowledge_base(self) -> Dict[str, Any]:
        """Load cultural knowledge base for cross-referencing."""
        # This would typically load from a database or file
        return {
            'mythological_sites': {
                'Olympus': {'lat': 40.0, 'lon': 22.0, 'country': 'Greece'},
                'Delphi': {'lat': 38.5, 'lon': 22.5, 'country': 'Greece'},
                'Stonehenge': {'lat': 51.2, 'lon': -1.8, 'country': 'UK'}
            }
        }

    def _deduplicate_entities(self, entities: List[MythologicalEntity]) -> List[MythologicalEntity]:
        """Remove duplicate entities and merge similar ones."""
        seen = set()
        deduplicated = []

        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)

        return deduplicated

    def _convert_to_features(
        self,
        text_data: List[Dict[str, Any]],
        entities: List[MythologicalEntity]
    ) -> List[Dict[str, Any]]:
        """Convert text data to feature format for quality checking."""
        features = []

        for text_item in text_data:
            feature = {
                'name': f"Text Document ({text_item.get('word_count', 0)} words)",
                'description': f"Processed text with {len(entities)} entities",
                'properties': {
                    'word_count': text_item.get('word_count', 0),
                    'sentence_count': text_item.get('sentence_count', 0),
                    'entity_count': len(entities)
                }
            }
            features.append(feature)

        return features

    async def publish_processing_results(self, result: TextProcessingResult) -> None:
        """Publish processing results via NATS."""
        if not self.messaging:
            return

        try:
            message = AgentMessage.create(
                sender_id="text_processor",
                message_type="text_processing_complete",
                payload={
                    'result': {
                        'success': result.success,
                        'text_count': len(result.text_data),
                        'entity_count': len(result.entities),
                        'cross_reference_count': len(result.cross_references) if result.cross_references else 0
                    },
                    'metadata': result.metadata
                }
            )

            await self.messaging.publish_discovery(message.payload)

        except Exception as e:
            self.logger.error(f"Failed to publish processing results: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.nats_client:
            await self.nats_client.disconnect()