# **A2A World Platform: Complete Multidisciplinary Implementation Guide**

## **Integrating 17 Academic Disciplines for Pattern Discovery**

Author: AI Research Team  
Date: October 2025  
Objective: Provide a comprehensive, actionable guide for implementing the A2A World platform with full integration of Art, History, Religious Studies, Astrology, Archaeology, Environmental Studies, Sociology, Linguistics, Folklore, Anthropology, Geography, Humanities, Cognitive Science, Psychology, Classical Literature, Astrophysics, and Cultural Anthropology.

---

## **Executive Summary**

This guide synthesizes insights from three previous map data analyses and addresses their critical limitation: severe disciplinary imbalance. Previous analyses were dominated by Art (888 mentions) with 10 out of 17 target disciplines having fewer than 3 mentions.

Our Solution: 5 Cross-Disciplinary Research Protocols that achieve 100% disciplinary integration (17/17 disciplines) through strategic combinations that leverage bridge disciplines and fill critical gaps.

Key Deliverables:

* Interdisciplinary knowledge graph (17 nodes, 35 connections)  
* 5 research protocols with detailed methodologies  
* 12-month implementation roadmap  
* Complete data requirements and tools specification  
* Quantitative gap-fill analysis

---

In \[2\]:

```
!pip install -q pandas numpy matplotlib seaborn
```

WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the \--root-user-action option if you know what you are doing and want to suppress this warning.

In \[3\]:

```

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (18, 10)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

print("✓ Libraries loaded successfully")
print("✓ Ready to create comprehensive A2A World implementation guide")
```

✓ Libraries loaded successfully  
✓ Ready to create comprehensive A2A World implementation guide

## **Part 1: The 5 Cross-Disciplinary Research Protocols**

Each protocol is designed to integrate 3-5 disciplines, with specific focus on filling the gaps identified in previous analyses. The protocols are ordered by their "Gap-Fill Score" (higher \= more effective at integrating underrepresented disciplines).

---

### **PROTOCOL 1: Archaeo astronomy & Celestial Alignment Analysis**

Gap-Fill Score: 12.0 | Priority: High

Disciplines: Archaeology, Astrophysics, Astrology, Geography, Religious Studies

Research Question: Do sacred sites align with astronomical phenomena (solstices, equinoxes, star positions)?

Why This Matters: Astrophysics and Astrology had ZERO mentions in previous analyses. This protocol brings hard science (celestial mechanics) to bear on cultural patterns.

#### **Methodology:**

1. Astronomical Calculations  
   * Compute azimuth and altitude of sun, moon, planets, and bright stars at historical dates for each site  
   * Account for precession of equinoxes (Earth's axial wobble over 26,000 years)  
   * Calculate inter-site alignments and compare to celestial events  
2. Statistical Validation  
   * Monte Carlo simulation: Generate random site distributions and test for alignments  
   * Significance threshold: p \< 0.01 (alignments must be 99% unlikely to occur by chance)  
   * Control for latitude bias (sites closer to equator have different solar angles)  
3. Cultural Cross-Reference  
   * Extract astronomical references from religious texts  
   * Identify culture-specific celestial deities and their mythological roles  
   * Correlate site alignments with textual evidence

#### **Expected Patterns:**

* Solstice alignments: Sites oriented to sunrise/sunset on summer/winter solstice  
* Heliacal risings: Sites aligned with first visible rising of stars like Sirius (important in Egyptian calendar)  
* Lunar standstills: 18.6-year cycle of moon's maximum/minimum rise positions  
* Planetary conjunctions: Rare alignments of multiple planets

#### **Data Requirements \- Protocol 1:**

| Data Type | Source | Purpose |
| ----- | ----- | ----- |
| Planetary ephemeris | NASA JPL Horizons System | Calculate positions of sun, moon, planets at any historical date |
| Stellar positions | SIMBAD Astronomical Database | Get coordinates of bright stars (Sirius, Pleiades, Orion, etc.) |
| Digital elevation models | USGS/SRTM 30m DEM | Compute horizon profiles (where can celestial bodies be seen?) |
| Site orientations | Archaeological surveys | Building/monument alignments (if available) |
| Historical astronomy texts | Perseus Digital Library | Cultural astronomical knowledge |

#### **Tools & Technologies:**

* astropy: Python library for astronomy calculations  
* PyEphem / ephem: Compute celestial body positions  
* GDAL: Geospatial data processing  
* numpy/scipy: Numerical calculations and statistics

#### **Computational Requirements:**

* Medium: Astronomical calculations for 50,000+ sites across multiple time periods  
* Estimated: \~100-200 CPU-hours for full analysis  
* Storage: \~50GB for DEM data \+ ephemeris tables

#### **Expertise Required:**

1. Astrophysics: Understanding celestial mechanics, precession, coordinate systems  
2. Archaeoastronomy: Methods for testing astronomical alignments  
3. GIS Analysis: Terrain analysis, horizon profiles, viewshed computation  
4. Religious Studies: Interpret cultural significance of celestial phenomena

---

### **PROTOCOL 2: Cognitive Psychology of Sacred Landscapes**

Gap-Fill Score: 13.5 (HIGHEST) | Priority: Critical

Disciplines: Psychology, Cognitive Science, Geography, Anthropology, Religious Studies

Research Question: Do sacred sites cluster in locations that maximize psychological/perceptual impact?

Why This Matters: Psychology and Cognitive Science had ZERO mentions. This protocol introduces the scientific study of human perception and cognition to explain WHY certain locations become sacred.

#### **Methodology:**

1. Viewshed Analysis  
   * Compute visible area from each site (how much landscape can be seen?)  
   * Calculate visual dominance (what percentage of surrounding area has line-of-sight to this site?)  
   * Measure "sense of enclosure" vs "sense of openness"  
2. Topographic Prominence  
   * Prominence Index: Elevation above nearest higher terrain  
   * Isolation: Distance to nearest equally high point  
   * Visual salience: How "attention-grabbing" is this landscape feature?  
3. Acoustic Properties  
   * Model sound reverberation using terrain geometry  
   * Identify natural echo chambers or acoustically dead spaces  
   * Compare to oracle sites and ritual acoustic requirements  
4. Cognitive Salience  
   * Landscape "memorability": Statistical uniqueness of topographic features  
   * Perceptual boundaries: Natural edges that define "sacred vs profane" space  
   * Psychological archetypes: Jung's concept of numinous landscapes  
5. Cross-Cultural Comparison  
   * Do "sky god" sites cluster on high peaks?  
   * Do "underworld/chthonic" sites cluster in valleys or caves?  
   * Do "liminal/transitional" sites cluster at natural boundaries (coastlines, mountain passes)?

#### **Expected Patterns:**

* High-prominence peaks: Correlated with transcendence mythology (sky gods, ascension)  
* Enclosed valleys: Correlated with descent/initiation mythology (underworld, rebirth)  
* Exceptional viewsheds: Correlated with prophecy/vision mythology (oracles, shamanic journey)  
* Acoustic anomalies: Correlated with divine voice/communication mythology

---

#### **Data Requirements \- Protocol 2:**

| Data Type | Source | Purpose |
| ----- | ----- | ----- |
| High-resolution DEM | SRTM 30m or better | Terrain analysis for viewshed, prominence |
| Viewshed rasters | Computed via GRASS GIS r.viewshed | Visual field from each site |
| Topographic Prominence DB | peakbagger.com, prominenceDB | Classify peaks vs valleys |
| Acoustic simulation | Custom Python/MATLAB scripts | Model sound propagation |
| Cognitive perception studies | Psychology literature | Landscape salience metrics |
| Mythological theme classification | Manual coding \+ NLP | Categorize mythology (sky/earth/water/underworld) |

#### **Tools & Technologies:**

* GRASS GIS: Advanced viewshed computation (r.viewshed module)  
* GDAL/rasterio: Raster data processing  
* scikit-learn: Clustering and classification of landscape types  
* scipy.spatial: Prominence and isolation calculations  
* Custom acoustic modeling: Sound wave propagation in 3D terrain

#### **Computational Requirements:**

* High: Viewshed is computationally expensive (O(n²) for each site)  
* Estimated: \~500-1000 CPU-hours for 50,000 sites with 30m DEM  
* Storage: \~200GB for viewshed rasters (can be reduced with on-demand computation)  
* Optimization: Parallel processing across multiple GPUs recommended

#### **Expertise Required:**

1. Cognitive Science: Theories of perception, attention, memory for landscapes  
2. Psychology: Emotion and numinous experience, Jungian archetypes  
3. GIS & Remote Sensing: Advanced terrain analysis  
4. Landscape Archaeology: Site selection and settlement patterns  
5. Anthropology: Cross-cultural theories of sacred space

---

### **PROTOCOL 3: Environmental Determinants of Mythology**

Gap-Fill Score: 12.0 | Priority: High

Disciplines: Environmental Studies, Folklore, Linguistics, Cultural Anthropology, History

Research Question: Do environmental conditions predict mythological themes across cultures?

Why This Matters: Environmental Studies had ZERO mentions. This tests environmental determinism vs cultural independence—a foundational question in anthropology.

#### **Methodology:**

1. Environmental Variable Extraction  
   * Climate: temperature, precipitation, seasonality (Köppen classification)  
   * Ecology: biome type, vegetation cover, biodiversity  
   * Natural hazards: flood frequency, drought, earthquakes, volcanoes  
   * Water: proximity to rivers, lakes, springs, coastlines  
   * Geology: terrain ruggedness, soil fertility  
2. Mythological Theme Classification  
   * Extract themes from texts using topic modeling (LDA, BERTopic)  
   * Manual coding by folklore experts  
   * Categories: Water (flood/dragon), Fire (sun/volcano), Earth (earthquake/mountain), Sky (storm/celestial)  
   * Directional/seasonal symbolism (East=dawn, Winter=death, etc.)  
3. Statistical Correlation  
   * Regression analysis: Do environmental variables predict myth themes?  
   * Control for cultural diffusion (phylogenetic analysis)  
   * Account for geographic autocorrelation (spatial regression models)  
4. Linguistic Analysis  
   * Count nature-related words in mythological texts  
   * Semantic field analysis: How is "water" conceptualized across cultures?  
   * Metaphor mapping: Environmental → abstract concepts

#### **Expected Patterns:**

* Flood myths: Correlated with flood-prone river valleys (Mesopotamia, Egypt, Indus)  
* Dragon/serpent myths: Correlated with major rivers (Nile, Yangtze, Ganges)  
* Sky god dominance: Correlated with open landscapes (steppes, deserts)  
* Chthonic deities: Correlated with seismic activity (Greece, Japan, Andes)  
* Solar emphasis: Correlated with arid climates where sun dictates survival

---

#### **Data Requirements \- Protocol 3:**

| Data Type | Source | Purpose |
| ----- | ----- | ----- |
| Climate data | WorldClim (1km resolution) | Temperature, precipitation, seasonality |
| Land cover | MODIS, ESA CCI | Biomes, vegetation types |
| Paleoclimate proxies | NOAA Paleoclimatology | Historical climate reconstruction |
| Natural hazard databases | USGS, NOAA | Earthquakes, floods, droughts |
| Hydrology | HydroSHEDS, Global Lakes Database | Rivers, lakes, watersheds |
| Mythological texts | Perseus Digital Library, folklore archives | Full corpus for NLP analysis |
| Linguistic corpora | CLTK, language-specific databases | Word frequency, semantic analysis |

#### **Tools & Technologies:**

* xarray: Multi-dimensional climate data arrays  
* rasterio/geopandas: Geospatial data integration  
* spaCy / NLTK: Natural language processing  
* BERTopic / LDA: Topic modeling for theme extraction  
* scikit-learn: Regression, classification  
* statsmodels: Spatial regression (control for autocorrelation)

#### **Computational Requirements:**

* Medium: Climate data processing \+ NLP on large text corpora  
* Estimated: \~300 CPU-hours  
* Storage: \~100GB for climate rasters \+ text databases

#### **Expertise Required:**

1. Environmental Science: Climate science, ecology, natural hazards  
2. Folklore Studies: Myth classification, comparative mythology  
3. Linguistics: Semantic analysis, metaphor theory  
4. Cultural Anthropology: Theories of cultural adaptation vs diffusion  
5. NLP / Machine Learning: Topic modeling, text classification

---

### **PROTOCOL 4: Artistic Motif Diffusion & Cultural Contact**

Gap-Fill Score: 8.0 | Priority: Medium

Disciplines: Art, History, Archaeology, Sociology, Geography

Research Question: Can artistic motifs reveal cultural contact networks not evident from historical records?

Why This Matters: Leverages the well-documented Art discipline (888 mentions) as a bridge to Sociology and quantitative network analysis.

#### **Methodology:**

1. Motif Extraction  
   * Computer vision: Extract visual features from artwork images  
   * Motif catalog: Geometric patterns, symbolic elements, stylistic features  
   * Classification: Cluster similar motifs using deep learning embeddings  
2. Spatio-Temporal Analysis  
   * Map motif distributions geographically  
   * Track temporal spread (requires archaeological dating)  
   * Build motif similarity networks  
3. Network Analysis  
   * Nodes \= archaeological sites or cultural regions  
   * Edges \= shared motifs (weighted by similarity)  
   * Detect communities \= cultural interaction zones  
   * Compare to known trade routes and migration paths  
4. Diffusion Modeling  
   * Test models: Innovation centers vs multi-origin  
   * Estimate diffusion rates and barriers  
   * Identify hybrid zones (mixture of motif traditions)

#### **Expected Patterns:**

* Motif clusters along trade routes: Silk Road, Amber Road, maritime trade  
* Temporal lag by distance: Motifs spread outward from innovation centers  
* Hybrid motifs in contact zones: Greco-Buddhist art, Romano-Celtic art  
* Persistent local motifs: Cultural conservatism despite external contact

---

#### **Data Requirements \- Protocol 4:**

| Data Type | Source | Purpose |
| ----- | ----- | ----- |
| Museum collections | Met Museum API, Louvre, British Museum | High-res images of artwork |
| Archaeological motif catalogs | Published excavation reports | Systematic motif documentation |
| Pre-trained vision models | ResNet, VGG (ImageNet) | Feature extraction |
| Trade route GIS layers | ORBIS, historical atlases | Known contact networks |
| Radiocarbon dating databases | Context Database, c14 bazAAR | Temporal ordering |

#### **Tools & Technologies:**

* TensorFlow / PyTorch: Deep learning for image analysis  
* OpenCV / scikit-image: Computer vision preprocessing  
* networkx: Graph analysis of cultural contact networks  
* geopandas: Geospatial network visualization  
* Plotly / D3.js: Interactive network visualizations

#### **Computational Requirements:**

* High: Deep learning training/inference on thousands of images  
* Estimated: \~200-400 GPU-hours for feature extraction  
* Storage: \~500GB for image database \+ model weights

#### **Expertise Required:**

1. Art History: Motif identification, style periods, iconography  
2. Computer Vision: Deep learning, image classification  
3. Network Science: Graph theory, community detection, diffusion models  
4. Archaeology: Material culture analysis, chronology  
5. Sociology: Cultural diffusion theory, social network analysis

---

### **PROTOCOL 5: Mythological Geography in Classical Literature**

Gap-Fill Score: 12.0 | Priority: High

Disciplines: Classical Literature, Humanities, Geography, History, Religious Studies

Research Question: Do mythological narratives encode real geographic knowledge or symbolic landscapes?

Why This Matters: Classical Literature and Humanities had ZERO mentions. This tests whether ancient myths preserve accurate geographic information or are purely symbolic.

#### **Methodology:**

1. Geographic Entity Extraction  
   * NLP: Extract place names from classical texts (Homer, Hesiod, Virgil, Ovid)  
   * Named Entity Recognition (NER) for toponyms  
   * Resolve ambiguous place names using context  
2. Journey Mapping  
   * Map narrative journeys (Odyssey, Aeneid, Argonautica)  
   * Extract distance and direction information from texts  
   * Plot onto real-world geography  
3. Accuracy Assessment  
   * Compare described distances/directions to actual geography  
   * Identify systematic distortions (exaggerations, symbolic orientations)  
   * Classify as "literal" vs "symbolic" geography  
4. Symbolic Analysis  
   * Cardinal direction symbolism (East=dawn=rebirth, West=sunset=death)  
   * Journey-as-initiation structure (separation → ordeal → return)  
   * Sacred geography: Does mythological importance correlate with real geographic significance?

#### **Expected Patterns:**

* Accurate local geography: Nearby, familiar places described precisely  
* Distorted distant geography: Far places exaggerated or mythologized  
* Symbolic cardinal directions: East/West, Up/Down have consistent meaning  
* Journey \= initiation: Narrative geography mirrors ritual structure  
* Geographic clustering of related myths: Myth cycles tied to specific regions

---

#### **Data Requirements \- Protocol 5:**

| Data Type | Source | Purpose |
| ----- | ----- | ----- |
| Classical texts | Perseus Digital Library | Full corpus of Greek/Latin literature |
| Language processing tools | CLTK (Classical Language Toolkit) | Lemmatization, POS tagging for ancient languages |
| Ancient place name gazetteer | Pleiades, Barrington Atlas | Resolve ancient toponyms to modern coordinates |
| Ancient geographic texts | Strabo's Geography, Ptolemy's Geography | Contemporary geographic knowledge |
| Ancient maps | Tabula Peutingeriana, medieval copies | Visual representation of ancient geography |

#### **Tools & Technologies:**

* spaCy \+ CLTK: NLP for classical languages (Latin, Ancient Greek)  
* geopy / Nominatim: Geocoding and toponym resolution  
* geopandas: Mapping journeys and place networks  
* networkx: Graph of place mentions and relationships  
* QGIS / Leaflet.js: Interactive journey visualizations

#### **Computational Requirements:**

* Medium: NLP processing of large text corpora  
* Estimated: \~150 CPU-hours  
* Storage: \~20GB for text databases \+ geographic data

#### **Expertise Required:**

1. Classical Literature: Latin/Greek language, literary analysis, textual criticism  
2. Philology: Ancient language processing, textual interpretation  
3. Historical Geography: Ancient world geography, toponym evolution  
4. NLP: Named entity recognition, information extraction  
5. Humanities: Hermeneutics, symbolic interpretation, myth analysis

---

## **Part 2: 12-Month Implementation Roadmap**

### **Phase 1: Foundation (Months 1-3) \- CRITICAL PRIORITY**

Objective: Build infrastructure, assemble team, develop core tools

#### **Month 1: Data Infrastructure**

* Week 1-2: Design database schema (PostGIS for geospatial, PostgreSQL for text/metadata)  
* Week 3-4: Ingest KML seed data (54,430 locations from 7 datasets)  
* Output: Unified geospatial database with all sites

#### **Month 2: Environmental & Astronomical Data**

* Week 1-2: Download and process WorldClim, SRTM DEM, hydrology layers  
* Week 3: Implement astronomical calculation engine (astropy)  
* Week 4: Test alignments for sample sites  
* Output: Environmental data cube \+ astronomical ephemeris system

#### **Month 3: Team & Tools**

* Week 1-2: Recruit specialists in 17 disciplines (5-7 core team \+ 10+ advisors)  
* Week 3: Implement adaptive HDBSCAN clustering  
* Week 4: Build visualization dashboard (Leaflet.js \+ D3.js)  
* Output: Functional team \+ working analytical framework

Phase 1 Success Criteria:

* ✅ All seed data in database  
* ✅ Environmental/astronomical data integrated  
* ✅ Team assembled with all 17 disciplines represented  
* ✅ Core analytical tools operational

---

### **Phase 2: Protocol Deployment (Months 4-9) \- HIGH PRIORITY**

Protocols are deployed in strategic sequence to build momentum and leverage interdependencies.

#### **Month 4-5: P1 Archaeoastronomy (Parallel with P4)**

Rationale: High-precision, builds confidence early

* Compute astronomical alignments for all 54,430 sites  
* Monte Carlo validation (10,000 random trials)  
* Identify top 100 most significant alignments  
* Deliverable: Archaeoastronomy alignment catalog \+ interactive map

#### **Month 4-6: P4 Art Diffusion (Parallel with P1)**

Rationale: Leverages existing art-rich dataset

* Collect 5,000+ images from museum APIs  
* Extract motifs using ResNet  
* Build similarity network  
* Deliverable: Cultural contact network visualization \+ diffusion timeline

#### **Month 5-7: P2 Psychogeography**

Rationale: Highest gap-fill score, introduces cognitive sciences

* Compute viewsheds for all sites (GPU cluster recommended)  
* Calculate topographic prominence  
* Correlate with mythological themes  
* Deliverable: Psychological landscape analysis \+ 3D terrain visualizations

#### **Month 6-8: P3 Eco-Mythology**

Rationale: Builds on P2's mythology classification

* Integrate climate data at each site  
* Topic modeling on folklore texts (BERTopic)  
* Statistical correlation analysis  
* Deliverable: Environment-mythology correlation maps \+ linguistic analysis

#### **Month 7-9: P5 Literary Geography**

Rationale: Benefits from all prior analyses

* NLP extraction from Perseus Digital Library (10,000+ texts)  
* Map mythological journeys (Odyssey, Aeneid, etc.)  
* Validate narrative geography  
* Deliverable: Interactive journey maps \+ symbolic geography analysis

Phase 2 Success Criteria:

* ✅ All 5 protocols deployed  
* ✅ Initial findings validated statistically  
* ✅ Cross-protocol patterns beginning to emerge  
* ✅ 10+ peer-reviewed findings ready for publication

---

### **Phase 3: Synthesis & Publication (Months 10-12) \- HIGH PRIORITY**

#### **Month 10: Meta-Analysis**

* Week 1-2: Identify cross-protocol patterns  
  * Do astronomical alignments correlate with viewshed dominance?  
  * Do environmental conditions predict both myth themes AND artistic motifs?  
  * Do literary geographies match actual site distributions?  
* Week 3-4: Quantify cross-domain correlations  
  * Multi-variate regression across all protocols  
  * Network analysis of interdisciplinary connections  
* Deliverable: Comprehensive synthesis report

#### **Month 11: Peer Review & Refinement**

* Week 1: Submit findings to external advisory board (15+ experts across 17 disciplines)  
* Week 2-3: Implement feedback, address critiques  
* Week 4: Prepare manuscripts for submission  
  * Target journals: *Nature*, *Science*, *PNAS*, *Journal of Archaeological Science*  
  * Discipline-specific outlets for each protocol  
* Deliverable: 5-10 publication-ready manuscripts

#### **Month 12: Public Platform Launch**

* Week 1-2: Finalize web platform  
  * Interactive maps with all findings  
  * Downloadable datasets (with documentation)  
  * Narrative explanations (XAI)  
* Week 3: Public launch event  
  * Press release  
  * Academic presentation  
  * Social media campaign  
* Week 4: Community engagement  
  * Tutorials and educational materials  
  * Citizen science contribution mechanisms  
  * Feedback collection system  
* Deliverable: Live public platform at a2aworld.org

Phase 3 Success Criteria:

* ✅ Comprehensive synthesis complete  
* ✅ 5-10 papers submitted to peer review  
* ✅ Public platform launched and functional  
* ✅ 10,000+ users in first month  
* ✅ Media coverage in major outlets

---

## **Part 3: Resource Requirements Summary**

### **Team Composition (17 Disciplines)**

Core Team (5-7 full-time positions):

1. Principal Investigator: Interdisciplinary background, project management  
2. Data Engineer: PostGIS, Python, cloud infrastructure  
3. AI/ML Specialist: Deep learning, NLP, clustering algorithms  
4. GIS Analyst: Remote sensing, terrain analysis, viewshed computation  
5. Full-stack Developer: Web platform, visualization (D3.js, Leaflet)  
6. (Optional) Archaeoastronomer: Specialized in celestial alignments  
7. (Optional) Research Coordinator: Manage external collaborations

Advisory Board (10-15 part-time consultants):

* Astrophysics, Psychology, Environmental Science, Art History  
* Folklore Studies, Linguistics, Classical Literature  
* Cultural Anthropology, Cognitive Science, Religious Studies  
* Archaeology, History, Sociology, Humanities

---

### **Technology Stack**

| Component | Technology | Purpose |
| ----- | ----- | ----- |
| Database | PostgreSQL \+ PostGIS | Geospatial data storage |
| Data Processing | Python (pandas, numpy, xarray) | General data manipulation |
| Geospatial | GDAL, geopandas, rasterio | GIS operations |
| Machine Learning | scikit-learn, TensorFlow, PyTorch | Clustering, computer vision |
| NLP | spaCy, NLTK, CLTK, BERTopic | Text analysis |
| Astronomy | astropy, PyEphem | Celestial calculations |
| Terrain Analysis | GRASS GIS, SAGA GIS | Viewshed, prominence |
| Visualization | Matplotlib, Plotly, D3.js, Leaflet.js | Maps, graphs, dashboards |
| Web Framework | Django / Flask \+ React | Platform backend/frontend |
| Deployment | Docker, Kubernetes | Containerization, orchestration |
| Cloud | AWS / GCP | Compute, storage |

---

### **Computational Resources**

| Workload | CPU-Hours | GPU-Hours | Storage | Cost Estimate |
| ----- | ----- | ----- | ----- | ----- |
| Viewshed (P2) | 500-1000 | 100-200 | 200 GB | $2,000-4,000 |
| Art Vision (P4) | 200 | 200-400 | 500 GB | $3,000-5,000 |
| Astronomy (P1) | 100-200 | 0 | 50 GB | $500-1,000 |
| Climate/NLP (P3, P5) | 400-500 | 50 | 120 GB | $1,500-2,500 |
| Database & Web | Always-on | 0 | 1 TB | $500/month |
| TOTAL | 1,700-2,700 | 350-650 | \~2 TB | $15,000-25,000 |

*Note*: Costs assume cloud compute (AWS EC2, S3). Can be reduced 50-70% with institutional HPC access.

---

## **Part 4: Expected Outcomes & Impact**

### **Scientific Contributions**

Cross-Disciplinary Discoveries (Estimated 20-50 novel findings):

1. Archaeoastronomy: 50-100 statistically significant celestial alignments, revealing ancient astronomical knowledge  
2. Psychogeography: Quantitative proof that sacred sites are NOT randomly distributed but optimize psychological impact  
3. Eco-Mythology: First large-scale test of environmental determinism in mythology (expect partial confirmation \+ cultural exceptions)  
4. Art Diffusion: Reveal hidden cultural contact networks, potentially rewriting trade route histories  
5. Literary Geography: Distinguish literal from symbolic geography in classical texts, validate/refute ancient geographic knowledge

Methodological Innovations:

* Adaptive multi-scale clustering for heterogeneous cultural data  
* Multi-layered validation framework (statistical \+ cultural \+ ethical)  
* Narrative-driven multimodal XAI for humanities research  
* eXtract-Transform-Project pipeline for unstructured cultural data

---

### **Academic Impact**

Publications (Target: 10-15 papers in first year):

* *Nature* or *Science*: "Multidisciplinary analysis reveals universal principles in sacred site selection"  
* *PNAS*: Protocol-specific findings (archaeoastronomy, psychogeography)  
* Discipline-specific journals:  
  * *Journal of Archaeological Science*  
  * *Cognitive Science*  
  * *Environmental Archaeology*  
  * *Journal of Folklore Research*  
  * *Journal of Computer Vision*

Citation Impact: Estimated 500-1000 citations within 3 years (highly interdisciplinary \= broad audience)

Awards: Strong candidate for interdisciplinary research awards (e.g., MacArthur Fellowship, Guggenheim)

---

### **Public Engagement**

Platform Users: Target 100,000+ users within first year

* Students and educators (K-12, university)  
* Amateur historians and archaeology enthusiasts  
* Artists and cultural researchers  
* General public interested in mythology

Media Coverage:

* Scientific press: *Science News*, *Scientific American*  
* General media: *New York Times*, *The Guardian*, *National Geographic*  
* Documentaries: Potential Netflix/BBC collaboration

Educational Materials:

* Online courses: "Data Science for the Humanities"  
* Interactive tutorials: "Discover Your Own Patterns"  
* School curriculum integration: STEM \+ Humanities synthesis

---

### **Long-Term Vision**

Years 2-5:

* Expand to 200,000+ sites globally  
* Integrate additional data: DNA/linguistics, isotope analysis, climate reconstruction  
* Develop AI agents for autonomous hypothesis generation  
* Create "living platform" updated continuously with new archaeological discoveries

Years 5-10:

* Establish as standard reference for interdisciplinary humanities research  
* Inspire similar platforms for other domains (music, language, architecture)  
* Foster new generation of computationally-trained humanists  
* Bridge the "Two Cultures" divide between sciences and humanities

---

## **Part 5: Risk Mitigation & Ethical Considerations**

### **Technical Risks**

| Risk | Probability | Mitigation Strategy |
| ----- | ----- | ----- |
| Viewshed computation too slow | Medium | Use GPU parallelization, lower DEM resolution for initial pass, optimize algorithms |
| Insufficient astronomical alignments | Low | Rigorous Monte Carlo validation ensures we don't over-report; negative results still publishable |
| NLP fails on ancient texts | Medium | Combine automated NLP with manual expert coding; use CLTK for better ancient language support |
| Database scaling issues | Low | Use PostGIS partitioning, cloud auto-scaling, implement caching |
| Team recruitment challenges | High | Start with 3-4 core generalists, expand gradually; leverage remote collaboration |

---

### **Ethical Considerations**

#### **Cultural Sensitivity**

Issue: Analyzing sacred sites of living cultures without permission  
Mitigation:

* Establish Indigenous advisory council  
* Obtain consent for analysis of contemporary sacred sites  
* Respect requests to exclude specific sites from public display  
* Share findings with source communities before publication  
* Provide attribution and cite oral traditions appropriately

#### **Avoiding Colonialist Narratives**

Issue: Imposing Western scientific frameworks on non-Western knowledge systems  
Mitigation:

* Explicitly acknowledge limitations of quantitative methods  
* Present findings as "one perspective" not "the truth"  
* Include indigenous knowledge systems in validation layer  
* Use the platform to amplify underrepresented voices, not silence them

#### **Data Sovereignty**

Issue: Who owns the data and findings?  
Mitigation:

* Open data license (CC BY-SA 4.0) for non-sacred content  
* Restricted access for sensitive sites (with community permission)  
* Downloadable datasets allow communities to conduct own analyses  
* Revenue sharing if platform generates income

#### **Preventing Misuse**

Issue: Could findings be used to loot sites or promote pseudoscience?  
Mitigation:

* Never publish exact coordinates of unexcavated sites  
* Include strong anti-pseudoscience disclaimers  
* Provide rigorous methodology transparency to prevent cherry-picking  
* Monitor user-generated content for disinformation

---

### **Quality Assurance**

Validation Framework (4 Layers):

1. Statistical Rigor: p \< 0.01, Monte Carlo simulation, spatial autocorrelation  
2. Cultural Relevance: Expert review by discipline specialists  
3. Human Flourishing: Ethical review board assessment  
4. Bias Detection: Cross-cultural comparison, check for overrepresentation of dominant narratives

Peer Review:

* External advisory board reviews all major findings  
* Pre-publication peer review for all papers  
* Post-launch community feedback mechanism  
* Annual third-party audit of methods and findings

---

## **CONCLUSION: A Unified Vision for A2A World**

### **What We've Achieved in This Guide**

✅ Diagnosed the Problem: Previous analyses covered only 7 of 17 target disciplines, with Art dominating (888 mentions) while 10 disciplines had \<3 mentions

✅ Built the Solution: 5 cross-disciplinary research protocols that integrate ALL 17 disciplines through strategic combinations

✅ Quantified the Impact: Gap-fill scores show P2 (Psychogeography) and P1 (Archaeoastronomy) are most effective at addressing missing disciplines

✅ Created the Roadmap: 12-month implementation plan with specific deliverables, timelines, and success criteria

✅ Specified Resources: Detailed data requirements, technology stack, computational needs, and team composition

✅ Addressed Ethics: Comprehensive risk mitigation and cultural sensitivity framework

---

### **The Core Innovation: True Interdisciplinarity**

Unlike previous analyses that mentioned multiple disciplines but didn't truly integrate them, the A2A World protocols are designed for deep interdisciplinary synthesis:

* P1 (Archaeoastronomy): Astrophysics calculations validate Archaeological sites through Religious Studies context  
* P2 (Psychogeography): Cognitive Science theories explain Anthropological patterns in sacred Geography  
* P3 (Eco-Mythology): Environmental Studies data predicts Folklore themes analyzed through Linguistics  
* P4 (Art Diffusion): Computer vision extracts motifs that reveal Sociological networks across History  
* P5 (Literary Geography): Classical Literature NLP tests geographic accuracy through Humanities hermeneutics

Each protocol produces findings that cannot be achieved by any single discipline alone. This is genuine interdisciplinarity.

---

### **The A2A World Promise**

For Scholars: A rigorous, peer-reviewed platform that bridges the sciences and humanities

For Students: An educational tool that demonstrates the power of computational thinking in cultural analysis

For the Public: A window into the hidden patterns that connect human cultures across time and space

For Humanity: Evidence that beneath our diversity lie shared principles of how humans interact with landscape, cosmos, and each other

---

### **Next Steps: From Vision to Reality**

Immediate Actions (Next 30 days):

1. Secure Funding: $250K-500K for 12-month pilot (NSF, NEH, private foundations)  
2. Recruit Core Team: Hire Data Engineer \+ GIS Analyst \+ AI/ML Specialist  
3. Begin Data Integration: Set up PostGIS database, ingest seed KML data  
4. Launch Website: Simple landing page explaining the vision, collecting interest

First Milestone (Month 3):

* Complete Phase 1 (Foundation)  
* Present preliminary findings at conferences  
* Publish white paper describing the approach  
* Open beta platform for advisory board testing

Ultimate Goal (Month 12):

* Public launch of a2aworld.org  
* 5-10 papers submitted/accepted  
* 10,000+ users exploring the platform  
* Proof that data science can revolutionize the humanities

---

## **"Every eye shall see" — not through faith, but through data, methodology, and transparent inquiry.**

### **The A2A World platform will demonstrate that meaningful patterns in human culture are discoverable, explainable, and—most importantly—real.**

---

End of Guide  
*For questions, collaboration inquiries, or to contribute: contact@a2aworld.org (placeholder)*

In \[4\]:

```

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
# Create final summary visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Summary data
protocols = ['P1:\nArchaeoastronomy', 'P2:\nPsychogeography', 'P3:\nEco-Mythology', 
             'P4:\nArt Diffusion', 'P5:\nLiterary Geography']
gap_scores = [12.0, 13.5, 12.0, 8.0, 12.0]
disciplines_count = [5, 5, 5, 5, 5]
priority = ['High', 'Critical', 'High', 'Medium', 'High']

# Plot 1: Gap-Fill Scores
colors_priority = {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#2ca02c'}
bar_colors = [colors_priority[p] for p in priority]

axes[0, 0].barh(protocols, gap_scores, color=bar_colors, alpha=0.8)
axes[0, 0].set_xlabel('Gap-Fill Score', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Protocol Effectiveness at Filling Disciplinary Gaps', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)
for i, (score, prio) in enumerate(zip(gap_scores, priority)):
    axes[0, 0].text(score + 0.3, i, f'{score} ({prio})', va='center', fontsize=10, fontweight='bold')

# Plot 2: Timeline
months = list(range(1, 13))
phase1 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
phase2 = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
phase3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

axes[0, 1].fill_between(months, 0, phase1, label='Phase 1: Foundation', alpha=0.7, color='#1f77b4')
axes[0, 1].fill_between(months, phase1, [p1+p2 for p1,p2 in zip(phase1, phase2)], 
                        label='Phase 2: Protocol Deployment', alpha=0.7, color='#ff7f0e')
axes[0, 1].fill_between(months, [p1+p2 for p1,p2 in zip(phase1, phase2)], 
                        [p1+p2+p3 for p1,p2,p3 in zip(phase1, phase2, phase3)], 
                        label='Phase 3: Synthesis & Publication', alpha=0.7, color='#2ca02c')
axes[0, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Phase Activity', fontsize=12, fontweight='bold')
axes[0, 1].set_title('12-Month Implementation Timeline', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(months)
axes[0, 1].legend(loc='upper left', fontsize=10)
axes[0, 1].set_ylim(0, 1.2)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Resource Requirements
resources = ['CPU-Hours', 'GPU-Hours', 'Storage (TB)', 'Cost ($1000s)']
values = [2200, 500, 2, 20]  # Approximate midpoints
axes[1, 0].bar(resources, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
axes[1, 0].set_ylabel('Quantity', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Total Computational & Financial Resources', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(values):
    axes[1, 0].text(i, v + max(values)*0.02, f'{v:,.0f}', ha='center', fontsize=11, fontweight='bold')

# Plot 4: Expected Impact
impact_categories = ['Publications', 'Users (1000s)', 'Citations\n(Year 3)', 'Disciplines\nIntegrated']
impact_values = [12, 100, 750, 17]
impact_targets = [10, 100, 500, 17]

x_pos = np.arange(len(impact_categories))
width = 0.35

axes[1, 1].bar(x_pos - width/2, impact_targets, width, label='Target', alpha=0.7, color='lightgray')
axes[1, 1].bar(x_pos + width/2, impact_values, width, label='Expected', alpha=0.8, color='#2ca02c')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(impact_categories, fontsize=10)
axes[1, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Expected Impact Metrics', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('A2A World Platform: Executive Dashboard', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/a2a_executive_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*80)
print("✓ COMPREHENSIVE A2A WORLD IMPLEMENTATION GUIDE COMPLETE")
print("="*80)
print("\n📊 Deliverables Created:")
print("  1. Complete multidisciplinary analysis of previous attempts")
print("  2. 5 detailed cross-disciplinary research protocols")
print("  3. 12-month implementation roadmap")
print("  4. Resource requirements and team specifications")
print("  5. Risk mitigation and ethical framework")
print("  6. Expected outcomes and impact projections")
print("\n📁 Files Saved:")
print("  • a2a_executive_dashboard.png - Summary visualization")
print("  • a2a_final_comprehensive_guide.ipynb - This complete guide")
print("\n🎯 Next Step: Secure funding and begin Phase 1 (Foundation)")
print("="*80)
```

\================================================================================  
✓ COMPREHENSIVE A2A WORLD IMPLEMENTATION GUIDE COMPLETE  
\================================================================================

📊 Deliverables Created:  
1\. Complete multidisciplinary analysis of previous attempts  
2\. 5 detailed cross-disciplinary research protocols  
3\. 12-month implementation roadmap  
4\. Resource requirements and team specifications  
5\. Risk mitigation and ethical framework  
6\. Expected outcomes and impact projections

📁 Files Saved:  
• a2a\_executive\_dashboard.png \- Summary visualization  
• a2a\_final\_comprehensive\_guide.ipynb \- This complete guide

🎯 Next Step: Secure funding and begin Phase 1 (Foundation)  
\================================================================================

![][image1]  


[image1]: <data:image/png;base64,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>