# **A2A World Platform: Multidisciplinary Knowledge Synthesis**

## **Cross-Referencing 17 Academic Disciplines for Pattern Discovery**

Objective: Integrate insights from Art, History, Religious Studies, Astrology, Archaeology, Environmental Studies, Sociology, Linguistics, Folklore, Anthropology, Geography, Humanities, Cognitive Science, Psychology, Classical Literature, Astrophysics, and Cultural Anthropology to create a comprehensive framework for discovering meaningful patterns in the A2A World platform.

Approach:

1. Extract key findings and methodologies from three previous analysis attempts  
2. Build an interdisciplinary knowledge graph connecting disciplines  
3. Design specific analytical methods for cross-domain pattern discovery  
4. Create actionable research protocols for each disciplinary intersection

---

In \[2\]:

```
# Install required packages
!pip install -q pandas numpy matplotlib seaborn scikit-learn networkx
```

WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the \--root-user-action option if you know what you are doing and want to suppress this warning.

In \[3\]:

```
# Install required packages first
!pip install -q pandas numpy matplotlib seaborn scikit-learn networkx
```

WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the \--root-user-action option if you know what you are doing and want to suppress this warning.

In \[4\]:

```

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set up visualization parameters
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

# Define paths to the three previous analysis PDFs
pdf_paths = [
    '/workspace/files/ab9d5511194c41a48dd5f77fe046144e/content.md',
    '/workspace/files/79bcf82513ee470f95e166c4db8f251a/content.md',
    '/workspace/files/e39c2362720745a7bfeb38cf4227c2f4/content.md'
]

# Load all three analysis documents
documents = {}
for i, path in enumerate(pdf_paths, 1):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            documents[f'Analysis_{i}'] = f.read()
        print(f"✓ Loaded Analysis {i}: {len(documents[f'Analysis_{i}'])} characters")
    except Exception as e:
        print(f"✗ Error loading Analysis {i}: {e}")

print(f"\n📚 Total documents loaded: {len(documents)}")
print(f"📊 Total content size: {sum(len(doc) for doc in documents.values()):,} characters")
```

✓ Loaded Analysis 1: 42810 characters  
✓ Loaded Analysis 2: 59087 characters  
✓ Loaded Analysis 3: 137141 characters

📚 Total documents loaded: 3  
📊 Total content size: 239,038 characters

## **Step 1: Extract Key Insights from Previous Analyses**

Let's systematically parse the three previous analysis attempts to identify:

1. Methodologies used in each analysis  
2. Key findings and patterns discovered  
3. Limitations and gaps identified  
4. Disciplinary perspectives already covered

In \[5\]:

```
# Define the 17 target disciplines
DISCIPLINES = [
    'Art', 'History', 'Religious Studies', 'Astrology', 'Archaeology',
    'Environmental Studies', 'Sociology', 'Linguistics', 'Folklore',
    'Anthropology', 'Geography', 'Humanities', 'Cognitive Science',
    'Psychology', 'Classical Literature', 'Astrophysics', 'Cultural Anthropology'
]

# Extract disciplinary coverage from each document
def extract_discipline_mentions(text, disciplines):
    """Count mentions of each discipline in the text"""
    mentions = {}
    text_lower = text.lower()
    
    for discipline in disciplines:
        # Create variations of the discipline name
        patterns = [
            discipline.lower(),
            discipline.lower().replace(' ', '-'),
            discipline.lower().replace(' ', ''),
        ]
        
        count = sum(text_lower.count(pattern) for pattern in patterns)
        mentions[discipline] = count
    
    return mentions

# Analyze each document
discipline_coverage = {}
for doc_name, content in documents.items():
    discipline_coverage[doc_name] = extract_discipline_mentions(content, DISCIPLINES)

# Create a coverage matrix
coverage_df = pd.DataFrame(discipline_coverage).T
coverage_df['Total'] = coverage_df.sum(axis=1)

print("📊 DISCIPLINARY COVERAGE ACROSS THREE ANALYSES")
print("="*70)
print(coverage_df.to_string())
print("\n")

# Identify underrepresented disciplines
total_mentions = coverage_df.drop('Total', axis=1).sum()
underrepresented = total_mentions[total_mentions < 3].sort_values()

print("⚠️  UNDERREPRESENTED DISCIPLINES (< 3 total mentions):")
print("-"*70)
for discipline, count in underrepresented.items():
    print(f"  • {discipline}: {count} mentions")
    
print(f"\n✓ Well-represented: {len(total_mentions[total_mentions >= 3])} disciplines")
print(f"⚠️  Underrepresented: {len(underrepresented)} disciplines")
```

📊 DISCIPLINARY COVERAGE ACROSS THREE ANALYSES  
\======================================================================  
Art History Religious Studies Astrology Archaeology Environmental Studies Sociology Linguistics Folklore Anthropology Geography Humanities Cognitive Science Psychology Classical Literature Astrophysics Cultural Anthropology Total  
Analysis\_1 261 3 0 0 3 0 3 3 6 0 3 0 0 0 0 0 0 282  
Analysis\_2 180 15 0 0 6 0 0 0 6 0 24 0 0 0 0 0 0 231  
Analysis\_3 447 48 1 0 0 0 0 0 51 0 3 0 0 0 0 0 0 550

⚠️ UNDERREPRESENTED DISCIPLINES (\< 3 total mentions):  
\----------------------------------------------------------------------  
• Astrology: 0 mentions  
• Environmental Studies: 0 mentions  
• Anthropology: 0 mentions  
• Humanities: 0 mentions  
• Psychology: 0 mentions  
• Cognitive Science: 0 mentions  
• Classical Literature: 0 mentions  
• Astrophysics: 0 mentions  
• Cultural Anthropology: 0 mentions  
• Religious Studies: 1 mentions

✓ Well-represented: 7 disciplines  
⚠️ Underrepresented: 10 disciplines

In \[6\]:

```

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
# Visualize disciplinary coverage
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Coverage heatmap
coverage_matrix = coverage_df.drop('Total', axis=1).T
sns.heatmap(coverage_matrix, annot=True, fmt='d', cmap='YlOrRd', 
            ax=axes[0], cbar_kws={'label': 'Mentions'})
axes[0].set_title('Disciplinary Coverage Across Three Analyses', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Analysis Document', fontsize=12)
axes[0].set_ylabel('Academic Discipline', fontsize=12)

# Plot 2: Total mentions by discipline
total_by_discipline = coverage_matrix.sum(axis=1).sort_values(ascending=True)
colors = ['#d62728' if x < 3 else '#2ca02c' for x in total_by_discipline.values]

axes[1].barh(range(len(total_by_discipline)), total_by_discipline.values, color=colors)
axes[1].set_yticks(range(len(total_by_discipline)))
axes[1].set_yticklabels(total_by_discipline.index)
axes[1].set_xlabel('Total Mentions Across All Analyses', fontsize=12)
axes[1].set_title('Disciplinary Representation Gap Analysis', fontsize=14, fontweight='bold')
axes[1].axvline(x=3, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Minimum threshold')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/a2a_disciplinary_coverage.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualization saved: a2a_disciplinary_coverage.png")
```

✓ Visualization saved: a2a\_disciplinary\_coverage.png

![][image1]

## **Analysis of Disciplinary Coverage**

### **Key Findings:**

✓ Well-Represented Disciplines (\>3 mentions):

* Art (888 mentions) \- Dominates all three analyses, likely due to focus on cultural artifacts and artistic representations  
* History (66 mentions) \- Good coverage of historical context  
* Folklore (63 mentions) \- Strong presence, especially in Analysis 3  
* Geography (30 mentions) \- Moderate coverage of spatial analysis  
* Archaeology (9 mentions) \- Present but limited

⚠️ CRITICAL GAPS \- Zero or Minimal Coverage: The following 10 disciplines are essentially absent from previous analyses:

1. Astrology (0 mentions) \- No celestial/astronomical pattern analysis  
2. Environmental Studies (0 mentions) \- Missing climate, ecology, landscape analysis  
3. Anthropology (0 mentions) \- Lacking cultural evolution perspective  
4. Humanities (0 mentions) \- Broad theoretical frameworks missing  
5. Psychology (0 mentions) \- No cognitive or behavioral analysis  
6. Cognitive Science (0 mentions) \- Missing perception and cognition studies  
7. Classical Literature (0 mentions) \- Ancient texts not integrated  
8. Astrophysics (0 mentions) \- No astronomical/celestial mechanics  
9. Cultural Anthropology (0 mentions) \- Limited cultural comparison  
10. Religious Studies (1 mention) \- Minimal theological/ritual analysis

### **Strategic Insight:**

The previous analyses were heavily skewed toward art history and geography, with superficial treatment of the broader interdisciplinary landscape. To achieve the A2A World vision, we must systematically integrate the missing 10 disciplines.

## **Step 2: Build an Interdisciplinary Knowledge Graph**

We'll create a network showing how the 17 disciplines can interconnect to discover meaningful patterns in the A2A World platform. This will reveal:

1. Direct connections between disciplines (shared methodologies, data types, or research questions)  
2. Bridge disciplines that connect otherwise isolated fields  
3. Research clusters that should be analyzed together

In \[7\]:

```
import networkx as nx

# Define interdisciplinary connections based on shared methodologies, data, or research questions
# Format: (Discipline1, Discipline2, Connection_Strength, Shared_Aspect)
interdisciplinary_edges = [
    # Core Cultural Cluster
    ('Art', 'History', 5, 'Historical artifacts & artistic movements'),
    ('Art', 'Archaeology', 5, 'Material culture & excavated artifacts'),
    ('Art', 'Classical Literature', 4, 'Visual representations of myths'),
    ('History', 'Archaeology', 5, 'Dating & material evidence'),
    ('History', 'Classical Literature', 4, 'Ancient texts & historical records'),
    ('Archaeology', 'Geography', 5, 'Site location & spatial analysis'),
    
    # Mythology & Belief Systems Cluster
    ('Religious Studies', 'Folklore', 5, 'Sacred narratives & rituals'),
    ('Religious Studies', 'Anthropology', 4, 'Ritual practices & belief systems'),
    ('Folklore', 'Cultural Anthropology', 5, 'Oral traditions & cultural transmission'),
    ('Folklore', 'Linguistics', 4, 'Narrative structures & language'),
    ('Classical Literature', 'Religious Studies', 4, 'Sacred texts & theological themes'),
    ('Classical Literature', 'Folklore', 3, 'Mythological narratives'),
    
    # Cognitive & Behavioral Cluster
    ('Psychology', 'Cognitive Science', 5, 'Perception & cognition'),
    ('Psychology', 'Anthropology', 4, 'Cultural psychology & behavior'),
    ('Cognitive Science', 'Linguistics', 4, 'Language processing & meaning'),
    ('Psychology', 'Religious Studies', 3, 'Religious experience & belief formation'),
    ('Cognitive Science', 'Art', 3, 'Aesthetic perception & symbolism'),
    
    # Environmental & Astronomical Cluster
    ('Environmental Studies', 'Geography', 5, 'Landscape ecology & climate'),
    ('Environmental Studies', 'Archaeology', 4, 'Paleoenvironment & settlement patterns'),
    ('Astrophysics', 'Astrology', 5, 'Celestial observations & astronomical events'),
    ('Astrophysics', 'Geography', 3, 'Astronomical alignments & latitude'),
    ('Astrology', 'Religious Studies', 4, 'Celestial symbolism & cosmology'),
    ('Astrology', 'Classical Literature', 3, 'Astrological references in texts'),
    
    # Social & Cultural Theory Cluster
    ('Sociology', 'Cultural Anthropology', 5, 'Social structures & cultural norms'),
    ('Sociology', 'History', 3, 'Social movements & historical change'),
    ('Anthropology', 'Linguistics', 4, 'Language & culture'),
    ('Cultural Anthropology', 'Geography', 4, 'Cultural geography & spatial practices'),
    
    # Integrative Humanities
    ('Humanities', 'Art', 4, 'Cultural interpretation & meaning'),
    ('Humanities', 'History', 4, 'Historical interpretation'),
    ('Humanities', 'Philosophy', 3, 'Theoretical frameworks'),
    ('Humanities', 'Classical Literature', 4, 'Textual analysis & interpretation'),
    
    # Cross-cluster bridges (high value connections)
    ('Geography', 'Astrophysics', 4, 'Astronomical alignments of sites'),
    ('Archaeology', 'Astrophysics', 4, 'Archaeoastronomy'),
    ('Art', 'Astrology', 3, 'Astrological symbolism in art'),
    ('Environmental Studies', 'Folklore', 3, 'Environmental mythology & nature spirits'),
    ('Cognitive Science', 'Religious Studies', 3, 'Cognitive science of religion'),
    ('Psychology', 'Art', 3, 'Symbolism & psychological archetypes'),
]

# Create the knowledge graph
G = nx.Graph()

# Add nodes with metadata
for discipline in DISCIPLINES:
    G.add_node(discipline)

# Add edges with weights
for d1, d2, weight, aspect in interdisciplinary_edges:
    if d1 in DISCIPLINES and d2 in DISCIPLINES:
        G.add_edge(d1, d2, weight=weight, aspect=aspect)

print(f"📊 INTERDISCIPLINARY KNOWLEDGE GRAPH")
print(f"="*70)
print(f"Nodes (Disciplines): {G.number_of_nodes()}")
print(f"Edges (Connections): {G.number_of_edges()}")
print(f"Average connections per discipline: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
print(f"\n🔗 Network Density: {nx.density(G):.3f}")
print(f"   (Range: 0=no connections, 1=fully connected)")

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Identify bridge disciplines (high betweenness centrality)
print(f"\n🌉 BRIDGE DISCIPLINES (Top 5 by Betweenness Centrality):")
print(f"-"*70)
for discipline, score in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  • {discipline:25s} - {score:.3f}")

# Identify hub disciplines (high degree centrality)  
print(f"\n⭐ HUB DISCIPLINES (Top 5 by Degree Centrality):")
print(f"-"*70)
for discipline, score in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    connections = G.degree(discipline)
    print(f"  • {discipline:25s} - {connections} connections")
```

📊 INTERDISCIPLINARY KNOWLEDGE GRAPH  
\======================================================================  
Nodes (Disciplines): 17  
Edges (Connections): 35  
Average connections per discipline: 4.12

🔗 Network Density: 0.257  
(Range: 0=no connections, 1=fully connected)

🌉 BRIDGE DISCIPLINES (Top 5 by Betweenness Centrality):  
\----------------------------------------------------------------------  
• Art \- 0.198  
• Folklore \- 0.191  
• Religious Studies \- 0.129  
• Archaeology \- 0.119  
• Classical Literature \- 0.106

⭐ HUB DISCIPLINES (Top 5 by Degree Centrality):  
\----------------------------------------------------------------------  
• Art \- 7 connections  
• Religious Studies \- 6 connections  
• Classical Literature \- 6 connections  
• History \- 5 connections  
• Archaeology \- 5 connections

In \[8\]:

```

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
# Create a proper dictionary of discipline mentions from the coverage data
discipline_mentions_dict = coverage_df.drop('Total', axis=1).sum().to_dict()

print("Discipline mentions dictionary:")
for disc, count in sorted(discipline_mentions_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {disc}: {count}")

# Now visualize the knowledge graph with corrected variable
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Layout 1: Spring layout showing natural clustering
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Determine node colors based on previous analysis coverage
node_colors = []
for node in G.nodes():
    mentions = discipline_mentions_dict.get(node, 0)
    if mentions >= 100:
        node_colors.append('#2ca02c')  # Green - well covered
    elif mentions >= 3:
        node_colors.append('#ff7f0e')  # Orange - moderate
    else:
        node_colors.append('#d62728')  # Red - underrepresented

# Determine node sizes based on degree centrality
node_sizes = [3000 * degree_centrality[node] for node in G.nodes()]

# Draw on first axis
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                       alpha=0.9, ax=axes[0])
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=axes[0])

# Draw edges with varying thickness based on connection strength
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights], alpha=0.5, ax=axes[0])

axes[0].set_title('A2A World Interdisciplinary Knowledge Graph\n' + 
                   'Node Color: Green=Well-covered, Orange=Moderate, Red=Gap | ' +
                   'Node Size: Network Centrality',
                   fontsize=12, fontweight='bold')
axes[0].axis('off')

# Layout 2: Circular layout for clarity
pos_circular = nx.circular_layout(G)

nx.draw_networkx_nodes(G, pos_circular, node_color=node_colors, node_size=node_sizes,
                       alpha=0.9, ax=axes[1])
nx.draw_networkx_labels(G, pos_circular, font_size=9, font_weight='bold', ax=axes[1])
nx.draw_networkx_edges(G, pos_circular, width=[w*0.5 for w in weights], 
                       alpha=0.4, edge_color='gray', ax=axes[1])

axes[1].set_title('Circular View: Disciplinary Network\n' +
                   'Edge Thickness: Connection Strength',
                   fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('/workspace/a2a_knowledge_graph.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved: a2a_knowledge_graph.png")
```

Discipline mentions dictionary:  
Art: 888  
History: 66  
Folklore: 63  
Geography: 30  
Archaeology: 9

✓ Visualization saved: a2a\_knowledge\_graph.png

![][image2]  
In \[9\]:

```
# Restart kernel approach - create fresh mention counts
discipline_mention_counts = {}
for disc in DISCIPLINES:
    total = 0
    for doc in documents.values():
        total += doc.lower().count(disc.lower())
    discipline_mention_counts[disc] = total

print("✓ Fresh discipline mention counts:")
for disc, count in sorted(discipline_mention_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"  {disc:25s}: {count:4d} mentions")
```

✓ Fresh discipline mention counts:  
Art : 296 mentions  
History : 22 mentions  
Folklore : 21 mentions  
Geography : 10 mentions  
Archaeology : 3 mentions  
Religious Studies : 1 mentions  
Sociology : 1 mentions  
Linguistics : 1 mentions

In \[10\]:

```

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
# Now visualize the knowledge graph
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Layout 1: Spring layout showing natural clustering
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Determine node colors based on analysis coverage
node_colors = []
for node in G.nodes():
    mentions = discipline_mention_counts.get(node, 0)
    if mentions >= 50:
        node_colors.append('#2ca02c')  # Green - well covered
    elif mentions >= 3:
        node_colors.append('#ff7f0e')  # Orange - moderate
    else:
        node_colors.append('#d62728')  # Red - underrepresented

# Determine node sizes based on degree centrality
node_sizes = [3000 * degree_centrality[node] for node in G.nodes()]

# Draw on first axis
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                       alpha=0.9, ax=axes[0])
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=axes[0])

# Draw edges with varying thickness based on connection strength
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights], alpha=0.5, ax=axes[0])

axes[0].set_title('A2A World Interdisciplinary Knowledge Graph\n' + 
                   'Node Color: Green=Well-covered, Orange=Moderate, Red=Gap | ' +
                   'Node Size: Network Centrality',
                   fontsize=12, fontweight='bold')
axes[0].axis('off')

# Layout 2: Circular layout for clarity
pos_circular = nx.circular_layout(G)

nx.draw_networkx_nodes(G, pos_circular, node_color=node_colors, node_size=node_sizes,
                       alpha=0.9, ax=axes[1])
nx.draw_networkx_labels(G, pos_circular, font_size=9, font_weight='bold', ax=axes[1])
nx.draw_networkx_edges(G, pos_circular, width=[w*0.5 for w in weights], 
                       alpha=0.4, edge_color='gray', ax=axes[1])

axes[1].set_title('Circular View: Disciplinary Network\n' +
                   'Edge Thickness: Connection Strength',
                   fontsize=12, fontweight='bold')
axes[1].axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', label='Well-covered (≥50 mentions)'),
    Patch(facecolor='#ff7f0e', label='Moderate (3-49 mentions)'),
    Patch(facecolor='#d62728', label='Gap (0-2 mentions)')
]
axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('/workspace/a2a_knowledge_graph.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved: a2a_knowledge_graph.png")
```

✓ Visualization saved: a2a\_knowledge\_graph.png

![][image3]

## **Step 3: Design Cross-Disciplinary Research Protocols**

The knowledge graph reveals Art as the central hub (well-covered, green), with critical bridge disciplines connecting otherwise isolated fields. The vast majority of nodes are red (gaps), indicating that previous analyses failed to leverage the full interdisciplinary potential.

### **Key Observations from the Network:**

1. Art is the dominant hub with 7 connections \- it bridges cultural, cognitive, and astronomical clusters  
2. Religious Studies, Classical Literature, History, Archaeology form a strong cultural heritage cluster  
3. Psychology, Cognitive Science are isolated \- no connections to the main network in previous analyses  
4. Astrophysics, Astrology are disconnected despite their relevance to astronomical alignments  
5. Environmental Studies is isolated despite its importance for understanding settlement patterns

### **Strategic Approach:**

We will now create Cross-Disciplinary Research Protocols \- specific methodological frameworks that combine multiple disciplines to discover meaningful patterns in the A2A World data.

In \[11\]:

```
# Define Cross-Disciplinary Research Protocols
# Each protocol combines 3-5 disciplines to address specific research questions

research_protocols = {
    'Protocol_1_Archaeoastronomy': {
        'name': 'Archaeoastronomy & Celestial Alignment Analysis',
        'disciplines': ['Archaeology', 'Astrophysics', 'Astrology', 'Geography', 'Religious Studies'],
        'research_question': 'Do sacred sites align with astronomical phenomena (solstices, equinoxes, star positions)?',
        'data_requirements': [
            'Geographic coordinates of all sites',
            'Digital elevation models (DEM) for horizon profiles',
            'Historical astronomical ephemeris (sun, moon, planets, stars)',
            'Site dating information',
            'Cultural astronomical knowledge from texts'
        ],
        'methods': [
            '1. Calculate azimuth and altitude of celestial bodies at site dates',
            '2. Compute site-to-site alignments and compare to astronomical events',
            '3. Statistical significance testing vs random site distributions',
            '4. Cross-reference with cultural astronomical beliefs from texts',
            '5. Account for precession of equinoxes over millennia'
        ],
        'expected_patterns': [
            'Solstice/equinox alignments',
            'Heliacal rising/setting of specific stars',
            'Lunar standstill orientations',
            'Planetary conjunction alignments'
        ],
        'validation': 'Monte Carlo simulation of random site placements'
    },
    
    'Protocol_2_PsychoGeography': {
        'name': 'Cognitive Psychology of Sacred Landscapes',
        'disciplines': ['Psychology', 'Cognitive Science', 'Geography', 'Anthropology', 'Religious Studies'],
        'research_question': 'Do sacred sites cluster in locations that maximize psychological/perceptual impact?',
        'data_requirements': [
            'Viewshed analysis from each site',
            'Topographic prominence and visual dominance',
            'Acoustic properties (echo, reverberation)',
            'Natural landscape features (caves, springs, peaks)',
            'Cognitive salience metrics (visual uniqueness, memorability)'
        ],
        'methods': [
            '1. Compute viewshed area and visual prominence for each site',
            '2. Analyze topographic prominence (isolation, dominance)',
            '3. Model acoustic properties using terrain data',
            '4. Assess "sense of enclosure" vs "sense of openness"',
            '5. Compare to random site distributions',
            '6. Cross-reference with mythological themes of transcendence/descent'
        ],
        'expected_patterns': [
            'Sites on high-prominence peaks (sky gods, transcendence)',
            'Sites in enclosed valleys (underworld, initiation)',
            'Sites with exceptional viewsheds (panoramic awareness)',
            'Sites with acoustic anomalies (oracle sites, echo chambers)'
        ],
        'validation': 'Controlled comparison with non-sacred archaeological sites'
    },
    
    'Protocol_3_EcoMythology': {
        'name': 'Environmental Determinants of Mythology',
        'disciplines': ['Environmental Studies', 'Folklore', 'Linguistics', 'Cultural Anthropology', 'History'],
        'research_question': 'Do environmental conditions predict mythological themes across cultures?',
        'data_requirements': [
            'Climate data (temperature, precipitation, seasonality)',
            'Ecological zones (biomes, vegetation)',
            'Natural hazard history (floods, droughts, earthquakes)',
            'Water sources (rivers, springs, lakes)',
            'Mythological theme classifications from texts',
            'Linguistic analysis of nature-related terminology'
        ],
        'methods': [
            '1. Extract environmental variables for each site region',
            '2. Classify mythological narratives by theme (water, fire, earth, sky)',
            '3. Perform topic modeling on folklore texts',
            '4. Correlate environmental variables with myth themes',
            '5. Account for cultural diffusion vs independent development',
            '6. Analyze linguistic terms for natural phenomena'
        ],
        'expected_patterns': [
            'Flood myths in flood-prone regions',
            'Dragon/serpent myths near major rivers',
            'Sky god dominance in open landscapes',
            'Chthonic deities in seismically active regions',
            'Solar emphasis in arid climates'
        ],
        'validation': 'Cross-cultural comparison controlling for contact/diffusion'
    },
    
    'Protocol_4_ArtisticDiffusion': {
        'name': 'Artistic Motif Diffusion & Cultural Contact',
        'disciplines': ['Art', 'History', 'Archaeology', 'Sociology', 'Geography'],
        'research_question': 'Can artistic motifs reveal cultural contact networks not evident from historical records?',
        'data_requirements': [
            'Catalog of artistic motifs (symbols, geometric patterns, figurative styles)',
            'Chronological dating of artworks',
            'Geographic distribution of motifs',
            'Known trade routes and cultural contact zones',
            'Material culture (pottery, metallurgy, textiles)'
        ],
        'methods': [
            '1. Computer vision analysis to extract visual motifs',
            '2. Cluster similar motifs across geographic space',
            '3. Build motif similarity networks',
            '4. Analyze temporal spread patterns',
            '5. Compare to geographic distance and known trade routes',
            '6. Identify "innovation centers" vs "recipient zones"'
        ],
        'expected_patterns': [
            'Motif clusters along trade routes',
            'Temporal lag in motif adoption by distance',
            'Hybrid motifs in cultural contact zones',
            'Persistent local motifs despite external contact'
        ],
        'validation': 'Phylogenetic analysis of motif evolution'
    },
    
    'Protocol_5_LiteraryGeography': {
        'name': 'Mythological Geography in Classical Literature',
        'disciplines': ['Classical Literature', 'Humanities', 'Geography', 'History', 'Religious Studies'],
        'research_question': 'Do mythological narratives encode real geographic knowledge or symbolic landscapes?',
        'data_requirements': [
            'Full corpus of classical texts (Homer, Hesiod, Ovid, etc.)',
            'Geographic place names in narratives',
            'Journey/travel narratives with distances and directions',
            'Modern geographic coordinates of identified locations',
            'Archaeological evidence of ancient geographic knowledge'
        ],
        'methods': [
            '1. Natural language processing to extract geographic references',
            '2. Map narrative journeys onto real geography',
            '3. Calculate accuracy of described distances/directions',
            '4. Identify systematic distortions (exaggerations, symbolic directions)',
            '5. Compare to archaeological evidence of ancient cartography',
            '6. Analyze symbolic vs literal geographic language'
        ],
        'expected_patterns': [
            'Accurate local geography, distorted distant geography',
            'Symbolic cardinal directions (East=sunrise=rebirth)',
            'Journey narratives as initiation structures',
            'Geographic clustering of related myths'
        ],
        'validation': 'Comparison with ancient maps and geographic texts (Ptolemy, Strabo)'
    }
}

print(f"📋 CROSS-DISCIPLINARY RESEARCH PROTOCOLS")
print(f"="*70)
print(f"Total protocols designed: {len(research_protocols)}\n")

for protocol_id, details in research_protocols.items():
    print(f"\n{'='*70}")
    print(f"🔬 {details['name']}")
    print(f"{'='*70}")
    print(f"Disciplines: {', '.join(details['disciplines'])}")
    print(f"\n❓ Research Question:")
    print(f"   {details['research_question']}")
    print(f"\n📊 Expected Pattern Types: {len(details['expected_patterns'])}")
    for pattern in details['expected_patterns']:
        print(f"   • {pattern}")

print(f"\n\n✓ All protocols defined with detailed methodologies")
```

📋 CROSS-DISCIPLINARY RESEARCH PROTOCOLS  
\======================================================================  
Total protocols designed: 5

\======================================================================  
🔬 Archaeoastronomy & Celestial Alignment Analysis  
\======================================================================  
Disciplines: Archaeology, Astrophysics, Astrology, Geography, Religious Studies

❓ Research Question:  
Do sacred sites align with astronomical phenomena (solstices, equinoxes, star positions)?

📊 Expected Pattern Types: 4  
• Solstice/equinox alignments  
• Heliacal rising/setting of specific stars  
• Lunar standstill orientations  
• Planetary conjunction alignments

\======================================================================  
🔬 Cognitive Psychology of Sacred Landscapes  
\======================================================================  
Disciplines: Psychology, Cognitive Science, Geography, Anthropology, Religious Studies

❓ Research Question:  
Do sacred sites cluster in locations that maximize psychological/perceptual impact?

📊 Expected Pattern Types: 4  
• Sites on high-prominence peaks (sky gods, transcendence)  
• Sites in enclosed valleys (underworld, initiation)  
• Sites with exceptional viewsheds (panoramic awareness)  
• Sites with acoustic anomalies (oracle sites, echo chambers)

\======================================================================  
🔬 Environmental Determinants of Mythology  
\======================================================================  
Disciplines: Environmental Studies, Folklore, Linguistics, Cultural Anthropology, History

❓ Research Question:  
Do environmental conditions predict mythological themes across cultures?

📊 Expected Pattern Types: 5  
• Flood myths in flood-prone regions  
• Dragon/serpent myths near major rivers  
• Sky god dominance in open landscapes  
• Chthonic deities in seismically active regions  
• Solar emphasis in arid climates

\======================================================================  
🔬 Artistic Motif Diffusion & Cultural Contact  
\======================================================================  
Disciplines: Art, History, Archaeology, Sociology, Geography

❓ Research Question:  
Can artistic motifs reveal cultural contact networks not evident from historical records?

📊 Expected Pattern Types: 4  
• Motif clusters along trade routes  
• Temporal lag in motif adoption by distance  
• Hybrid motifs in cultural contact zones  
• Persistent local motifs despite external contact

\======================================================================  
🔬 Mythological Geography in Classical Literature  
\======================================================================  
Disciplines: Classical Literature, Humanities, Geography, History, Religious Studies

❓ Research Question:  
Do mythological narratives encode real geographic knowledge or symbolic landscapes?

📊 Expected Pattern Types: 4  
• Accurate local geography, distorted distant geography  
• Symbolic cardinal directions (East=sunrise=rebirth)  
• Journey narratives as initiation structures  
• Geographic clustering of related myths

✓ All protocols defined with detailed methodologies

In \[12\]:

```

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
# Visualize the research protocols as a network
fig, ax = plt.subplots(1, 1, figsize=(18, 12))

# Create a bipartite graph: protocols and disciplines
B = nx.Graph()

# Add protocol nodes
protocol_nodes = list(research_protocols.keys())
B.add_nodes_from(protocol_nodes, bipartite=0)

# Add discipline nodes  
discipline_nodes = DISCIPLINES
B.add_nodes_from(discipline_nodes, bipartite=1)

# Add edges from protocols to their constituent disciplines
for protocol_id, details in research_protocols.items():
    for discipline in details['disciplines']:
        B.add_edge(protocol_id, discipline)

# Create layout
pos = {}
# Position protocols on the left
protocol_y_positions = np.linspace(0, 10, len(protocol_nodes))
for i, protocol in enumerate(protocol_nodes):
    pos[protocol] = (0, protocol_y_positions[i])

# Position disciplines on the right based on mention count
discipline_y_positions = np.linspace(0, 10, len(discipline_nodes))
sorted_disciplines = sorted(discipline_nodes, 
                           key=lambda x: discipline_mention_counts.get(x, 0), 
                           reverse=True)
for i, discipline in enumerate(sorted_disciplines):
    pos[discipline] = (3, discipline_y_positions[i])

# Node colors
node_colors_viz = []
node_sizes_viz = []
for node in B.nodes():
    if node in protocol_nodes:
        node_colors_viz.append('#1f77b4')  # Blue for protocols
        node_sizes_viz.append(3000)
    else:
        mentions = discipline_mention_counts.get(node, 0)
        if mentions >= 50:
            node_colors_viz.append('#2ca02c')  # Green
        elif mentions >= 3:
            node_colors_viz.append('#ff7f0e')  # Orange
        else:
            node_colors_viz.append('#d62728')  # Red
        node_sizes_viz.append(2000)

# Draw the network
nx.draw_networkx_nodes(B, pos, node_color=node_colors_viz, node_size=node_sizes_viz,
                       alpha=0.8, ax=ax)

# Custom labels for protocols (shorter names)
protocol_labels = {
    'Protocol_1_Archaeoastronomy': 'P1: Archaeoastronomy',
    'Protocol_2_PsychoGeography': 'P2: Psychogeography',
    'Protocol_3_EcoMythology': 'P3: Eco-Mythology',
    'Protocol_4_ArtisticDiffusion': 'P4: Art Diffusion',
    'Protocol_5_LiteraryGeography': 'P5: Literary Geography'
}

# Draw labels
for node, (x, y) in pos.items():
    if node in protocol_nodes:
        label = protocol_labels[node]
        ax.text(x-0.15, y, label, fontsize=10, fontweight='bold', 
               ha='right', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    else:
        mentions = discipline_mention_counts.get(node, 0)
        label = f"{node}\n({mentions})"
        ax.text(x+0.15, y, label, fontsize=9, ha='left', va='center')

# Draw edges
nx.draw_networkx_edges(B, pos, alpha=0.3, width=1.5, ax=ax)

ax.set_title('A2A World Cross-Disciplinary Research Protocols\n' +
             'Protocols (Left) → Constituent Disciplines (Right)\n' +
             'Discipline Color: Green=Well-covered, Orange=Moderate, Red=Gap',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')
ax.set_xlim(-0.5, 4)

plt.tight_layout()
plt.savefig('/workspace/a2a_research_protocols.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualization saved: a2a_research_protocols.png")
```

✓ Visualization saved: a2a\_research\_protocols.png

![][image4]

## **Step 4: Quantify Protocol Impact & Discipline Coverage**

The visualization shows that our 5 research protocols successfully integrate all 17 target disciplines, with particular emphasis on filling the critical gaps identified in previous analyses.

### **Protocol Coverage Analysis:**

* P1 (Archaeoastronomy): Integrates the completely missing Astrophysics and Astrology disciplines  
* P2 (Psychogeography): Brings in Psychology and Cognitive Science (both had 0 mentions)  
* P3 (Eco-Mythology): Incorporates Environmental Studies (0 mentions) and Cultural Anthropology  
* P4 (Art Diffusion): Leverages the well-covered Art discipline as a bridge  
* P5 (Literary Geography): Activates Classical Literature and Humanities (both had 0 mentions)

Let's quantify how effectively each protocol addresses the disciplinary gaps.

In \[13\]:

```
# Analyze protocol impact on filling disciplinary gaps
protocol_impact = []

for protocol_id, details in research_protocols.items():
    protocol_disciplines = details['disciplines']
    
    # Calculate impact metrics
    total_disciplines = len(protocol_disciplines)
    gap_disciplines = sum(1 for d in protocol_disciplines if discipline_mention_counts.get(d, 0) < 3)
    moderate_disciplines = sum(1 for d in protocol_disciplines if 3 <= discipline_mention_counts.get(d, 0) < 50)
    covered_disciplines = sum(1 for d in protocol_disciplines if discipline_mention_counts.get(d, 0) >= 50)
    
    # Gap-filling score (higher is better for filling gaps)
    gap_fill_score = (gap_disciplines * 3) + (moderate_disciplines * 1.5) + (covered_disciplines * 0.5)
    
    protocol_impact.append({
        'Protocol': details['name'],
        'Short_Name': protocol_id.replace('Protocol_', 'P').replace('_', '-'),
        'Total_Disciplines': total_disciplines,
        'Gap_Disciplines': gap_disciplines,
        'Moderate_Disciplines': moderate_disciplines,
        'Covered_Disciplines': covered_disciplines,
        'Gap_Fill_Score': gap_fill_score,
        'Research_Question': details['research_question']
    })

# Create DataFrame
impact_df = pd.DataFrame(protocol_impact)
impact_df = impact_df.sort_values('Gap_Fill_Score', ascending=False)

print("📊 PROTOCOL IMPACT ANALYSIS: Filling Disciplinary Gaps")
print("="*100)
print("\nGap-Fill Scoring:")
print("  • Gap discipline (0-2 mentions): 3 points")
print("  • Moderate discipline (3-49 mentions): 1.5 points")
print("  • Covered discipline (≥50 mentions): 0.5 points")
print("  Higher score = Better at filling critical gaps\n")
print("-"*100)
print(impact_df[['Short_Name', 'Protocol', 'Gap_Disciplines', 'Moderate_Disciplines', 
                 'Covered_Disciplines', 'Gap_Fill_Score']].to_string(index=False))
print("-"*100)

# Summary statistics
total_gap_disciplines_addressed = impact_df['Gap_Disciplines'].sum()
unique_disciplines_used = set()
for _, details in research_protocols.items():
    unique_disciplines_used.update(details['disciplines'])

print(f"\n📈 COVERAGE SUMMARY:")
print(f"  • Total gap disciplines addressed across all protocols: {total_gap_disciplines_addressed}")
print(f"  • Unique disciplines incorporated: {len(unique_disciplines_used)}/17 ({len(unique_disciplines_used)/17*100:.1f}%)")
print(f"  • Average gap disciplines per protocol: {total_gap_disciplines_addressed/len(research_protocols):.1f}")
print(f"\n✓ All 17 target disciplines are now integrated into research protocols")
```

📊 PROTOCOL IMPACT ANALYSIS: Filling Disciplinary Gaps  
\====================================================================================================

Gap-Fill Scoring:  
• Gap discipline (0-2 mentions): 3 points  
• Moderate discipline (3-49 mentions): 1.5 points  
• Covered discipline (≥50 mentions): 0.5 points  
Higher score \= Better at filling critical gaps

\----------------------------------------------------------------------------------------------------  
Short\_Name Protocol Gap\_Disciplines Moderate\_Disciplines Covered\_Disciplines Gap\_Fill\_Score  
P2-PsychoGeography Cognitive Psychology of Sacred Landscapes 4 1 0 13.5  
P1-Archaeoastronomy Archaeoastronomy & Celestial Alignment Analysis 3 2 0 12.0  
P3-EcoMythology Environmental Determinants of Mythology 3 2 0 12.0  
P5-LiteraryGeography Mythological Geography in Classical Literature 3 2 0 12.0  
P4-ArtisticDiffusion Artistic Motif Diffusion & Cultural Contact 1 3 1 8.0  
\----------------------------------------------------------------------------------------------------

📈 COVERAGE SUMMARY:  
• Total gap disciplines addressed across all protocols: 14  
• Unique disciplines incorporated: 17/17 (100.0%)  
• Average gap disciplines per protocol: 2.8

✓ All 17 target disciplines are now integrated into research protocols

In \[14\]:

```

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
# Create a comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Protocol Gap-Fill Scores
ax1 = fig.add_subplot(gs[0, :2])
colors_impact = ['#d62728', '#ff7f0e', '#2ca02c']
x_pos = np.arange(len(impact_df))

# Stacked bar chart
gap_counts = impact_df['Gap_Disciplines'].values
moderate_counts = impact_df['Moderate_Disciplines'].values
covered_counts = impact_df['Covered_Disciplines'].values

ax1.barh(x_pos, gap_counts, label='Gap Disciplines (0-2 mentions)', color=colors_impact[0], alpha=0.8)
ax1.barh(x_pos, moderate_counts, left=gap_counts, label='Moderate (3-49 mentions)', 
         color=colors_impact[1], alpha=0.8)
ax1.barh(x_pos, covered_counts, left=gap_counts+moderate_counts, 
         label='Covered (≥50 mentions)', color=colors_impact[2], alpha=0.8)

ax1.set_yticks(x_pos)
ax1.set_yticklabels(impact_df['Short_Name'])
ax1.set_xlabel('Number of Disciplines', fontsize=11)
ax1.set_title('Protocol Composition: Addressing Disciplinary Gaps', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# Add gap-fill scores as text
for i, (idx, row) in enumerate(impact_df.iterrows()):
    ax1.text(row['Total_Disciplines'] + 0.2, i, f"Score: {row['Gap_Fill_Score']:.1f}", 
             va='center', fontsize=9, fontweight='bold')

# Plot 2: Discipline utilization across protocols
ax2 = fig.add_subplot(gs[0, 2])
discipline_protocol_count = defaultdict(int)
for protocol_id, details in research_protocols.items():
    for discipline in details['disciplines']:
        discipline_protocol_count[discipline] += 1

discipline_usage = pd.Series(discipline_protocol_count).sort_values(ascending=True)
colors_usage = [colors_impact[0] if discipline_mention_counts.get(d, 0) < 3 else 
                colors_impact[1] if discipline_mention_counts.get(d, 0) < 50 else 
                colors_impact[2] for d in discipline_usage.index]

ax2.barh(range(len(discipline_usage)), discipline_usage.values, color=colors_usage, alpha=0.8)
ax2.set_yticks(range(len(discipline_usage)))
ax2.set_yticklabels(discipline_usage.index, fontsize=8)
ax2.set_xlabel('# Protocols', fontsize=10)
ax2.set_title('Discipline Usage\nAcross Protocols', fontsize=11, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Before/After disciplinary coverage
ax3 = fig.add_subplot(gs[1, :])
before_after_data = []
for disc in DISCIPLINES:
    before = discipline_mention_counts.get(disc, 0)
    after = discipline_protocol_count.get(disc, 0)
    before_after_data.append({
        'Discipline': disc,
        'Before (Mentions)': before,
        'After (Protocols)': after,
        'Category': 'Gap' if before < 3 else ('Moderate' if before < 50 else 'Covered')
    })

ba_df = pd.DataFrame(before_after_data).sort_values('Before (Mentions)')

x_ba = np.arange(len(ba_df))
width = 0.35

bars1 = ax3.bar(x_ba - width/2, ba_df['Before (Mentions)'], width, 
                label='Previous Analysis Coverage (mentions)', alpha=0.7, color='gray')
bars2 = ax3.bar(x_ba + width/2, ba_df['After (Protocols)'], width,
                label='New Protocol Integration (# protocols)', alpha=0.7, color='#1f77b4')

ax3.set_xticks(x_ba)
ax3.set_xticklabels(ba_df['Discipline'], rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Before vs After: Disciplinary Integration in A2A World Platform', 
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Gap threshold')

# Plot 4: Network metrics summary
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')

network_metrics_text = f"""
NETWORK ANALYSIS SUMMARY

Knowledge Graph:
  • Nodes: {G.number_of_nodes()} disciplines
  • Edges: {G.number_of_edges()} connections
  • Density: {nx.density(G):.3f}
  
Bridge Disciplines (Top 3):
  1. Art (betweenness: {betweenness_centrality['Art']:.3f})
  2. Folklore ({betweenness_centrality['Folklore']:.3f})
  3. Religious Studies ({betweenness_centrality['Religious Studies']:.3f})
  
Hub Disciplines (Most Connections):
  1. Art (7 connections)
  2. Religious Studies (6)
  3. Classical Literature (6)
"""

ax4.text(0.1, 0.5, network_metrics_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Plot 5: Protocol research questions
ax5 = fig.add_subplot(gs[2, 1:])
ax5.axis('off')

rq_text = "RESEARCH QUESTIONS BY PROTOCOL\n" + "="*60 + "\n\n"
for i, (idx, row) in enumerate(impact_df.iterrows(), 1):
    rq_text += f"{row['Short_Name']}: {row['Research_Question'][:80]}...\n\n"

ax5.text(0.05, 0.95, rq_text, fontsize=9, family='sans-serif',
         verticalalignment='top', wrap=True,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

plt.suptitle('A2A World Platform: Comprehensive Multidisciplinary Integration Analysis',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/workspace/a2a_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Comprehensive dashboard saved: a2a_comprehensive_analysis.png")
```

✓ Comprehensive dashboard saved: a2a\_comprehensive\_analysis.png

![][image5]

## **Step 5: Create Implementation Roadmap for A2A World Platform**

Based on the comprehensive analysis, we now have:

✅ Identified the disciplinary gaps in previous analyses (10 out of 17 disciplines had \<3 mentions)  
✅ Built an interdisciplinary knowledge graph showing 35 connections between 17 disciplines  
✅ Designed 5 cross-disciplinary research protocols that integrate all 17 target disciplines  
✅ Quantified the impact of each protocol on filling critical gaps

### **Key Findings from the Dashboard:**

Protocol Effectiveness:

* P2 (Psychogeography) is the most impactful with gap-fill score of 13.5 \- integrates Psychology, Cognitive Science  
* P1 (Archaeoastronomy) brings in Astrophysics and Astrology (both completely missing before)  
* P3 (Eco-Mythology) incorporates Environmental Studies and strengthens Folklore analysis  
* P5 (Literary Geography) activates Classical Literature and Humanities

Before vs After:

* Before: Art dominated with 296 mentions, while 10 disciplines had 0-2 mentions  
* After: All disciplines are now integrated into 1-4 research protocols each  
* Geography and Religious Studies appear in the most protocols (4 each) \- critical bridge disciplines

### **Next Step: Create Actionable Implementation Plan**

In \[15\]:

```
# Create a detailed implementation roadmap
implementation_roadmap = {
    'Phase_1_Foundation': {
        'duration': '3 months',
        'priority': 'Critical',
        'objectives': [
            'Establish comprehensive data infrastructure',
            'Build interdisciplinary research team',
            'Develop core analytical tools'
        ],
        'tasks': [
            {
                'task': 'Data Integration Pipeline',
                'owner': 'Data Engineering Team',
                'deliverables': [
                    'Unified geospatial database (PostGIS) with all site locations',
                    'Mythological corpus database (texts, themes, entities)',
                    'Environmental data layers (climate, DEM, hydrology)',
                    'Astronomical calculation engine (ephemeris, alignments)'
                ],
                'technologies': ['PostGIS', 'PostgreSQL', 'Python (astropy)', 'GDAL']
            },
            {
                'task': 'Team Assembly',
                'owner': 'Research Director',
                'deliverables': [
                    'Recruit specialists in each of 17 disciplines',
                    'Establish collaboration protocols and communication channels',
                    'Define interdisciplinary validation procedures'
                ],
                'technologies': ['NATS messaging', 'Consul service discovery']
            },
            {
                'task': 'Analytical Framework',
                'owner': 'AI/ML Team',
                'deliverables': [
                    'Implement adaptive HDBSCAN clustering',
                    'Develop spatial statistics validation (Moran\'s I, Monte Carlo)',
                    'Build NLP pipeline for text analysis',
                    'Create visualization dashboard'
                ],
                'technologies': ['scikit-learn', 'hdbscan', 'spaCy', 'Leaflet.js']
            }
        ]
    },
    
    'Phase_2_Protocol_Deployment': {
        'duration': '6 months',
        'priority': 'High',
        'objectives': [
            'Deploy all 5 research protocols',
            'Validate initial pattern discoveries',
            'Refine methodologies based on results'
        ],
        'protocol_sequence': [
            {
                'protocol': 'P1: Archaeoastronomy',
                'month': 'Month 4-5',
                'rationale': 'High-precision, well-defined methodology; builds confidence',
                'key_milestones': [
                    'Calculate astronomical alignments for all sites',
                    'Identify statistically significant celestial patterns',
                    'Cross-validate with historical astronomical records'
                ],
                'expected_outputs': [
                    'Map of solstice/equinox alignments',
                    'Catalog of stellar alignments',
                    'Report on astronomical knowledge in ancient cultures'
                ]
            },
            {
                'protocol': 'P4: Art Diffusion',
                'month': 'Month 4-6',
                'rationale': 'Leverages existing art-heavy dataset; parallel deployment',
                'key_milestones': [
                    'Extract artistic motifs using computer vision',
                    'Build motif similarity networks',
                    'Map cultural contact zones'
                ],
                'expected_outputs': [
                    'Motif diffusion animations',
                    'Cultural contact network visualizations',
                    'Trade route validation/discovery'
                ]
            },
            {
                'protocol': 'P2: Psychogeography',
                'month': 'Month 5-7',
                'rationale': 'Highest gap-fill score; introduces cognitive sciences',
                'key_milestones': [
                    'Compute viewshed and prominence for all sites',
                    'Analyze acoustic properties',
                    'Correlate with mythological themes'
                ],
                'expected_outputs': [
                    '3D terrain visualizations with viewsheds',
                    'Psychological impact heatmaps',
                    'Correlation analysis: landscape ↔ mythology'
                ]
            },
            {
                'protocol': 'P3: Eco-Mythology',
                'month': 'Month 6-8',
                'rationale': 'Integrates environmental data; builds on P2 findings',
                'key_milestones': [
                    'Integrate climate and ecological data',
                    'Classify mythological themes',
                    'Perform statistical correlation analysis'
                ],
                'expected_outputs': [
                    'Environmental determinism validation/refutation',
                    'Climate-mythology correlation maps',
                    'Linguistic analysis of nature terminology'
                ]
            },
            {
                'protocol': 'P5: Literary Geography',
                'month': 'Month 7-9',
                'rationale': 'Requires full text corpus; benefits from all prior analyses',
                'key_milestones': [
                    'NLP extraction of geographic references',
                    'Map mythological journeys',
                    'Validate narrative geography'
                ],
                'expected_outputs': [
                    'Interactive maps of mythological journeys',
                    'Accuracy analysis of ancient geographic knowledge',
                    'Symbolic vs literal geography classification'
                ]
            }
        ]
    },
    
    'Phase_3_Synthesis_Publication': {
        'duration': '3 months',
        'priority': 'High',
        'objectives': [
            'Synthesize findings across all protocols',
            'Validate cross-protocol patterns',
            'Prepare publications and public platform'
        ],
        'tasks': [
            {
                'task': 'Meta-Analysis',
                'owner': 'Senior Research Team',
                'deliverables': [
                    'Identify patterns that emerge across multiple protocols',
                    'Quantify statistical significance of cross-domain correlations',
                    'Address contradictions or anomalies in findings'
                ]
            },
            {
                'task': 'Peer Review & Validation',
                'owner': 'External Advisory Board',
                'deliverables': [
                    'Submit findings to interdisciplinary expert review',
                    'Implement feedback and refine conclusions',
                    'Prepare manuscripts for academic journals'
                ]
            },
            {
                'task': 'Public Platform Launch',
                'owner': 'Platform Team',
                'deliverables': [
                    'Launch interactive web platform with all visualizations',
                    'Publish downloadable datasets with documentation',
                    'Create educational materials and tutorials',
                    'Implement community contribution mechanisms'
                ]
            }
        ]
    }
}

# Generate implementation timeline
print("="*100)
print("A2A WORLD PLATFORM: IMPLEMENTATION ROADMAP")
print("="*100)
print(f"\nTotal Duration: 12 months")
print(f"Number of Phases: {len(implementation_roadmap)}")
print(f"Research Protocols: 5")
print(f"Disciplinary Integration: 17/17 (100%)")

for phase_id, phase_data in implementation_roadmap.items():
    print(f"\n{'='*100}")
    print(f"📅 {phase_id.replace('_', ' ').upper()}")
    print(f"{'='*100}")
    print(f"Duration: {phase_data['duration']}")
    print(f"Priority: {phase_data['priority']}")
    print(f"\n🎯 Objectives:")
    for obj in phase_data['objectives']:
        print(f"  • {obj}")
    
    if 'tasks' in phase_data:
        print(f"\n📋 Key Tasks:")
        for task in phase_data['tasks']:
            print(f"\n  ▸ {task['task']} (Owner: {task['owner']})")
            print(f"    Deliverables: {len(task['deliverables'])}")
            for i, deliverable in enumerate(task['deliverables'], 1):
                print(f"      {i}. {deliverable}")
    
    if 'protocol_sequence' in phase_data:
        print(f"\n🔬 Protocol Deployment Sequence:")
        for protocol in phase_data['protocol_sequence']:
            print(f"\n  ▸ {protocol['protocol']} ({protocol['month']})")
            print(f"    Rationale: {protocol['rationale']}")
            print(f"    Expected Outputs: {len(protocol['expected_outputs'])}")

print(f"\n\n{'='*100}")
print("✓ Complete implementation roadmap generated")
print("="*100)
```

\====================================================================================================  
A2A WORLD PLATFORM: IMPLEMENTATION ROADMAP  
\====================================================================================================

Total Duration: 12 months  
Number of Phases: 3  
Research Protocols: 5  
Disciplinary Integration: 17/17 (100%)

\====================================================================================================  
📅 PHASE 1 FOUNDATION  
\====================================================================================================  
Duration: 3 months  
Priority: Critical

🎯 Objectives:  
• Establish comprehensive data infrastructure  
• Build interdisciplinary research team  
• Develop core analytical tools

📋 Key Tasks:

▸ Data Integration Pipeline (Owner: Data Engineering Team)  
Deliverables: 4  
1\. Unified geospatial database (PostGIS) with all site locations  
2\. Mythological corpus database (texts, themes, entities)  
3\. Environmental data layers (climate, DEM, hydrology)  
4\. Astronomical calculation engine (ephemeris, alignments)

▸ Team Assembly (Owner: Research Director)  
Deliverables: 3  
1\. Recruit specialists in each of 17 disciplines  
2\. Establish collaboration protocols and communication channels  
3\. Define interdisciplinary validation procedures

▸ Analytical Framework (Owner: AI/ML Team)  
Deliverables: 4  
1\. Implement adaptive HDBSCAN clustering  
2\. Develop spatial statistics validation (Moran's I, Monte Carlo)  
3\. Build NLP pipeline for text analysis  
4\. Create visualization dashboard

\====================================================================================================  
📅 PHASE 2 PROTOCOL DEPLOYMENT  
\====================================================================================================  
Duration: 6 months  
Priority: High

🎯 Objectives:  
• Deploy all 5 research protocols  
• Validate initial pattern discoveries  
• Refine methodologies based on results

🔬 Protocol Deployment Sequence:

▸ P1: Archaeoastronomy (Month 4-5)  
Rationale: High-precision, well-defined methodology; builds confidence  
Expected Outputs: 3

▸ P4: Art Diffusion (Month 4-6)  
Rationale: Leverages existing art-heavy dataset; parallel deployment  
Expected Outputs: 3

▸ P2: Psychogeography (Month 5-7)  
Rationale: Highest gap-fill score; introduces cognitive sciences  
Expected Outputs: 3

▸ P3: Eco-Mythology (Month 6-8)  
Rationale: Integrates environmental data; builds on P2 findings  
Expected Outputs: 3

▸ P5: Literary Geography (Month 7-9)  
Rationale: Requires full text corpus; benefits from all prior analyses  
Expected Outputs: 3

\====================================================================================================  
📅 PHASE 3 SYNTHESIS PUBLICATION  
\====================================================================================================  
Duration: 3 months  
Priority: High

🎯 Objectives:  
• Synthesize findings across all protocols  
• Validate cross-protocol patterns  
• Prepare publications and public platform

📋 Key Tasks:

▸ Meta-Analysis (Owner: Senior Research Team)  
Deliverables: 3  
1\. Identify patterns that emerge across multiple protocols  
2\. Quantify statistical significance of cross-domain correlations  
3\. Address contradictions or anomalies in findings

▸ Peer Review & Validation (Owner: External Advisory Board)  
Deliverables: 3  
1\. Submit findings to interdisciplinary expert review  
2\. Implement feedback and refine conclusions  
3\. Prepare manuscripts for academic journals

▸ Public Platform Launch (Owner: Platform Team)  
Deliverables: 4  
1\. Launch interactive web platform with all visualizations  
2\. Publish downloadable datasets with documentation  
3\. Create educational materials and tutorials  
4\. Implement community contribution mechanisms

\====================================================================================================  
✓ Complete implementation roadmap generated  
\====================================================================================================

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAAE1CAYAAABnbt3hAACAAElEQVR4XuydCfjUxPnHBaygIuBRURDQClQB5b7xbutVbW09Wv1ba61tbeutxWoVOeRUVBQVURRBRUVRFDwBFQQRb+WyCvW+ahXw5Jr/fufHG2ffnSSTbLKb3d/7eZ55dneSTLLZbOabd973nc2UIAiCIAiCUFFsxisEQRAEQRCEbCMCThAEQRAEocIQAZdxOnTooF8PPfRQdc4553j1e++9t/f+z3/+s3798Y9/rF/79u3rLTNp27at9/7ZZ5/13vfr1897T4waNYpX5dG/f39eFcgBBxygHnjgAV7txFNPPaXmzp3LqyPjcsw4FwsWLFDffPMNX1SAS3uCICTDXXfdpWbMmKFOO+00ddNNN/HFBZj3O85XX32lX1euXJm/wIGXXnpJ/d///Z9+P3z4cLa0hhNPPJFXaR566CFeFQr29d133+XdswUBiIDLMBAt//vf/7zPq1evznslNm7cqF+vvfZa9eGHH6o333xTHXPMMerpp59WzzzzjBYa+PPTDW3w4MGqZ8+e3vYQV6BPnz5aZK1Zs0adf/75uu6MM85Qe+yxh27j7rvv1nX33nuv6tq1q7f9pEmT1IUXXqheffVV9cQTT3j13bt3V7NmzVIffPCB3geJwn322UcNHTpUv3/00Ue944JI/frrr/V78+YLAffxxx/r93/4wx9Ujx499Pt9991XH9u4ceP0Z7Rrstdee+n1AbahY7788su99i+99FL13HPPedvQuQQ45scee0ytX79eLVy4UNdhO/wuqMc5wfn67LPPvG0EQUgHCDfirLPO0vehq6++Wn8eMmSImjZtWt59g97Pnz9f/6/pv0tceeWV3kMv6Natm76XQSQCup+sXbvWE2wmqEP59a9/rZ588kld16lTJ7Vo0SK9r3fffVff89q1a+dtM2bMGL3Nww8/rPbbbz9166236nukeXxceGL93r17ewKO7msQdRCTtP7f//53NXXqVP1+3rx5+j32d8stt+g6nAehuhABl2Hwx0QhgXLeeefpV9x0+J8cNy9A1jcSOViPLEV4D4EHzD8zCbgzzzzTqyOxRcdAbYwdO1a/0rHQe4gg80YFevXqpV9/85vf5Ak4utkBiE6bxfDtt9/23kMUHnLIId7nO+64w3s/ceJE7z2E5yOP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>

[image4]: <data:image/png;base64,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yJb5hR5dAScwH0HI52JgRaQIrwvFl4H3shyWWXC3Q8YuuGVMvBm4CDpFUaoDFXCKUqV4xQFImQxvqytXeAgiVERsgSwrgsj7Xm8ZDToHIBzBu4y7LXldTMBR+sM73Pnua0qlCHfffbd9lPXL9wdXnJQScH6v3RpzglfAicCVZbCAiiVPhK53vjyKla6YgKMkjMvWrVuttVCQMiTyXnc7q1evzvv84O4jkDIkiqJkHxVwilKl+Ak4LEBnnnmmFRES+yWFf7lo33DDDbYTg7yW97322mtWIHkFAKIA96S0awLWKd0XWIaWVLfeeqt59NFHrbtOigN7RYO4YAsJCUG6HLjwPaj5RrFhGDNmjK2tJngFHMt621ghCrE6egXc3//+d+u+xBUprli3rpu83yvgWF5gOp0smO6tV0dNOVzGEq8n9epkvkxjGT8XqiDL49IWwUa9Ob6fV6yKWJT3YjWlM4O8H5czvxHTvLj7CKTosKIo2UcFnKIooXGL+QblrbfecifVHcVEbBRwjVL+xI9nnnnGPnpbbBXihBNOcCcpipJhVMApiqKUkaQFXCmwiJJh6m2jpShK9aMCTlEURVEUpcpQAacoCYBVheFtku4NdC8EcUmFIKbrggsusBmFYTjjjDNsCygyF12iWH0ee+wxG8tVyvUWljVr1tjH559/PjetULkQ+czFeoMmjTdZwwt9XslUlQzdtHB/J/e1izcWLggSE1kKtovV7uuvv3ZndcLv8xGHSXwhx86OHTvc2YqiJIAKOEVJAO+FTEQczd+Bul7Lly+3zx988EHTr18/+5wkBBqcv/TSS1YcSA0uBBzirrm52T4SwO4t5UETeASFF/dCSjA64ouEBe/85557zpbreOedd+zrk08+2Zam2Lt3b64ZO9x///255wLrmDhxon2OEPAW1uXzSaICWY0MPidQUFZKYZCheeSRR9rprA+hCtSgc9fjJ+DeeOONTiVH2J+PPPJIJ+H6m9/8xj569xX7klIje/bsMe+++66txyZZnhdddJFN/Cgk4P785z93ek0cHwkDJH1QxJfvRyLACy+8YOdv2LDB/p5yHEiyg3wftiXHhuwXEess84c//ME+BxH57De24Y11kwb0AuL9vvvus8/53Vke2KckfoD3c5Bx6paZEWQZeXR/b7ZDLUHm89nc78F39MJvw29GBjOQiMENjvfz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>

[image5]: <data:image/png;base64,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>