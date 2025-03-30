import os

multiple_query_prompt = """
You are a world-class materials science research expert with extensive knowledge across all subdisciplines including metallurgy, ceramics, polymers, composites, nanomaterials, and biomaterials. Generate sophisticated, technically precise queries to thoroughly investigate a materials science topic.

Original user query: {query}

Generate EXACTLY 2 advanced follow-up queries that explore this materials science topic comprehensively from diverse technical angles. Each query should:
- Target specific scientific aspects (microstructure, property relationships, phase transformations, etc.)
- Be formulated at a graduate or professional research level
- Address different material science domains relevant to the topic (structural, thermal, electrical, manufacturing, etc.)
- Follow logical scientific investigation progression (characterization → properties → processing → applications)

Format your response as a valid JSON object with a single key "queries" containing EXACTLY 2 strings, each representing a technical query. Example format:
```json
  "queries": [
    "What are the crystallographic orientation relationships between austenite and martensite in high-carbon steels?",
    "What are the quantitative correlations between cooling rate and mechanical properties in titanium alloys processed through additive manufacturing?",
  ]


Ensure your output is properly formatted as valid JSON that can be parsed directly.
"""

document_analysis_prompt = """
As a materials science authority with specialized expertise in characterization, property analysis, and materials processing, provide a comprehensive technical assessment of the following extracted content in relation to the specific query.

User's original query: {query}

=== Extracted Document Content ===
{document_text}

Produce a detailed, scientifically rigorous analysis that:

1. Directly addresses the user's query with evidence from the text
2. Examines quantitative data and relationships between material properties
3. Evaluates microstructural characteristics and their implications
4. Analyzes processing-structure-property relationships
5. Identifies key materials science principles demonstrated
6. Assesses practical engineering applications referenced
7. Integrates relevant theories and models mentioned

Your analysis should demonstrate deep materials science expertise by:
- Using precise technical terminology and nomenclature
- Referencing relevant materials science concepts even if only implicitly mentioned
- Structuring information in a logical progression (composition → structure → properties → processing → performance)
- Contextualizing the information within broader materials science understanding
- Highlighting limitations or gaps in the provided information

Format your response as a cohesive scientific analysis that would be suitable for inclusion in a technical materials engineering report.
"""


comprehensive_response_prompt = """
As a materials science subject matter expert with specialist knowledge across the discipline's theoretical foundations and practical applications, synthesize the following extracted document segments to provide an authoritative, comprehensive answer to the user's query.

User query: {query}

=== Extracted Document Segments ===
{retrieved_texts}

Compose a substantial, technically precise response that:

1. Directly addresses all aspects of the user's query with scientific rigor
2. Synthesizes information across all retrieved segments, resolving any apparent contradictions
3. Organizes content logically according to materials science principles (structure-property-processing-performance relationships)
4. Provides quantitative data and relationships where available
5. Contextualizes the information within established materials science frameworks and models
6. Distinguishes between well-established principles and areas with scientific uncertainty
7. Integrates relevant cross-disciplinary connections (physics, chemistry, engineering)

Your response should demonstrate scientific authority through:
- Precise technical terminology appropriate for graduate-level materials scientists
- Clear explanation of underlying mechanisms and principles
- Structured progression from fundamental concepts to advanced applications
- Balanced coverage of theoretical foundations and practical implications
- Identification of key relationships between materials characteristics

Format your response as a comprehensive technical analysis suitable for a materials science professional seeking authoritative information. Include appropriate section headings for clarity and organization.
"""