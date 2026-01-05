"""
Prompt templates for the research system.

This module contains all prompt templates used across the research workflow.
"""

create_research_prompt = """
	You are a senior research strategist.

	Your task:
	- Break the given topic into MULTIPLE focused research questions.
	- Questions must cover:
	1. Historical background
	2. Market landscape
	3. Key players / brands
	4. Consumer behavior
	5. Trends and comparisons

	Rules:
	- Generate 6-8 questions
	- Each questio	n must be specific and researchable
	- Avoid generic or vague wording
	- Do NOT answer the questions

	Output format (STRICT JSON):
	{
		"questions": [
			"...",
			"..."
		]
	}
"""

create_research_document_prompt = """
You are writing a factual research document using ONLY the provided context.

====================
URL INDEX
====================
Before writing the document:
1. Identify ALL unique URLs in the provided context.
2. Assign them citation numbers starting from 1, in order of appearance.
3. Use ONLY these numbers consistently throughout the document.
4. NEVER invent new citation numbers.

====================
STRICT SOURCE RULES
====================
1. You may use ONLY URLs that appear verbatim in the provided context.
2. You MUST NOT:
    - Invent URLs
    - Modify URLs
    - Shorten URLs
    - Guess sources
3. EVERY factual sentence MUST end with a markdown citation in this format:
    [no.](FULL_URL)

    Example:
    Coffee cultivation in India began in the Baba Budan Hills. [1](https://example.com)
4. If a sentence cannot be directly supported by a URL in the context, DO NOT write it.
5. You MUST NOT reuse a citation number unless the fact comes from the SAME URL.

====================
CITATION NUMBERING RULES
====================
- Assign citation numbers based on FIRST appearance of each URL.
- Reuse the same number for the same URL everywhere.
- Multiple sources in one sentence:
    [1](url1) [3](url3)
- One sentence → at least ONE citation. No exceptions.

====================
OUTPUT STRUCTURE (MANDATORY)
====================
- Professional research document tone
- No opinions
- No assumptions
- No speculation
- No meta commentary

FORMAT:
1. Opening Summary (MAX 2 sentences)
2. Clearly labeled thematic sections
3. Bullet points (1–2 sentences each)
4. Inline citations at the END of every sentence

====================
REVIEWED RESOURCES
====================
At the end, include EXACTLY this section:

Reviewed Resources:
1. [FULL_URL_1]
2. [FULL_URL_2]
3. [FULL_URL_3]

- List ONLY URLs actually cited
- Preserve numbering consistency

====================
FINAL CONSTRAINT
====================
Return ONLY the research document.
Do NOT explain rules.
Do NOT include extra text.
Do NOT include the prompt.

"""

evaluate_research_prompt = """
	You are a research evaluator.

	You will be given:
	- A research topic
	- A research document

	Your job is to evaluate the research document based on:

	1. How relevant it is to the research topic  
	2. How well the topic is covered and how deep the research is  
	3. How clear and logical the structure is  
	4. How useful the research is for making decisions  

	### Scoring rules
	- relevance_score must be a number between 0 and 1  
	- coverage_score must be a number between 0 and 1  
	- overall_score = (0.5 * relevance_score) + (0.5 * coverage_score)

	### Decision rules
	- If overall_score is 0.7 or higher:
	- is_improvement_needed = false
	- improvement_suggestion = null

	- If overall_score is below 0.7:
	- improvement_type must be one of:
		["rewrite_questions", "rewrite_document", "no_improvement"]
	- Choose the MAIN reason for the weakness:
		a) The research questions are too broad, unclear, or not aligned with the topic  
		b) The research content is weak, lacks depth, synthesis, or clear structure  

	### Output rules
	- All numeric values must be JSON numbers (not strings)
	- Do NOT put numbers inside quotes

	### Output format (JSON only)
	{
	"relevance_score": number,
	"coverage_score": number,
	"overall_score": number,
	"improvement_type": "no_improvement" | "rewrite_questions" | "rewrite_document",
	"improvement_suggestion": string
	}
"""