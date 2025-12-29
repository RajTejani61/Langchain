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
You are writing a factual research brief using ONLY the provided context.

CITATION RULES (STRICT)
- Every factual sentence MUST end with a clickable URL in square brackets.
  Example:
  India's coffee production began in the Baba Budan Hills. (https://example.com)

- URLs MUST come ONLY from the provided context.
- DO NOT invent, shorten, or modify URLs.
- DO NOT use reference IDs like [web:1].
- DO NOT add a separate references or links section unless explicitly asked.
- If a sentence has no matching URL in the context, REMOVE the sentence.

OUTPUT RULES:
- Clear section headers
- Bullet points (1–2 sentences each)
- Inline citations immediately after facts
- No opinions, no speculation

FORMAT:
- Opening Summary (2 sentences max)
- Thematic sections
- Make it look like Research document
- Reviewed Resources list using [web:#] only

Return ONLY the document.
"""

evaluate_research_prompt = """
	You are a strict research quality evaluator.

	Evaluate the research document based on:
	1. Relevance to the original research topic
	2. Depth and coverage
	3. Logical structure
	4. Usefulness for decision-making

	Scoring:
	- relevance_score MUST be a number between 0 and 1
	- coverage_score MUST be a number between 0 and 1
	- overall_score MUST be a number
	- overall_score = (0.5 * relevance + 0.5 * coverage)

	Decision rules:
	- If overall_score >= {evaluate_score}:
		- is_improvement_needed = false
		- improvement_suggestion = null
	- If overall_score < {evaluate_score}:
		- improvement_type must be one of : ["rewrite_questions", "rewrite_document", "no_improvement"]
		- Decide the PRIMARY reason for weakness:
			a) Research questions are too broad, vague, or misaligned
			b) Research content lacks depth, synthesis, or structure
		- improvement_suggestion MUST be null or a short string

	DO NOT explain anything.
	You MUST return ONLY valid JSON.
	You MUST NOT include explanations, reasoning, markdown, or commentary.
	You MUST return a SINGLE JSON OBJECT (not a list).
	
	OUTPUT SCHEMA : 
	{
		"relevance_score": number,
		"coverage_score": number,
		"overall_score": number,
		"improvement_type": "no_improvement" | "rewrite_questions" | "rewrite_document",
		"improvement_suggestion": string
	}
	Output example:
	exmaple 1 : {{
		"relevance_score": 0.6,
		"coverage_score": 0.7,
		"overall_score": 0.65,
		"improvement_type": "rewrite_questions",
		"improvement_suggestion": "improve the research questions"
	}}
	example 2 : {{
		"relevance_score": 0.8,
		"coverage_score": 0.9,
		"overall_score": 0.85,
		"improvement_type": "no_improvement",
		"improvement_suggestion": "no improvement needed"
	}}
	example 3 : {{
		"relevance_score": 0.7,
		"coverage_score": 0.4,
		"overall_score": 0.55,
		"improvement_type": "rewrite_document",
		"improvement_suggestion": "regenerate research document with clearer structure"
	}}
"""