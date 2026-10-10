"""Compact, JSON-constrained prompt templates used by AMRQA."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful multi-hop question-answering assistant. Use only supplied evidence when "
    "deciding whether an answer is supported. Follow the requested output format exactly."
)


def reference_decomposition_prompt(question: str, count: int) -> str:
    return f"""Decompose the multi-hop question into at most {count} concise reference
sub-questions. Together, the sub-questions should express the dependencies needed to answer the
original question. Do not answer them and do not add background knowledge. Return only a JSON
array of strings.

Original question:
{question}
"""


def candidate_prompt(question: str, history: str, count: int, demonstrations: str = "") -> str:
    demonstration_section = ""
    if demonstrations:
        demonstration_section = f"""\nThe following ReAct trajectories demonstrate the intended
multi-hop decomposition style. They are examples only: do not copy their entities, observations,
or answers.\n\n{demonstrations}\n\n"""
    return f"""Task: generate candidate sub-questions.

Given the original question and current reasoning history, generate {count} diverse,
concise, non-overlapping next-hop sub-questions. Each sub-question must fill one missing
information need and be answerable from a retrieval corpus. Return only a JSON array of strings.
{demonstration_section}

Original question:
{question}

Reasoning history:
{history}
"""


def semantic_score_prompt(question: str, candidate: str) -> str:
    return f"""Assess the global semantic relevance and logical compatibility of a candidate
sub-question with the original multi-hop question. Judge whether the candidate addresses a
dependency that can contribute to answering the original question. Do not judge whether the
dependency has already been resolved because no reasoning history is supplied.
Return only one decimal number in [0, 1]. A score of 1 means directly relevant and logically
compatible, while 0 means unrelated or contradictory.

Original question: {question}
Candidate sub-question: {candidate}
Semantic relevance:"""


def sufficiency_prompt(question: str, history: str) -> str:
    return f"""Determine whether the reasoning history contains sufficient evidence to answer the
original question. Do not infer facts not present in the observations. Return only JSON with keys
`sufficient` (boolean) and `thought` (short string). Do not generate the answer in this step.

Original question:
{question}

Reasoning history:
{history}
"""


def atomic_fact_prompt(context: str, demonstrations: str = "") -> str:
    demonstration_section = ""
    if demonstrations:
        demonstration_section = f"""\nExamples:
{demonstrations}
"""
    return f"""Task Description: Extract atomic facts from a paragraph. Each atomic fact must be
a single, standalone, verifiable statement explicitly supported by the paragraph.

Instructions:
- Express exactly one piece of information.
- Write a complete declarative sentence.
- Preserve the original tense and factual detail.
- Do not combine facts, paraphrase, or add information not present in the paragraph.
- Return only a numbered list of atomic facts.
{demonstration_section}
paragraph: {context}
atomic facts:
"""


def final_answer_prompt(question: str, history: str) -> str:
    return f"""Answer the original question using only the reasoning history. Return only the final
short answer, with no explanation. If the history is insufficient, return `No-answer`.

Original question:
{question}

Reasoning history:
{history}
"""
