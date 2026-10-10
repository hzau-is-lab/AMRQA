# Manuscript-to-code alignment

This document maps manuscript terminology and settings to the executable AMRQA implementation.

| Manuscript operation | Implementation |
| --- | --- |
| Parse the original question once | `AMRLibPathProvider.extract` in `src/amrqa/amr.py` |
| Enumerate anchor-to-`amr-unknown` simple paths | `AMRLibPathProvider._simple_paths` |
| Verbalize paths as soft textual references | `AMRLibPathProvider._verbalize` |
| Generate three candidates from complete path-specific TAO history at temperature 0.7 | `AMRQAReasoner._generate_candidates` |
| Compute maximum AMR-reference cosine similarity | `CandidateEvaluator.evaluate` in `src/amrqa/scoring.py` |
| Obtain global semantic relevance from the original question and candidate only | `semantic_score_prompt` in `src/amrqa/prompts.py` |
| Combine both scores with `lambda=0.5` | `reasoning.amr_reference_weight` |
| Retain top `k=2` locally and globally | `reasoning.path_width` |
| Retrieve five passages and rerank to two | `EvidenceRetriever.retrieve` |
| Extract and retain three relevant atomic facts | `AMRQAReasoner._observe` |
| Score a path by the running mean of its selected joint scores | `AMRQAReasoner._extend_score` |
| Select the highest-scoring sufficient path, then generate the answer | `AMRQAReasoner.predict` |
| Return `No-answer` after six unsuccessful iterations | `reasoning.max_depth` and `reasoning.no_answer` |

Graph structure is used to construct the reference set. Candidate-to-reference comparison is
performed in text embedding space and is serialized as `amr_reference_similarity`. Step context
is supplied by candidate generation through `ReasoningPath.history`; the semantic critic receives
no TAO history.
Candidate sub-question generation uses temperature 0.7 to support diverse proposals. Semantic
evaluation, LLM-generated reference controls, sufficiency assessment, atomic-fact extraction, and
final answer generation use temperature 0.0.

The default trace includes all scored candidates, retained explored paths, retrieved passages,
atomic facts with source-unit provenance, the selected complete path, and resource/termination
metadata. The 180-second timer covers the online stages from AMR parsing through final answer
generation. Model loading, corpus/index construction, and process initialization are excluded.

The controlled variants are exposed as configuration changes:

- `reasoning.candidate_history_mode=question_only` removes preceding TAO history from candidate
  generation while leaving scoring and downstream processing unchanged.
- `reasoning.observation_mode=sentences` replaces atomic facts with ranked benchmark-native
  evidence sentences while leaving retrieval and path reasoning unchanged.
- `amr.reference_source=llm` replaces AMR-derived references with an LLM-generated decomposition
  while leaving candidate generation, scoring, retrieval, and path retention unchanged.
- `reasoning.candidate_scoring_mode=semantic_only`, `amr_only`, or `none` reproduces the
  w/o AMR-RefSim, w/o LLM-Sem, and w/o Dual Eval controls without constructing or calling an
  evaluator signal that the corresponding control removes.

An AMR run stops after an AMR parsing, path-extraction, or graph-to-text error. The
`amr.reference_source=question` option activates the question-reference control and is recorded in
the saved configuration.
