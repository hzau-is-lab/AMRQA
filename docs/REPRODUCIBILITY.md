# Reproducibility protocol

Each reported score should retain its configuration file, input checksum, environment capture,
and output JSONL trace.

## Evaluation scope

The paper evaluates English multi-hop QA on the following supplied JSONL files:

| Dataset | Reported evaluation examples | Accepted source layout |
| --- | ---: | --- |
| HotpotQA | 7,405 | `context: [[title, [sentence, ...]], ...]` |
| MuSiQue | 2,417 | `paragraphs: [{title, paragraph_text}, ...]` |
| 2WikiMultiHopQA | 12,576 | `context: [[title, [sentence, ...]], ...]` |

Use the original benchmark releases and record the release URL/version, split name, licence, and
SHA-256 checksum in a run manifest. The repository does not redistribute these datasets or a
Wikipedia retrieval corpus. The current implementation retrieves only from the contexts supplied
with each example; a different corpus or preprocessing procedure is a different experimental
condition and must be reported as such.

## Required components

| Component | Paper configuration | Configuration field / local setting |
| --- | --- | --- |
| Reasoning model | Qwen2.5-32B-Instruct, GPT-3.5-Turbo, or GPT-4o-mini | `model.model_name` |
| Embedding model | `BAAI/bge-base-en-v1.5` | `embedding.model_name` and `AMRQA_EMBEDDING_MODEL_DIR` if local |
| Reranker | `BAAI/bge-reranker-v2-m3` | `reranker.model_name` and `AMRQA_RERANKER_MODEL_DIR` if local |
| AMR parser | AMRlib structure-to-graph checkpoint | `AMRQA_STOG_MODEL_DIR` if not in AMRlib's default location |
| AMR generator | AMRlib graph-to-text checkpoint | `AMRQA_GTOS_MODEL_DIR` if not in AMRlib's default location |

The repository stores only environment-variable *names*, never checkpoint paths. Copy
`.env.example` to a local ignored `.env` file and fill it on the executing machine. Before
releasing a configuration or log, remove endpoint URLs, API credentials, usernames, and paths.

`requirements-paper.txt` records the direct package versions from the reference environment.
Retain `pip freeze` and the driver/CUDA information for each final run because AMR and model
packages also install transitive dependencies.

## Method configuration

`configs/default.yaml` is the paper-method configuration:

| Parameter | Value |
| --- | ---: |
| matched random seeds | 42, 43, 44 |
| maximum reasoning depth | 6 |
| candidate subquestions per step | 3 |
| candidate history | complete path-specific TAO history |
| path width $k$ for local retention and global beam | 2 |
| retrieval / reranking depth | 5 / 2 |
| atomic facts per observation | 3 |
| observation representation | atomic facts |
| AMR-reference similarity weight $\lambda$ | 0.5 |
| candidate scoring mode | joint AMR-reference similarity and semantic relevance |
| reference source | AMR anchor-to-unknown paths (`amr`) |
| maximum retained AMR references | 8 |
| ReAct demonstrations | enabled (8 trajectories) |
| atomic-fact demonstrations | enabled (4 examples) |
| atomic-fact output limit | 1,024 tokens |
| candidate-generation temperature | 0.7 |
| all other LLM-call temperatures | 0.0 (greedy decoding) |
| model output limit | 1,024 tokens for Qwen2.5-32B-Instruct and GPT-3.5-Turbo; 512 for GPT-4o-mini |
| provider token-usage response | required |
| per-question wall-clock timeout | 180 seconds |
| provider retries | 0 |

Use `scripts/run_three_seeds.sh` to execute seeds 42, 43, and 44 under one configuration. Use a
copied YAML configuration for every ablation and name it after the factor varied. Command-line
overrides are convenient for local work, but save the final expanded settings beside the trace.

The online timing boundary starts immediately before AMR parsing and ends when the final answer or
termination status is returned. It includes candidate generation and scoring, retrieval and
reranking, atomic-fact extraction, path aggregation, and final answer generation. Corpus/index
construction, model loading, and process initialization are excluded. Timeout and maximum-step
termination are stored as mutually exclusive reasons. A run is a timeout only when it exceeds 180
seconds; it is a maximum-step termination only when it reaches six iterations first. The
`llm_calls` trace field counts every chat-completion invocation in these stages, including
sufficiency assessment, candidate generation, semantic scoring, atomic-fact extraction, and final
answer generation.

## Run record

For each reported result, retain a machine-independent manifest without credentials or personal
machine identifiers, containing:

```text
dataset and official release/split/checksum
configuration-file checksum and fully expanded configuration
model identifier, provider, endpoint type, candidate and non-candidate temperatures, and decoding limit
AMR, embedding, and reranker checkpoint identifiers or revisions
Python/package versions, GPU model/driver, and random seed
start/end time, number of evaluated examples, and trace-file checksum
EM/F1 command output, efficiency summary, and failure/retry policy
```

For the primary statistical family, list all AMRQA comparisons with DEC, TRQA, and GenGround in
one manifest before analysis. `amrqa compare` averages each question's metric across the three
matched seeds, forms paired question-level differences, computes pointwise intervals with 10,000
question-cluster bootstrap resamples, computes two-sided p-values with 100,000 paired
randomization trials and plus-one correction, and applies Holm correction jointly to the declared
family. The complete manuscript family contains 54 tests.

Generate an environment capture locally with `python --version`, `pip freeze`, and (when
applicable) `nvidia-smi`. The model provider can change a hosted model behind a stable name;
therefore provider date/revision and raw traces are necessary to assess reproducibility.

## Verification sequence

1. Install the optional dependencies and configure local secrets/checkpoints outside Git.
2. Run the offline smoke test from the top-level README.
3. Run a small fixed subset with `configs/default.yaml`; inspect the output JSONL trace manually.
4. Run the complete intended split and retain the trace before computing metrics.
5. Recompute EM/F1 with `amrqa evaluate --predictions ...` and resource use with
   `amrqa efficiency --predictions ...`.
6. Run supporting-evidence, gold-ranking, or paired statistical analysis from the saved traces
   when applicable.
7. Run the test suite: `pytest`; run static checks: `ruff check .`.
8. Run `python scripts/check_release.py .` after removing local caches from the release tree
   and before upload. Review public contact details and Git history separately.

The smoke test demonstrates only software plumbing. It is not evidence that a full model run
reproduces a paper number.
