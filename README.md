# AMRQA

AMRQA is an implementation of **AMR-guided traceable multi-path reasoning** for multi-hop
question answering. This is the code repository for the paper accepted for publication in
**Expert Systems with Applications**:

> **AMRQA: Abstract meaning representation-guided traceable multi-path reasoning for multi-hop
> question answering**  
> Wentao Deng, Zhenyun Deng, Can Cheng, Wolfgang Mayer, Debo Cheng, and Zaiwen Feng.  
> DOI: [10.1016/j.eswa.2026.134639](https://doi.org/10.1016/j.eswa.2026.134639)

Repository: [hzau-is-lab/AMRQA](https://github.com/hzau-is-lab/AMRQA).

The implementation follows three main stages:

1. extract AMR anchor-to-`amr-unknown` simple paths and verbalize them as soft textual references;
2. generate candidates from the complete path-specific TAO history, then rank them by joint
   AMR-reference similarity and global semantic relevance;
3. retrieve and rerank evidence, aggregate atomic facts, and retain a complete reasoning trace.

The AMR graph determines which textual references are constructed. Candidate comparison is not
graph matching: it is cosine similarity to those references in the BGE embedding space. The LLM
critic receives only the original question and one candidate, while step conditioning enters at
candidate generation through the accumulated Thought, Action, and Observation history.

This repository contains code, configurations, a toy input, annotation schemas, and
reproducibility documentation. It does not include datasets, model checkpoints, credentials,
cached embeddings, manuscript sources, or generated experiment outputs.

## Contents

- AMRQA inference code with complete reasoning-trace export;
- the method configuration and retained ReAct/atomic-fact few-shot demonstrations;
- input adapters for HotpotQA, MuSiQue, and 2WikiMultiHopQA JSONL files;
- normalized EM/F1 evaluation code; and
- a toy input and automated tests for installation verification.

## Installation

Create an isolated Python 3.9+ environment and install the optional components used by the full
pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-paper.txt
pip install -e ".[dev]"
cp .env.example .env
```

Fill `.env` locally with the API credentials and optional local checkpoint locations. It is ignored
by Git and must never be committed. The AMR component requires both an AMR parser and an
AMR-to-text generator checkpoint; the BGE encoder and reranker are also required for a full run.
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) specifies every required setting without
containing any machine-specific path.

## Run

Convert an official JSON array or validate an existing JSONL file without dropping benchmark
fields such as supporting-evidence identifiers:

```bash
amrqa prepare --input /path/to/hotpot_dev.json --output data/processed/hotpot_dev.jsonl
```

The command reports the example count and SHA-256 checksum used in the run manifest. The inference
runner constructs the per-question retrieval index directly from each normalized example's
supplied contexts.

```bash
amrqa run \
  --config configs/default.yaml \
  --data /path/to/musique_dev.jsonl \
  --output outputs/musique_predictions.jsonl \
  --limit 100
```

The default configuration records the Qwen2.5-32B-Instruct setting reported in the manuscript:
six reasoning steps, three generated candidates, path width `k=2`, top-five retrieval, top-two
reranking, three atomic facts per observation, and AMR-reference weight `lambda=0.5`. The same
`k` controls both the candidates retained per parent and the global active beam. The run uses a
uniform 180-second per-question timeout and no provider retries. Candidate generation uses eight
retained ReAct demonstrations, and atomic-fact extraction uses four retained demonstrations.
Candidate sub-question generation uses temperature `0.7`; all other LLM calls use temperature
`0.0` for greedy decoding.
Disable either component for an ablation with:

```bash
amrqa run --config configs/default.yaml --data /path/to/data.jsonl \
  --output outputs/no_few_shot.jsonl --set prompts.use_react_demonstrations=false
```

For the atomic-fact ablation, use
`--set prompts.use_atomic_fact_demonstrations=false`.

The controlled history-generation and sentence-observation variants use the same pipeline and can
be selected without changing other components:

```bash
# History-conditioning ablation: remove preceding TAO history only from candidate generation.
amrqa run --config configs/default.yaml --data /path/to/data.jsonl \
  --output outputs/no_history.jsonl --set reasoning.candidate_history_mode=question_only

# Evidence-granularity control: rank native evidence sentences instead of extracted facts.
amrqa run --config configs/default.yaml --data /path/to/data.jsonl \
  --output outputs/sentence_observations.jsonl --set reasoning.observation_mode=sentences
```

The AMR method and reference-source controls are selected explicitly. An AMR run stops if parsing,
path extraction, or graph-to-text generation cannot construct an anchor-to-unknown reference:

```bash
# Main AMRQA method.
amrqa run --config configs/default.yaml --data /path/to/data.jsonl \
  --output outputs/amr.jsonl --set amr.reference_source=amr

# LLM-Ref control: replace the AMR-derived reference set with an LLM decomposition.
amrqa run --config configs/default.yaml --data /path/to/data.jsonl \
  --output outputs/llm_ref.jsonl --set amr.reference_source=llm

# Question-reference plumbing control used by the dependency-free smoke test.
amrqa run --config configs/offline.yaml --data data/demo.jsonl \
  --output outputs/question_ref.jsonl --set amr.reference_source=question
```

The evaluator ablations select the active signals without changing candidate generation or
downstream processing:

```bash
# w/o AMR-RefSim
--set reasoning.candidate_scoring_mode=semantic_only

# w/o LLM-Sem
--set reasoning.candidate_scoring_mode=amr_only

# w/o Dual Eval
--set reasoning.candidate_scoring_mode=none
```

Each prediction stores the question, final answer, AMR references, all explored paths, candidate
scores, the selected path, retrieved passages, provenance-linked atomic facts, and runtime fields.
Every run also writes a `.manifest.json` sidecar containing the expanded configuration and
SHA-256 checksums of the configuration, input, and prediction files. It records only basenames and
environment versions, not absolute paths, endpoints, credentials, usernames, or hostnames.
The runtime fields include end-to-end online latency, LLM and retrieval calls, provider-reported
input/output tokens, iteration count, and a mutually exclusive termination reason. Recompute
answer metrics and summarize the efficiency fields with:

`llm_calls` counts every invocation of the configured chat-completion client, including
sufficiency assessment, candidate generation, semantic scoring, atomic-fact extraction, and final
answer generation. It also includes LLM reference generation when the `llm` reference control is
active.

```bash
amrqa evaluate --predictions outputs/musique_predictions.jsonl
amrqa efficiency --predictions outputs/musique_predictions.jsonl
```

Run the three matched paper seeds, 42, 43, and 44, with:

```bash
./scripts/run_three_seeds.sh configs/default.yaml /path/to/hotpot_dev.jsonl \
  outputs/qwen_hotpot_amrqa
```

The backbone-specific configurations are `configs/default.yaml`,
`configs/gpt-3.5-turbo.yaml`, and `configs/gpt-4o-mini.yaml`.

For supporting-evidence evaluation, selected atomic facts are mapped through their stored
provenance to benchmark-native sentence/evidence identifiers:

```bash
amrqa support-evaluate --predictions outputs/hotpot_predictions.jsonl \
  --data /path/to/hotpot_dev.jsonl --representation facts
```

The paired question-cluster bootstrap, paired randomization test, and Holm correction are exposed
through a single predeclared comparison-family manifest:

```bash
amrqa compare --manifest configs/statistics.example.yaml \
  --output outputs/primary_statistics.json
```

Gold-decomposition ranking metrics are recomputed from saved candidate scores and alignment labels:

```bash
amrqa gold-ranking-evaluate --annotations /path/to/musique_gold_ranking.jsonl
```

The sub-question quality analysis reads explicit sub-questions from saved prediction traces:

```bash
amrqa subquestion-evaluate --predictions outputs/musique_predictions.jsonl
```

For a dependency-free parser/output smoke test only:

```bash
amrqa run --config configs/offline.yaml --data data/demo.jsonl --output outputs/smoke.jsonl
```

The dry-run backend deliberately returns `No-answer`; it validates the data and trace pipeline,
not model quality.

## Documentation

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): run manifest, dependencies, data split,
  model identifiers, seeds, and release checklist.
- [`docs/PAPER_ALIGNMENT.md`](docs/PAPER_ALIGNMENT.md): direct mapping from manuscript operations
  and terminology to implementation points.
- [`docs/METRICS.md`](docs/METRICS.md): exact normalized EM/F1 implementation and aggregation.
- [`docs/ANNOTATIONS.md`](docs/ANNOTATIONS.md): human-label definitions, aggregation, agreement,
  and gold-decomposition ranking protocol.
- [`data/README.md`](data/README.md): accepted benchmark input schemas.

## Citation

If you use this implementation, please cite the paper:

```bibtex
@article{deng2026amrqa,
  title = {AMRQA: Abstract meaning representation-guided traceable multi-path reasoning for multi-hop question answering},
  author = {Deng, Wentao and Deng, Zhenyun and Cheng, Can and Mayer, Wolfgang and Cheng, Debo and Feng, Zaiwen},
  journal = {Expert Systems with Applications},
  year = {2026},
  doi = {10.1016/j.eswa.2026.134639}
}
```

Volume and other final bibliographic details can be added when confirmed by the publisher.

## Public-release check

Run the automated preflight scan before sharing the repository:

```bash
python scripts/check_release.py .
```

Paper authors, affiliations, and intended public contact details may appear in this public
repository. Manually review contact details and `git log --format=fuller` before sharing. Do not
commit credentials, private `.env` files, absolute local paths, local caches, or symlinks whose
target exposes private files. Keep manuscript PDFs out of this code release; share the paper via
its DOI link. The automated scan does not inspect Git history or establish copyright ownership.
