# Answer metrics

`amrqa evaluate` recomputes exact match (EM) and token F1 from a JSONL prediction trace. Both
metrics are reported on a 0–100 scale and averaged once per example.

Before comparison, both prediction and reference are normalized by:

1. converting to lowercase;
2. removing ASCII punctuation;
3. removing the English articles `a`, `an`, and `the`;
4. collapsing consecutive whitespace.

EM is 100 when the normalized strings match, otherwise 0. Token F1 uses multiset token overlap:

```text
precision = overlap / predicted-token-count
recall    = overlap / reference-token-count
F1        = 2 * precision * recall / (precision + recall)
```

Empty prediction/reference pairs receive F1 = 1 only when both are empty; otherwise F1 = 0.

The evaluator uses the `prediction` and `gold_answer` fields emitted in every trace. It does not
silently choose a different reference, filter failed cases, or substitute a library metric. If a
benchmark has aliases or multiple gold answers, resolve that policy before the run and document it
in the run manifest; the current input adapter accepts one gold answer per record.

## Supporting evidence

`amrqa support-evaluate` computes precision, recall, and F1 per question and then averages each
metric over the dataset. HotpotQA contexts retain sentence identifiers of the form `[title,
sentence_index]`; MuSiQue contexts retain their native paragraph/evidence identifier. Every
selected atomic fact is linked to one native unit in its source passage using maximum BGE cosine
similarity, and duplicate unit identifiers are removed before evaluation. Use `--representation
facts` for the atomic-fact variant and `--representation passages` to score all native units inside
the selected retrieved passages.

## Paired statistical comparison

`amrqa compare` consumes a YAML manifest containing matched seed-level JSONL files. For each
question and metric, it averages the three seed-level outcomes within each method and computes the
paired AMRQA-minus-baseline difference. Pointwise 95% confidence intervals use 10,000 paired
question-cluster bootstrap resamples. Two-sided p-values use 100,000 paired randomization trials
with plus-one correction. Holm adjustment is performed jointly over every test declared in the
manifest; therefore the complete 54-test family must be present when reproducing the manuscript.

## Sub-question quality

`amrqa subquestion-evaluate` computes length-normalized GPT-2 negative log-likelihood and the
cosine similarity between each generated sub-question and its original question using
`sentence-transformers/all-MiniLM-L6-v2`. The released binary form check requires a question-word
or auxiliary opener and a terminal question mark. All three values are averaged over the explicit
sub-questions in the supplied trace. Methods that do not emit explicit sub-questions are reported
as not applicable rather than assigned a synthetic score.
