# Annotation protocols

The paper uses separate annotation tasks for atomic-fact quality, next-step validity,
AMR-reference auditing, and failure analysis. Three doctoral researchers in natural language
processing independently annotate the same items in each task. Agreement is computed from their
individual labels before aggregation. No annotator names or affiliations are stored in this
public release, to preserve annotator privacy.

## Atomic-fact quality

One hundred extracted facts are sampled from each of HotpotQA, MuSiQue, and 2WikiMultiHopQA.
Each fact receives independent ratings from 1 to 5 for correctness, relevance, and completeness.
Dataset-level table entries are means over the three ratings. Agreement is reported over all 300
facts using ordinal Krippendorff's alpha and three-way exact agreement.

## Next-step and repeat-hop labels

Reasoning states are sampled from saved AMRQA traces. A reasoning state consists of the root
question, the path-specific preceding Thought, Action, and Observation history, and the three
generated candidates. The three nominal labels are:

- `valid_next_step`: advances an unresolved dependency toward the final answer;
- `repeat_hop`: asks for an entity relation that has already been resolved in the path history or
  whose answer is already present in the accumulated observations;
- `other_invalid`: neither a valid advance nor a repeated resolved hop.

The final label is the majority vote. Nominal Krippendorff's alpha is computed before voting. The
history-generation comparison uses the combined 936-candidate pool from 156 Step-2+ states under
the full and question-only variants. Step 1 is excluded because no preceding TAO history exists.

## MuSiQue gold-decomposition alignment

A candidate is `gold_aligned` when it asks for the unresolved component represented by the next
MuSiQue gold-decomposition step. Ranking diagnostics are conditional on a reasoning state
containing at least one such candidate. Multiple candidates in one state may align with the same
gold component. Pairwise accuracy is micro-averaged over every eligible aligned/non-aligned pair;
a correctly ordered pair scores one, a reversed pair zero, and a score tie one half.

The command below recomputes availability, aligned-candidate multiplicity, conditional Hit@1,
Hit@2, reciprocal rank, and the raw correct/tied/incorrect pair counts:

```bash
amrqa gold-ranking-evaluate --annotations /path/to/musique_gold_ranking.jsonl
```

The accepted JSONL schema is illustrated by `data/gold_ranking.example.jsonl`. The public
analysis file should contain one object per state, three candidates per object, a boolean
`gold_aligned` label, and the three saved scores.

## AMR-reference audit

The audit uses a binary label indicating whether the AMR-derived reference contains a material
error. An instance receives this label when the generated AMR graph or its verbalized reference
sub-questions misrepresent a semantic dependency expressed in the original question. This label
applies when a required entity or relation is omitted, an entity or relation is encoded
incorrectly, a predicate-argument attachment or relation direction is incorrect, or a reference
sub-question is semantically inconsistent with the original question. Surface-form variation or
minor disfluency is not labeled as an error unless it changes or obscures a dependency required
to answer the question. Annotators inspect only the original question, generated AMR graph, and
AMR-derived references. They do not access source contexts, gold answers, gold supporting-evidence
annotations, model predictions, evaluation scores, or system reasoning traces.
Instances selected by majority vote are corrected by consensus. Original and refined runs use the
same fixed audit instances and hold every non-reference component constant.

## Failure categories

For AMR-related failure screening, annotators first apply a binary nominal label. The primary
failure-stage analysis then assigns exactly one of four labels using the earliest stage whose
correction would most likely prevent the failure: `AMR parsing`, `sub-question selection`,
`retrieval evidence`, or `path aggregation`. Majority vote produces the reported label, and
nominal Krippendorff's alpha is computed on the individual labels.
