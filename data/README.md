# Data layout

Datasets are deliberately not redistributed with this repository. Download the development splits
from the official [HotpotQA](https://hotpotqa.github.io/),
[MuSiQue](https://github.com/stonybrooknlp/musique), and
[2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) releases and retain their original
licence and version information in the local run record.

Use `amrqa prepare --input SOURCE --output data/processed/DATASET.jsonl` to validate an official
JSON/JSONL export, preserve its source annotations, and record a content checksum.

The runner accepts JSON Lines (one example per line) with this minimal schema:

```json
{
  "id": "optional-id",
  "question": "Who is the mother of the director of Film X?",
  "answer": "Example Answer",
  "context": [["Document title", ["Sentence one.", "Sentence two."]]]
}
```

The loader also accepts `flat_contexts` (a list of strings), HotpotQA/2Wiki-style contexts, and
MuSiQue `paragraphs` records containing `title` plus `paragraph_text`. You may keep the original
files in any local data directory and pass an absolute path to `--data`; alternatively place them
under `data/raw/`, which is ignored by Git. The code never requires precomputed embeddings: it
builds a per-question retrieval index from the provided contexts.

For supporting-evidence evaluation, keep the original annotations in the input. HotpotQA and
2WikiMultiHopQA records may provide `supporting_facts` as `[title, sentence_index]` pairs. MuSiQue
paragraph records should retain `idx` and `is_supporting`. The adapter preserves these identifiers
through retrieval and atomic-fact provenance mapping.
