import json

from amrqa.datasets import iter_jsonl
from amrqa.metrics import exact_match, summarize, token_f1
from amrqa.preprocessing import prepare_jsonl


def test_normalizes_standard_context_schema(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps(
            {
                "_id": "item-1",
                "question": "Who wrote the book?",
                "answer": "Ada Lovelace",
                "context": [["Book", ["The book was written by Ada Lovelace."]]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    example = next(iter_jsonl(path))
    assert example.id == "item-1"
    assert example.documents[0].title == "Book"
    assert example.documents[0].text == "The book was written by Ada Lovelace."
    assert example.documents[0].native_units[0].id == '["Book",0]'


def test_normalizes_musique_paragraph_text_schema(tmp_path) -> None:
    path = tmp_path / "musique.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "musique-1",
                "question": "Who wrote the book?",
                "answer": "Ada Lovelace",
                "paragraphs": [{"title": "Book", "paragraph_text": "Ada Lovelace wrote the book."}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    example = next(iter_jsonl(path))
    assert example.documents[0].text == "Ada Lovelace wrote the book."


def test_qa_metrics_match_normalized_answers() -> None:
    assert exact_match("The Ada Lovelace!", "Ada Lovelace") == 1.0
    assert token_f1("Ada", "Ada Lovelace") == 2 / 3
    assert summarize([("Ada", "Ada Lovelace")])["count"] == 1


def test_prepare_converts_json_array_without_dropping_source_fields(tmp_path) -> None:
    source = tmp_path / "dataset.json"
    source.write_text(
        json.dumps(
            [
                {
                    "_id": "item-1",
                    "question": "Who wrote the book?",
                    "answer": "Ada Lovelace",
                    "context": [["Book", ["Ada Lovelace wrote it."]]],
                    "supporting_facts": [["Book", 0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "dataset.jsonl"
    result = prepare_jsonl(source, output)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert result["count"] == 1
    assert len(result["sha256"]) == 64
    assert record["supporting_facts"] == [["Book", 0]]
