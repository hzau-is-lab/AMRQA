from amrqa.config import AMRQAConfig
from amrqa.prompt_assets import load_atomic_fact_demonstrations, load_react_demonstrations
from amrqa.prompts import atomic_fact_prompt, candidate_prompt, semantic_score_prompt


def test_original_react_demonstrations_are_packaged() -> None:
    demonstrations = load_react_demonstrations()
    assert demonstrations.count("Question:") == 8
    assert "Finish[Hifikepunye Pohamba]" in demonstrations


def test_candidate_prompt_can_enable_or_disable_demonstrations() -> None:
    demonstrations = load_react_demonstrations()
    with_examples = candidate_prompt("Question?", "History", 3, demonstrations)
    without_examples = candidate_prompt("Question?", "History", 3)
    assert "Who succeeded the first President of Namibia?" in with_examples
    assert "Who succeeded the first President of Namibia?" not in without_examples
    assert AMRQAConfig().prompts.use_react_demonstrations is True


def test_original_atomic_fact_demonstrations_are_packaged() -> None:
    demonstrations = load_atomic_fact_demonstrations()
    prompt = atomic_fact_prompt("A new paragraph.", demonstrations)

    assert demonstrations.count("paragraph:") == 4
    assert "Maheen Khan is a Pakistani fashion designer." in prompt
    assert "paragraph: A new paragraph." in prompt
    assert AMRQAConfig().prompts.use_atomic_fact_demonstrations is True


def test_semantic_critic_is_global_not_history_aware() -> None:
    prompt = semantic_score_prompt("Original question?", "Candidate question?")
    assert "global semantic relevance" in prompt
    assert "no reasoning history is supplied" in prompt
    assert "necessary and logically valid next step" not in prompt
