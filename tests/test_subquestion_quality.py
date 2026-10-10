from amrqa.subquestion_quality import rule_based_well_formed


def test_released_question_form_rule() -> None:
    assert rule_based_well_formed("Who directed the film?") == 1
    assert rule_based_well_formed("director of the film") == 0
    assert rule_based_well_formed("Who directed the film") == 0
