from noharm_piil.decisions import noharm_select, rank_accepted_candidates


def test_noharm_accepts_candidate_with_smaller_radius():
    d = noharm_select(0.5, 1.0, candidate="learned", baseline="baseline")
    assert d.accepted is True
    assert d.safe_choice == "learned"


def test_noharm_falls_back_when_radius_is_larger():
    d = noharm_select(1.5, 1.0, candidate="learned", baseline="baseline")
    assert d.accepted is False
    assert d.safe_choice == "baseline"


def test_rank_accepted_candidates_selects_smallest_accepted_radius():
    safe, decisions = rank_accepted_candidates({"baseline": 1.0, "a": 0.7, "b": 0.4, "c": 1.5})
    assert safe == "b"
    assert len(decisions) == 3
