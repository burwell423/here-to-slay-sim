import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hts_sim.models import ActionCandidate, Engine, GameState, PlayerState, Policy


def _engine():
    return Engine(
        effects_by_card={},
        card_meta={
            1: {"id": 1, "type": "hero", "subtype": "wizard", "action_cost": 1, "name": "Hero One"},
            2: {"id": 2, "type": "monster", "subtype": "", "action_cost": 2, "name": "Monster A"},
            3: {"id": 3, "type": "magic", "subtype": "", "action_cost": 1, "name": "Magic"},
        },
        monster_attack_rules={},
        monster_effects={},
        modifier_options_by_card_id={},
    )


def test_play_features_add_party_class_score():
    engine = _engine()
    player = PlayerState(pid=0, hand=[1], party=[])
    state = GameState(players=[player], draw_pile=[])
    action = ActionCandidate(kind="play_card", cost=1, card_id=1)

    policy = Policy(feature_weights={"is_play": 1.0, "adds_party_class": 2.0, "played_card_value": 0.0})

    features = policy.extract_play_features(action, state, engine, pid=0)
    assert features["played_card_is_hero"] == 1.0
    assert features["adds_party_class"] == 1.0
    assert policy.score_action(action, state, engine, pid=0) == 3.0


def test_attack_features_include_urgency_and_value():
    engine = _engine()
    player = PlayerState(pid=0, hand=[], party=[], captured_monsters=[99])
    state = GameState(players=[player], draw_pile=[], monster_row=[2])
    action = ActionCandidate(kind="attack_monster", cost=2, monster_id=2)

    policy = Policy(feature_weights={"is_attack": 1.0, "monster_capture_urgency": 3.0, "monster_value": 0.0})

    features = policy.extract_attack_features(action, state, engine, pid=0)
    assert features["is_attack"] == 1.0
    assert features["monster_capture_urgency"] == pytest.approx(2 / 3)
    assert policy.score_action(action, state, engine, pid=0) == pytest.approx(1.0 + 3.0 * (2 / 3))
