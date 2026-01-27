#!/usr/bin/env python3
"""
Here to Slay - MVP simulator

Key features implemented:
- Separate draw deck / monster deck / party leader deck
- Setup: each player gets random party leader + 3 cards, reveal 3 monsters
- Turn structure: 3 actions; Draw (1), Play (1), Activate hero (1), Attack monster (2)
- Challenge cards: can challenge plays of hero/item/magic (not modifiers), cancelled on win
- Modifier cards: may be played by any player during any roll event; multi-option deltas supported
- Monsters:
  - on_attacked row defines success_condition / fail_condition + (optional) requirements
  - success_action / fail_action rows resolve via EffectSteps
  - captured monster passives via triggers like on_draw / on_challenge
"""

from hts_sim.game import run_game


if __name__ == "__main__":
    for line in run_game(seed=7, turns=8):
        print(line)
