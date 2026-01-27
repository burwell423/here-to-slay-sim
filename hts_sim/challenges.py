from typing import List

from .conditions import is_challengeable_card_type
from .effects import resolve_effect
from .models import Engine, GameState, Policy
from .rolls import resolve_roll_event


def maybe_challenge_play(
    state: GameState,
    engine: Engine,
    pid_playing: int,
    played_card_id: int,
    rng: "random.Random",
    policy: Policy,
    log: List[str],
) -> bool:
    """
    Returns True if play is cancelled by a successful challenge.
    """
    played_type = str(engine.card_meta.get(played_card_id, {}).get("type", "unknown")).lower()
    if not is_challengeable_card_type(played_type):
        return False

    ctx = {
        "challenge_target": engine.card_meta.get(played_card_id, {"id": played_card_id, "type": "unknown"}),
        "challenge.denied": False,
    }

    for pstate in state.players:
        for mid in pstate.captured_monsters:
            for step in engine.monster_effects.get(mid, []):
                if "on_challenge" in step.triggers():
                    resolve_effect(step, state, engine, pstate.pid, ctx, rng, policy, log)

    if ctx.get("challenge.denied"):
        log.append("[Challenge] DENIED by effect")
        return False

    challenge_pick = policy.choose_challenger(state, engine, pid_playing)
    if challenge_pick is None:
        return False

    challenger_pid, challenge_card_id = challenge_pick

    if not policy.should_challenge(rng):
        return False

    challenger = state.players[challenger_pid]
    challenger.hand.remove(challenge_card_id)
    state.discard_pile.append(challenge_card_id)

    log.append(
        f"[P{challenger_pid}] CHALLENGE played {challenge_card_id} "
        f"({engine.card_meta.get(challenge_card_id,{}).get('name','?')}) "
        f"to challenge {played_card_id} ({engine.card_meta.get(played_card_id,{}).get('name','?')}) "
        f"by P{pid_playing}"
    )

    r_challenger = resolve_roll_event(
        state=state,
        engine=engine,
        roller_pid=challenger_pid,
        roll_reason="challenge:challenger",
        rng=rng,
        log=log,
        mode="maximize",
    )
    r_playing = resolve_roll_event(
        state=state,
        engine=engine,
        roller_pid=pid_playing,
        roll_reason="challenge:played",
        rng=rng,
        log=log,
        mode="maximize",
    )

    log.append(f"[Challenge] P{challenger_pid} rolls {r_challenger} vs P{pid_playing} rolls {r_playing}")

    if r_challenger > r_playing:
        log.append("[Challenge] SUCCESS: play cancelled")
        return True

    log.append("[Challenge] FAIL: play continues")
    return False
