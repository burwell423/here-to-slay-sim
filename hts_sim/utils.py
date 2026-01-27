from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import PlayerState


def find_challenge_card_in_hand(player: "PlayerState", card_meta: Dict[int, Dict[str, str]]) -> Optional[int]:
    for cid in player.hand:
        if str(card_meta.get(cid, {}).get("type", "")).strip().lower() == "challenge":
            return cid
    return None


def find_modifier_cards(player: "PlayerState", card_meta: Dict[int, Dict[str, str]]) -> List[int]:
    return [cid for cid in player.hand if str(card_meta.get(cid, {}).get("type", "")).lower() == "modifier"]


def format_card_list(card_ids: List[int], card_meta: Dict[int, Dict[str, str]]) -> str:
    if not card_ids:
        return "—"
    return ", ".join(f"{cid}:{card_meta.get(cid, {}).get('name','?')}" for cid in card_ids)
