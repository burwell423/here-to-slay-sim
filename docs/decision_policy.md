# Player Decision Policy & RL Training

## Heuristic feature set
The policy scores every candidate action using a weighted feature vector. Features are split into base (state) features and per-action features:

### Base features (all actions)
- `bias`: constant 1.0.
- `action_cost`: action point cost of the action.
- `action_point_efficiency`: remaining action points after the action, normalized by actions per turn.
- `monsters_captured`: number of monsters already captured.
- `party_class_progress`: fraction of hero classes collected (unique classes / total required classes).
- `hand_size`: cards in hand.
- `party_size`: heroes in party.

### Attack features
- `is_attack`: 1.0 if the action is an attack.
- `monster_value`: card value of the target monster.
- `monster_capture_urgency`: normalized remaining captures needed for victory.

### Activate features
- `is_activate`: 1.0 if the action activates a hero.
- `activated_hero_value`: card value of the hero being activated.
- `remaining_activations`: count of heroes that can still be activated this turn.

### Play features
- `is_play`: 1.0 if the action plays a card.
- `played_card_value`: card value of the played card.
- `played_card_is_hero`: 1.0 for hero cards.
- `played_card_is_item`: 1.0 for item cards.
- `played_card_is_magic`: 1.0 for magic cards.
- `played_card_is_challenge`: 1.0 for challenge cards.
- `played_card_is_modifier`: 1.0 for modifier cards (should generally be avoided on the main action).
- `adds_party_class`: 1.0 if the played hero adds a new party class.

### Draw features
- `is_draw`: 1.0 for draw actions.
- `draw_pile_size`: number of cards remaining in the draw pile.

## RL training workflow
The RL module uses a simple linear Q-learning update over the same feature vector:

1. Build candidate actions each turn.
2. Choose an action using ε-greedy selection over the heuristic score.
3. Compute shaped reward after the action.
4. Update feature weights using the TD error:
   `w += α * (reward + γ * maxQ(next_state) - Q(state, action)) * feature_value`

### Reward shaping constants
- Win: `+10`
- Loss: `-8`
- Monster capture: `+2.5` per monster
- Party class completion: `+6`
- Party class progress: `+1.5` scaled by progress delta
- Wasted action: `-1`
- Card play value: `+0.02 * tuning_value`
- Monster attack value: `+0.02 * tuning_value`
- Hero activation value: `+0.01 * tuning_value`

## Usage

### Run a baseline game
```
python simulate.py run --seed 7 --turns 8
```

### Train weights
```
python simulate.py train --episodes 50 --output policy_weights.json
```

### Continue training from saved transitions
```
python simulate.py train --episodes 25 --output policy_weights.json \
  --transitions-in training_data.json --transitions-out training_data.json --replay-epochs 2
```

### Evaluate baseline vs tuned policy
```
python simulate.py evaluate --seeds 1 2 3 4 5 --weights policy_weights.json
```
