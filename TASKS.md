# Player Decision Improvements Tasks

## Heuristic-driven actions
- [ ] Add a `policy.score_action()` (or similar) interface that evaluates candidate actions using current state, engine metadata, and player context.
- [ ] Refactor `choose_and_take_action()` to build candidate actions and pick the highest-scoring option instead of fixed priority order.
- [ ] Extend heuristic features in `Policy` to include game progress signals (e.g., monsters captured, party class completion, action point efficiency).
- [ ] Add per-action feature extraction helpers (attack/activate/play/draw) with unit tests for expected score contributions.
- [ ] Update logging to capture chosen action scores for debugging and tuning.

## Reinforcement learning enhancements
- [ ] Add a lightweight RL training module that runs batched self-play games and collects `(state, action, reward, next_state)` transitions.
- [ ] Define a reward shaping strategy with clear constants (win/loss, monster capture, party class completion, wasted actions).
- [ ] Implement a learnable weight vector over heuristic features and a policy update loop (e.g., REINFORCE or Q-learning).
- [ ] Persist learned weights to a JSON (or similar) file and load them in `Policy` initialization.
- [ ] Add evaluation scripts to compare baseline heuristic policy vs. RL-tuned policy over multiple seeds.

## Documentation and maintenance
- [ ] Document the heuristic feature set and RL training workflow in a new `docs/` file.
- [ ] Add a usage example in `simulate.py` or a new CLI entrypoint to run training and evaluation.
