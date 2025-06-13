# Block Generator and Double Training

## Branch Structure

- `plots/`：training and evaluating results
- `src/`: source code
    - `train_gen.py`: trainer for block generator
    - `evaluate_gen.py`: evaluator for block generator
    - `train.py`: double trainer, will modify both dqn weights and gen weights
    - `evaluate.py`: load dqn weights, evaluate it on a random circumstance
- `dqn_weights.pkl`: weight of the tetris single agent
- `gen_weights.pkl`: weight of the block generator