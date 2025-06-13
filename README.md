# Double-Tetris: Double Gaming and Double Agent Training
ShanghaiTech U CS181 25Spring Artifitial Intelligence Final Project

## Install
```bash
conda create --name cs181 python=3.9
conda activate cs181
pip install -r requirements.txt
```

## Branches
We store the code for each part under a different branch:
- `single-player`: Vanilla tetris agent, implemented by ourselves
- `double-agent`: Training and evaluating an intelligence that can control two separate Tetris games simultaneously
- `counter-generator`: Block generator agent, can be trained with frozen single-agent, or train two agents against each other simultaneously.