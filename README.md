# Single Player Tetris DQN Agent

## Install
```bash
conda create --name cs181 python=3.9
conda activate cs181
pip install -r requirements.txt
```

## Play it !
```bash
python src/play.py
```
`Up` 旋转，`Left/Right`移动，`Space`下落

## Train
```bash
python src/train.py --episodes 2000
```
如果删除目录下的`*.pkl`文件会从头训练，否则是在之前的基础上训练。`plots`目录下的`*.png`可以用来观察训练过程。
更多命令行参数见`train.py`

## More Training
```bash
python src/train.py --episodes 5000 --epsilon 0.01 --epsilon_min 0.001  --target_update 1000 --lr 1e-5
```
从`dqn_weights.pkl`的基础上继续训练，减小epsilon的始末和学习率

## Evaluate
```bash
python src/evaluate.py --episodes 30 --render
```
会返回`episodes`次测试的平均得分，以及最优、最劣5次的消行总数。

## Reference

> caohch-1: [Trtris-AI](https://github.com/caohch-1/Tetris-AI)