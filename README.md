# Double-Tetris
ShanghaiTech U CS181 25Spring Artifitial Intelligence Final Project

## Install
```bash
conda create --name cs181 python=3.9
conda activate cs181
pip install -r requirements.txt
```

## Train
```bash
python src/train.py --agent qlearn --episodes 2000
```
最后的数字是轮数。如果删除目录下的`*.pkl`文件会从头训练，否则是在之前的基础上训练。目录下的`train_qlean.png`可以用来观察训练过程。

## Evaluate
```bash
python src/evaluate.py --agent qlearn --render
```