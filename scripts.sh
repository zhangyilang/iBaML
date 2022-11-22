#!/bin/bash

# miniImageNet
# K = 5, L = 5
nohup python main.py --algorithm iBaML --batch-size 2 --num-way 5 --num-supp 5 --cg-iter 5 \
  --task-iter 5 --task-lr 0.01 > log/miniImageNet/iBaML-5way5shot-5iter.log 2>&1 &
# K = 5, L = 10
nohup python main.py --algorithm iBaML --batch-size 2 --num-way 5 --num-supp 5 --cg-iter 5 \
  --task-iter 10 --task-lr 0.01 > log/miniImageNet/iBaML-5way5shot-5iter.log 2>&1 &
# K = 5, L = 15
nohup python main.py --algorithm iBaML --batch-size 2 --num-way 5 --num-supp 5 --cg-iter 5 \
  --task-iter 15 --task-lr 0.01 > log/miniImageNet/iBaML-5way5shot-5iter.log 2>&1 &

# Omniglot
# K = 5, L = 5
nohup python main.py --algorithm iBaML --batch-size 16 --num-way 5 --num-supp 1 --cg-iter 5 \
  --task-iter 5 --task-lr 0.1 > log/Omniglot/iBaML-5way1shot-5iter.log 2>&1 &
nohup python main.py --algorithm iBaML --batch-size 16 --num-way 5 --num-supp 5 --cg-iter 5 \
  --task-iter 5 --task-lr 0.1 > log/Omniglot/iBaML-5way5shot-5iter.log 2>&1 &
nohup python main.py --algorithm iBaML --batch-size 16 --num-way 20 --num-supp 1 --cg-iter 5 \
  --task-iter 5 --task-lr 0.1 > log/Omniglot/iBaML-20way1shot-5iter.log 2>&1 &
nohup python main.py --algorithm iBaML --batch-size 16 --num-way 20 --num-supp 5 --cg-iter 5 \
  --task-iter 5 --task-lr 0.1 > log/Omniglot/iBaML-20way5shot-5iter.log 2>&1 &
# ...
