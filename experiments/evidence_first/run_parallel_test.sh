#!/bin/bash
# 并行运行100题测试，分成4个批次

DATA="comagraag/data/hotpotqa_val.json"
KG="comagraag/data/hotpotqa_kgs.pkl"
OUT_DIR="results"
TAG="test_improvements_100_parallel"

# 每个批次25题
for i in 0 25 50 75; do
    python3 comagraag/evaluate.py \
        --data $DATA \
        --kg $KG \
        --mode full \
        --dataset hotpotqa \
        --out-dir $OUT_DIR \
        --start $i \
        --n 25 \
        --cache-tag ${TAG}_batch${i} \
        > /tmp/${TAG}_batch${i}.log 2>&1 &

    echo "启动批次 $i (PID: $!)"
done

echo ""
echo "4个批次已启动，每批25题"
echo "查看进度: tail -f /tmp/${TAG}_batch*.log"
echo "等待所有批次完成..."
wait
echo "所有批次完成！"
