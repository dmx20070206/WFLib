#!/bin/bash

# bash scripts/TemporalDrift/AWF.sh &
# PID_AWF=$!
# bash scripts/TemporalDrift/BAPM.sh &
# PID_BAPM=$!
# bash scripts/TemporalDrift/ARES.sh &
# PID_ARES=$!
bash scripts/TemporalDrift/DF.sh &
PID_DF=$!
# bash scripts/TemporalDrift/NetCLR.sh &
# PID_NetCLR=$!
bash scripts/TemporalDrift/Tik-Tok.sh &
PID_TikTok=$!
bash scripts/TemporalDrift/Var-CNN.sh &
PID_VarCNN=$!
bash scripts/TemporalDrift/RF.sh &
PID_RF=$!
# bash scripts/TemporalDrift/TF.sh &
# PID_TF=$!

echo "所有脚本已在后台启动，等待完成..."

# 等待每个进程并检查状态
declare -A PIDS
PIDS=(
    # ["AWF"]=$PID_AWF
    # ["BAPM"]=$PID_BAPM
    # ["ARES"]=$PID_ARES
    ["DF"]=$PID_DF
    # ["NetCLR"]=$PID_NetCLR
    ["Tik-Tok"]=$PID_TikTok
    ["Var-CNN"]=$PID_VarCNN
    ["RF"]=$PID_RF
    # ["TF"]=$PID_TF
)

for NAME in "${!PIDS[@]}"; do
    wait "${PIDS[$NAME]}"
    if [ $? -eq 0 ]; then
        echo "✅ $NAME 已完成"
    else
        echo "❌ $NAME 运行失败"
    fi
done

echo "所有脚本执行完毕"