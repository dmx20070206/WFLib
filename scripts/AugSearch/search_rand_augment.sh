#!/usr/bin/env bash
set -euo pipefail

drift_list=('NoDrift' 'TemporalDrift' 'BehaviorDrift' 'NetworkDrift' 'VersionDrift')
declare -A drift_dict=(
    ['NoDrift']="test"
    ['TemporalDrift']="day270"
    ['BehaviorDrift']="subpage"
    ['NetworkDrift']="USA"
    ['VersionDrift']="drift"
)

model=TikTok
summary_file="logs/AugSearch/search_summary.csv"

for dataset in "${drift_list[@]}"
do
    for n in {1..2}
    do
        for m in {1..2}
        do
            echo "dataset: ${dataset}, n: ${n}, m: ${m}"

            # data augmentation
            rm -f "datasets/${dataset}/train_aug.npz"
            python key_observation/wf_rand_augment.py \
                --src "datasets/${dataset}/train.npz" \
                --dst "datasets/${dataset}/train_aug.npz" \
                --n "${n}" \
                --m "${m}"

            run_tag="n${n}_m${m}"

            # model training
            python -u exp/train.py \
              --dataset "${dataset}" \
              --model "${model}" \
              --device cuda:1 \
              --feature DIR \
              --seq_len 5000 \
              --train_epochs 5 \
              --train_file train_aug \
              --batch_size 128 \
              --learning_rate 2e-3 \
              --optimizer Adamax \
              --eval_metrics Accuracy Precision Recall F1-score \
              --save_metric F1-score \
              --save_name "${run_tag}"

            # model testing
            file="${drift_dict[$dataset]}"
            result_name="${run_tag}_${file}"
            python -u exp/test.py \
                --dataset "${dataset}" \
                --model "${model}" \
                --device cuda:1 \
                --test_file "${file}" \
                --feature DIR \
                --seq_len 5000 \
                --batch_size 256 \
                --eval_metrics Accuracy Precision Recall F1-score \
                --load_name "${run_tag}" \
                --result_file "${result_name}"

            # collect and persist one-line summary for search ranking
            python key_observation/record_eval_result.py \
                --result-json "logs/${dataset}/${model}/${result_name}.json" \
                --summary-csv "${summary_file}" \
                --dataset "${dataset}" \
                --model "${model}" \
                --n "${n}" \
                --m "${m}" \
                --test-file "${file}"
        done
    done
done
