#!/usr/bin/env bash
dataset=TemporalDrift
model=DMX
device=cuda:7
shot=5

python -u exp/train.py \
    --dataset "${dataset}" \
    --model "${model}" \
    --device "${device}" \
    --train_file tam_train \
    --valid_file tam_valid \
    --feature TAM \
    --seq_len 1800 \
    --train_epochs 40 \
    --batch_size 200 \
    --loss CrossEntropyLoss \
    --weights 1.0 \
    --learning_rate 5e-4 \
    --optimizer Adam \
    --eval_metrics Accuracy Precision Recall F1-score \
    --save_metric F1-score \
    --save_name max_f1

for file_name in day270
do
    python -u exp/proteus.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --device "${device}" \
        --train_file tam_train \
        --test_file "tam_${file_name}" \
        --feature TAM \
        --seq_len 1800 \
        --batch_size 128 \
        --shot "${shot}" \
        --load_name max_f1 \
        --model_save_name "proteus_${file_name}_shot${shot}" \
        --result_file "Proteus_${file_name}_shot${shot}" \
        --stage2_epochs 10 \
        --stage3_epochs 50 \
        --map_lr 1e-3 \
        --ot_eta 0.1 \
        --ot_epsilon 0.05 \
        --ot_iterations 30 \
        --ot_max_descriptors 0 \
        --map_reg_weight 0.01 \
        --adapt_lr 1e-4 \
        --load_stage2 \
        --augmented_loss_weight 0.8 \
        --reliability_strength 1.0
done
