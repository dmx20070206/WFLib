#!/usr/bin/env bash
dataset="TemporalDrift"
model=DMX

# ============================================================
# 0. Preprocess (TAM features)
# ============================================================
for filename in train valid day14 day30 day90 day150 day270; do
    python -u exp/dataset_process/gen_tam.py \
      --dataset ${dataset} \
      --seq_len 5000 \
      --in_file ${filename}
done

# ============================================================
# 1. Train the base model on source domain (tam_train)
# ============================================================
python -u exp/train.py \
  --dataset ${dataset} \
  --model ${model} \
  --device cuda:2 \
  --train_file tam_train \
  --valid_file tam_valid \
  --feature TAM \
  --seq_len 1800 \
  --train_epochs 30 \
  --batch_size 200 \
  --learning_rate 5e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

wait

# ============================================================
# 2. Baseline A: Source-only  —— evaluate base model directly
#    on the target domain (no adaptation at all)
# ============================================================
# for file_name in day270
# do
#     python -u exp/test.py \
#       --dataset ${dataset} \
#       --model ${model} \
#       --device cuda:0 \
#       --test_file tam_${file_name} \
#       --feature TAM \
#       --seq_len 1800 \
#       --batch_size 256 \
#       --eval_metrics Accuracy Precision Recall F1-score \
#       --load_name max_f1 \
#       --result_file source_only_${file_name}
# done

# wait

# ============================================================
# 3. Baseline B: FT-only  —— fine-tune on target support set
#    Save model/log separately from the other baselines.
# ============================================================
# for file_name in day270
# do
#     python -u exp/ft_only.py \
#       --dataset ${dataset} \
#       --model ${model} \
#       --device cuda:0 \
#       --test_file tam_${file_name} \
#       --feature TAM \
#       --seq_len 1800 \
#       --batch_size 128 \
#       --shot 10 \
#       --support_seed 20070206 \
#       --load_name max_f1 \
#       --ft_epochs 50 \
#       --ft_lr 1e-4 \
#       --model_save_name ft_only_${file_name} \
#       --result_file ft_only_${file_name}
# done

# wait

# ============================================================
# 4. Oracle  —— train directly on full target data
#    (uses tam_day270 as both train and valid; reports on itself)
#    This is the "upper bound" — training on the *whole* target
#    domain as if it were fully labelled.
# ============================================================
# For Oracle we train a fresh model from scratch on day270.
# The exp/train.py script refuses to run if the .pth already exists,
# so we remove any stale checkpoint first.
# rm -rf checkpoints/${dataset}/${model}/oracle_day270.pth

# python -u exp/train.py \
#   --dataset ${dataset} \
#   --model ${model} \
#   --device cuda:1 \
#   --train_file tam_day270 \
#   --valid_file tam_day270 \
#   --feature TAM \
#   --seq_len 1800 \
#   --train_epochs 30 \
#   --batch_size 200 \
#   --learning_rate 5e-4 \
#   --optimizer Adam \
#   --eval_metrics Accuracy Precision Recall F1-score \
#   --save_metric F1-score \
#   --save_name oracle_day270

# wait

# Evaluate Oracle on day270
# python -u exp/test.py \
#   --dataset ${dataset} \
#   --model ${model} \
#   --device cuda:1 \
#   --test_file tam_day270 \
#   --feature TAM \
#   --seq_len 1800 \
#   --batch_size 256 \
#   --eval_metrics Accuracy Precision Recall F1-score \
#   --load_name oracle_day270 \
#   --result_file oracle_day270

# wait

# ============================================================
# 5. Proteus (your three-stage method)  — keep as reference
# ============================================================
rm -rf checkpoints/${dataset}/${model}/proteus.pth
cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
wait

for file_name in day270; do
    python -u exp/proteus.py \
      --dataset ${dataset} \
      --model ${model} \
      --device cuda:0 \
      --train_file tam_train \
      --test_file tam_${file_name} \
      --feature TAM \
      --seq_len 1800 \
      --batch_size 128 \
      --shot 10 \
      --load_name proteus \
      --model_save_name proteus_${file_name} \
      --result_file Proteus_${file_name} \
      --stage1_epochs 50 \
      --stage2_epochs 100 \
      --stage3_epochs 50 \
      --adapt_lr 1e-4 \
      --map_lr 1e-3 \
      --alpha 1.0 \
      --lambda_contrast 0.0

    rm -rf checkpoints/${dataset}/${model}/proteus.pth
    cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
done
