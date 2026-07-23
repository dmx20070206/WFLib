set -e

python -u -m other_experiments.ablation \
    --dataset TemporalDrift \
    --model Proteus \
    --device cuda:0 \
    --train_file tam_train \
    --extra_train_file tam_bg_train \
    --tune_file tam_day270 \
    --extra_tune_file tam_bg_tune \
    --test_file tam_day270 \
    --extra_test_file tam_bg_tune \
    --feature TAM \
    --seq_len 1800 \
    --batch_size 128 \
    --adapt_epochs 100 \
    --pseudo_threshold 0.8 \
    --tau_pct 99.0 \
    --tau_ema 0.9 \
    --tune_unknown_ratio 2.0 \
    --split_refresh 2 \
    --load_name max_f1 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --result_file ablation \
    "$@"