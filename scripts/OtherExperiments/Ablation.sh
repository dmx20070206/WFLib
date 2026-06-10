set -e

dataset=TemporalDrift
model=Proteus
device=cuda:0
feature=TAM
seq_len=1800
train_file=tam_train
extra_train_file=tam_bg_train
tune_file=tam_day270
extra_tune_file=tam_bg_tune
test_file=tam_day270
extra_test_file=tam_bg_tune
checkpoint_name=max_f1

checkpoint_file=checkpoints/${dataset}/${model}/${checkpoint_name}.pth
[[ -f ${checkpoint_file} ]] || { echo "Missing file: ${checkpoint_file}"; exit 1; }

python -u -m other_experiments.ablation \
    --dataset ${dataset} \
    --model ${model} \
    --device ${device} \
    --train_file ${train_file} \
    --extra_train_file ${extra_train_file} \
    --tune_file ${tune_file} \
    --extra_tune_file ${extra_tune_file} \
    --test_file ${test_file} \
    --extra_test_file ${extra_test_file} \
    --feature ${feature} \
    --seq_len ${seq_len} \
    --batch_size 128 \
    --adapt_epochs 100 \
    --pseudo_threshold 0.8 \
    --tau_pct 99.0 \
    --tau_ema 0.9 \
    --tune_unknown_ratio 2.0 \
    --split_refresh 2 \
    --load_name ${checkpoint_name} \
    --eval_metrics Accuracy Precision Recall F1-score \
    --result_file ablation \
    "$@"