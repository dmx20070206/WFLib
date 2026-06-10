source_dataset=TemporalDrift
target_root=MultiTab
overlap_dirs="overlap_20 overlap_40 overlap_60 overlap_80"
test_files="day14 day30 day90 day150 day270"

model=Proteus
device=cuda:2

for filename in bg_train bg_valid bg_tune
do
    python -u -m exp.dataset_process.gen_tam \
        --dataset "" \
        --seq_len 5000 \
        --in_file ${filename}
done

for filename in train valid day14 day30 day90 day150 day270
do
    python -u -m exp.dataset_process.gen_tam \
        --dataset ${source_dataset} \
        --seq_len 5000 \
        --in_file ${filename}
done

python -u -m exp.train \
    --open_set \
    --use_energy_loss \
    --dataset ${source_dataset} \
    --model ${model} \
    --device ${device} \
    --train_file tam_train \
    --extra_train_file tam_bg_train \
    --valid_file tam_valid \
    --extra_valid_file tam_bg_valid \
    --feature TAM \
    --seq_len 1800 \
    --train_epochs 60 \
    --batch_size 200 \
    --learning_rate 5e-4 \
    --optimizer Adam \
    --eval_metrics Accuracy Precision Recall F1-score \
    --save_metric F1-score \
    --save_name max_f1

for overlap_dir in ${overlap_dirs}
do
    target_dataset=${target_root}/${overlap_dir}

    for file_name in ${test_files}
    do
        python -u -m exp.dataset_process.gen_tam \
            --dataset ${target_dataset} \
            --seq_len 5000 \
            --in_file ${file_name}

        mkdir -p checkpoints/${target_dataset}/${model}
        cp checkpoints/${source_dataset}/${model}/max_f1.pth checkpoints/${target_dataset}/${model}/max_f1.pth
        cp checkpoints/${source_dataset}/${model}/max_f1.pth checkpoints/${target_dataset}/${model}/proteus.pth

        python -u -m exp.proteus_os \
            --dataset ${target_dataset} \
            --model ${model} \
            --device ${device} \
            --train_file ../../${source_dataset}/tam_train \
            --test_file tam_${file_name} \
            --tune_file tam_${file_name} \
            --extra_train_file tam_bg_train \
            --feature TAM \
            --seq_len 1800 \
            --batch_size 128 \
            --eval_metrics Accuracy Precision Recall F1-score \
            --load_name proteus \
            --model_save_name proteus \
            --result_file Proteus_${file_name} \
            --pseudo_threshold 0.7 \
            --tau_pct 99.0 \
            --tau_ema 0.9 \
            --energy_m_in -16.0 \
            --energy_m_out -2.0 \
            --energy_loss_weight 0.1 \
            --tune_unknown_ratio 2.0 \
            --split_refresh 2
    done
done