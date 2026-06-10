dataset=BehaviorDrift
model=Proteus

for filename in bg_train bg_valid bg_tune 
do
    python -u -m exp.dataset_process.gen_tam \
        --dataset "" \
        --seq_len 5000 \
        --in_file ${filename}
done


for file_name in train valid
do 
    python -u -m exp.dataset_process.gen_tam \
      --dataset ${dataset} \
      --seq_len 5000 \
      --in_file ${file_name}
done

python -u -m exp.train \
    --open_set \
    --use_energy_loss \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:7 \
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

for file_name in subpage test
do
    rm -rf checkpoints/${dataset}/${model}/proteus.pth
    cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
    wait

    python -u -m exp.dataset_process.gen_tam \
      --dataset ${dataset} \
      --seq_len 5000 \
      --in_file ${file_name}
    
    wait

    python -u -m exp.proteus_os \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:7 \
        --train_file tam_train \
        --test_file tam_${file_name} \
        --tune_file tam_${file_name} \
        --extra_train_file tam_bg_train \
        --extra_tune_file tam_bg_tune \
        --extra_test_file tam_bg_tune \
        --feature TAM \
        --seq_len 1800 \
        --batch_size 128 \
        --pseudo_threshold 0.85 \
        --tau_pct 99.0 \
        --tau_ema 0.9 \
        --energy_m_in -16.0 \
        --energy_m_out -2.0 \
        --energy_loss_weight 0.01 \
        --split_refresh 2 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name proteus \
        --model_save_name proteus \
        --result_file Proteus_${file_name} 

    rm -rf checkpoints/${dataset}/${model}/proteus.pth
    cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
done