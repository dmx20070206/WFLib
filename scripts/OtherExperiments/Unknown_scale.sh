dataset=TemporalDrift
model=Proteus
checkpoints=./checkpoints/OtherExperiments/UnknownScale
log_path=./logs/OtherExperiments/UnknownScale

python -u -m exp.train \
    --open_set \
    --use_energy_loss \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:2 \
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
    --save_name max_f1 \
    --checkpoints ${checkpoints}

wait
rm -rf ${checkpoints}/${dataset}/${model}/proteus.pth
cp ${checkpoints}/${dataset}/${model}/max_f1.pth ${checkpoints}/${dataset}/${model}/proteus.pth
wait

for scale in 0 0.5 1.0 1.5 2.0 2.5 3.0
do
    python -u -m exp.proteus_os \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:4 \
        --train_file tam_train \
        --test_file tam_day270 \
        --tune_file tam_day270 \
        --extra_train_file tam_bg_train \
        --extra_tune_file tam_bg_tune \
        --extra_test_file tam_bg_tune \
        --feature TAM \
        --seq_len 1800 \
        --batch_size 128 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --log_path ${log_path} \
        --checkpoints ${checkpoints} \
        --tune_unknown_ratio ${scale} \
        --load_name proteus \
        --model_save_name proteus \
        --result_file Proteus_${scale} \

    rm -rf ${checkpoints}/${dataset}/${model}/proteus.pth
    cp ${checkpoints}/${dataset}/${model}/max_f1.pth ${checkpoints}/${dataset}/${model}/proteus.pth
done