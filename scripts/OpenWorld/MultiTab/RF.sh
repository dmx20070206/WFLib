dataset=MultiTab
model=RF

for filename in train valid day14 day30 day90 day150 day270
do 
    python -u -m exp.dataset_process.gen_tam \
        --dataset ${dataset} \
        --seq_len 5000 \
        --in_file ${filename}
done

python -u -m exp.train \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:5 \
    --train_file tam_train \
    --extra_train_file tam_bg_train \
    --valid_file tam_valid \
    --extra_valid_file tam_bg_valid \
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
rm -rf checkpoints/${dataset}/${model}/proteus.pth
cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
wait

for file_name in day14 day30 day90 day150 day270
do
    python -u -m exp.test \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:5 \
        --test_file tam_${file_name} \
        --extra_test_file tam_bg_tune \
        --valid_file tam_valid \
        --extra_valid_file tam_bg_valid \
        --feature TAM \
        --seq_len 1800 \
        --batch_size 256 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name max_f1 \
        --result_file ${file_name}

    rm -rf checkpoints/${dataset}/${model}/proteus.pth
    cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
done