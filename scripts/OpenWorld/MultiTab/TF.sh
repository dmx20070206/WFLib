dataset=MultiTab
model=TF

python -u -m exp.train \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:2 \
    --train_file train \
    --extra_train_file bg_train \
    --valid_file valid \
    --extra_valid_file bg_valid \
    --feature DIR \
    --seq_len 5000 \
    --train_epochs 100 \
    --batch_size 512 \
    --learning_rate 1e-4 \
    --loss TripletMarginLoss \
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
        --device cuda:2 \
        --test_file ${file_name} \
        --extra_test_file bg_tune \
        --valid_file valid \
        --extra_valid_file bg_valid \
        --feature DIR \
        --seq_len 5000 \
        --batch_size 256 \
        --eval_method kNN \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name max_f1 \
        --result_file ${file_name}

    rm -rf checkpoints/${dataset}/${model}/proteus.pth
    cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
done