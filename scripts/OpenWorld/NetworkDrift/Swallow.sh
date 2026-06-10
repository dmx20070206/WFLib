dataset=NetworkDrift
model=Swallow
device=${SWALLOW_DEVICE:-cuda:2}
pretrain_epochs=${SWALLOW_PRETRAIN_EPOCHS:-40}
train_epochs=${SWALLOW_TRAIN_EPOCHS:-30}
pretrain_batch=${SWALLOW_PRETRAIN_BATCH_SIZE:-256}
train_batch=${SWALLOW_TRAIN_BATCH_SIZE:-64}
test_batch=${SWALLOW_TEST_BATCH_SIZE:-64}
workers=${SWALLOW_NUM_WORKERS:-8}

python -u -m exp.pretrain \
    --dataset ${dataset} \
    --model ${model} \
    --device ${device} \
    --train_file train \
    --extra_train_file bg_train \
    --train_epochs ${pretrain_epochs} \
    --batch_size ${pretrain_batch} \
    --learning_rate 3e-2 \
    --optimizer SGD \
    --save_name pretrain \
    --num_workers ${workers}

python -u -m exp.train \
    --dataset ${dataset} \
    --model ${model} \
    --device ${device} \
    --train_file train \
    --extra_train_file bg_train \
    --valid_file valid \
    --extra_valid_file bg_valid \
    --feature CIF \
    --seq_len 1000 \
    --train_epochs ${train_epochs} \
    --batch_size ${train_batch} \
    --learning_rate 1e-4 \
    --optimizer Adam \
    --eval_metrics Accuracy Precision Recall F1-score \
    --save_metric F1-score \
    --load_file checkpoints/${dataset}/${model}/pretrain.pth \
    --save_name max_f1

wait
rm -rf checkpoints/${dataset}/${model}/proteus.pth
cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
wait

for file_name in USA UK JP DE SG
do
    python -u -m exp.test \
        --dataset ${dataset} \
        --model ${model} \
        --device ${device} \
        --valid_file tam_valid \
        --extra_valid_file tam_bg_valid \
        --test_file ${file_name} \
        --extra_test_file bg_tune \
        --feature CIF \
        --seq_len 1000 \
        --batch_size ${test_batch} \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name max_f1 \
        --result_file ${file_name}

    rm -rf checkpoints/${dataset}/${model}/proteus.pth
    cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
done