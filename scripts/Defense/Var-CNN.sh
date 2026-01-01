for dataset in Defense
do
model=VarCNN

python -u exp/train.py \
  --dataset ${dataset} \
  --model ${model} \
  --device cuda:4 \
  --feature DT2 \
  --seq_len 5000 \
  --train_epochs 30 \
  --batch_size 50 \
  --learning_rate 1e-3 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

wait
rm -rf checkpoints/${dataset}/${model}/proteus.pth
cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
wait

for file_name in test day14 day30 day90 day150 day270
do
    python -u exp/test.py \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:4 \
    --test_file ${file_name} \
    --feature DT2 \
    --seq_len 5000 \
    --batch_size 256 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file ${file_name}

    python -u exp/proteus.py \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:4 \
        --train_file train \
        --test_file ${file_name} \
        --feature DT2 \
        --seq_len 5000 \
        --batch_size 128 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name proteus \
        --model_save_name proteus \
        --result_file Proteus_${file_name} 
done
done