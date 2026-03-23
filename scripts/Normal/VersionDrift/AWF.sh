for dataset in VersionDrift/045 VersionDrift/046 VersionDrift/047 VersionDrift/048
do
model=AWF

python -u exp/train.py \
  --dataset ${dataset} \
  --model ${model} \
  --device cuda:1 \
  --feature DIR \
  --seq_len 3000 \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 8e-4 \
  --optimizer RMSprop \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

wait
rm -rf checkpoints/${dataset}/${model}/proteus.pth
cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
wait

for file_name in drift
do
    python -u exp/test.py \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:1 \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 3000 \
        --batch_size 256 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name max_f1 \
        --result_file ${file_name}

    python -u exp/proteus.py \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:1 \
        --train_file train \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 3000 \
        --batch_size 128 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name proteus \
        --model_save_name proteus \
        --result_file Proteus_${file_name} 
done
done