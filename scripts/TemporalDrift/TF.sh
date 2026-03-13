dataset=TemporalDrift
model=TF

python -u exp/train.py \
  --dataset ${dataset} \
  --model ${model} \
  --device cuda:2 \
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
    python -u exp/test.py \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:2 \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 5000 \
    --batch_size 256 \
    --eval_method kNN \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file ${file_name}

    python -u exp/proteus_type2.py \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:2 \
    --train_file train \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 5000 \
    --batch_size 256 \
    --eval_method kNN \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name proteus \
    --model_save_name proteus \
    --result_file Proteus_${file_name} 
done