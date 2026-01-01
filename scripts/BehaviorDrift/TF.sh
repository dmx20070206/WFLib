dataset=BehaviorDrift
model=TF

python -u exp/train.py \
  --dataset ${dataset} \
  --model ${model} \
  --device cuda:7 \
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

for file_name in test subpage
do
    python -u exp/test.py \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:7 \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 5000 \
    --batch_size 256 \
    --eval_method kNN \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file ${file_name}
done