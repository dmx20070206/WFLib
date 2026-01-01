pretrian_dataset=OpenWorld
dataset=OpenWorld
model=NetCLR

python -u exp/pretrain.py \
  --dataset ${pretrian_dataset} \
  --model ${model} \
  --device cuda:0 \
  --train_epochs 100 \
  --train_file train \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --optimizer Adam \
  --save_name pretrain

python -u exp/train.py \
  --dataset ${dataset} \
  --model ${model} \
  --device cuda:0 \
  --feature DIR \
  --seq_len 5000 \
  --train_file train \
  --valid_file valid \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --load_file checkpoints/${pretrian_dataset}/NetCLR/pretrain.pth \
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
      --device cuda:0 \
      --test_file ${file_name} \
      --feature DIR \
      --seq_len 5000 \
      --batch_size 256 \
      --eval_metrics Accuracy Precision Recall F1-score \
      --load_name max_f1 \
      --result_file ${file_name}

    python -u exp/proteus.py \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:0 \
        --train_file train \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 5000 \
        --batch_size 128 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name proteus \
        --model_save_name proteus \
        --result_file Proteus_${file_name} 
done