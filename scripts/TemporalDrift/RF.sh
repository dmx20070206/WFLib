dataset=TemporalDrift
model=RF

for filename in train valid test day14 day30 day90 day150 day270
do 
    python -u exp/dataset_process/gen_tam.py \
      --dataset ${dataset} \
      --seq_len 5000 \
      --in_file ${filename}
done

python -u exp/train.py \
  --dataset ${dataset} \
  --model ${model} \
  --device cuda:2 \
  --train_file tam_train \
  --valid_file tam_valid \
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

for file_name in day270
do
    python -u exp/test.py \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:2 \
    --test_file tam_${file_name} \
    --feature TAM \
    --seq_len 1800 \
    --batch_size 256 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file ${file_name}

    python -u exp/proteus.py \
      --dataset ${dataset} \
      --model ${model} \
      --device cuda:2 \
      --train_file tam_train \
      --test_file tam_${file_name} \
      --feature TAM \
      --seq_len 1800 \
      --batch_size 128 \
      --eval_metrics Accuracy Precision Recall F1-score \
      --load_name proteus \
      --model_save_name proteus \
      --result_file Proteus_${file_name} 

done