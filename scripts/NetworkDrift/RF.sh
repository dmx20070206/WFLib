dataset=NetworkDrift
model=RF

for file_name in train valid
do 
    python -u exp/dataset_process/gen_tam.py \
      --dataset ${dataset} \
      --seq_len 5000 \
      --in_file ${file_name}
done

python -u exp/train.py \
  --dataset ${dataset} \
  --model ${model} \
  --device cuda:6 \
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

for file_name in SG USA UK JP DE
do
    rm -rf checkpoints/${dataset}/${model}/proteus.pth
    cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
    wait

    python -u exp/dataset_process/gen_tam.py \
      --dataset ${dataset} \
      --seq_len 5000 \
      --in_file ${file_name}
    
    wait

    python -u exp/test.py \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:6 \
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
      --device cuda:6 \
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