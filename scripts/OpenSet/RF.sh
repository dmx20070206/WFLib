dataset=OpenSet
model=RF

for filename in train valid day14 day30 day90 day150 day270
do 
    python -u exp/dataset_process/gen_tam.py \
      --dataset ${dataset} \
      --seq_len 5000 \
      --in_file ${filename}
done

python -u exp/train.py \
    --open_set \
    --use_energy_loss \
    --unknown_label 102 \
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
    --save_name max_f1 \
    --energy_m_in -18 \
    --energy_m_out -2 \
    --energy_weight 0.1

wait
rm -rf checkpoints/${dataset}/${model}/proteus.pth
cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth
wait

for file_name in day270
do
    python -u exp/test.py \
        --open_set \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:1 \
        --test_file tam_${file_name} \
        --feature TAM \
        --seq_len 1800 \
        --batch_size 256 \
        --eval_metrics F1-score Closed-F1 Open-AUROC \
        --load_name max_f1 \
        --result_file ${file_name}

    python -u exp/proteus_os1.py \
        --unknown_label 102 \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:3 \
        --train_file tam_train \
        --test_file tam_${file_name} \
        --feature TAM \
        --seq_len 1800 \
        --batch_size 128 \
        --eval_metrics F1-score Closed-F1 Open-AUROC \
        --load_name proteus \
        --model_save_name proteus \
        --result_file Proteus_${file_name} 

        rm -rf checkpoints/${dataset}/${model}/proteus.pth
        cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/proteus.pth

done