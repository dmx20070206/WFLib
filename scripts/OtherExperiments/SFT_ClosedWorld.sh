dataset=TemporalDrift
checkpoints=./checkpoints/OtherExperiments/SFT_ClosedWorld
log_path=./logs/OtherExperiments/SFT_ClosedWorld

pretrian_dataset=TemporalDrift
model=NetCLR

python -u -m exp.pretrain \
  --dataset ${pretrian_dataset} \
  --model ${model} \
  --device cuda:3 \
  --train_epochs 100 \
  --train_file train \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --optimizer Adam \
  --save_name pretrain \
  --checkpoints ${checkpoints}

python -u -m exp.train \
  --dataset ${dataset} \
  --model ${model} \
  --device cuda:3 \
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
  --load_file ${checkpoints}/${pretrian_dataset}/NetCLR/pretrain.pth \
  --save_name max_f1 \
  --checkpoints ${checkpoints}

for file_name in day14 day30 day90 day150 day270
do
    rm -rf ${checkpoints}/${dataset}/${model}/sft.pth
    cp ${checkpoints}/${dataset}/${model}/max_f1.pth ${checkpoints}/${dataset}/${model}/sft.pth

    python -u -m exp.sft \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:3 \
        --tune_file ${file_name} \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 5000 \
        --batch_size 128 \
        --k_shot 10 \
        --sft_epochs 30 \
        --sft_lr 1e-4 \
        --optimizer Adam \
        --eval_metrics Accuracy Precision Recall F1-score \
        --save_metric F1-score \
        --checkpoints ${checkpoints} \
        --load_name sft \
        --save_name sft \

    python -u -m exp.test \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:3 \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 5000 \
        --batch_size 256 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --log_path ${log_path} \
        --checkpoints ${checkpoints} \
        --load_name sft \
        --result_file ${model}_${file_name}
done

model=Swallow

python -u -m exp.pretrain \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:3 \
    --train_file train \
    --train_epochs 40 \
    --batch_size 256 \
    --learning_rate 3e-2 \
    --optimizer SGD \
    --save_name pretrain \
    --num_workers 8 \
    --checkpoints ${checkpoints}

python -u -m exp.train \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:3 \
    --train_file train \
    --valid_file valid \
    --feature CIF \
    --seq_len 1000 \
    --train_epochs 30 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --optimizer Adam \
    --eval_metrics Accuracy Precision Recall F1-score \
    --save_metric F1-score \
    --load_file ${checkpoints}/${dataset}/${model}/pretrain.pth \
    --save_name max_f1 \
    --checkpoints ${checkpoints}

for file_name in day14 day30 day90 day150 day270
do
    rm -rf ${checkpoints}/${dataset}/${model}/sft.pth
    cp ${checkpoints}/${dataset}/${model}/max_f1.pth ${checkpoints}/${dataset}/${model}/sft.pth

    python -u -m exp.sft \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:3 \
        --tune_file ${file_name} \
        --test_file ${file_name} \
        --feature CIF \
        --seq_len 1000 \
        --batch_size 128 \
        --k_shot 10 \
        --sft_epochs 30 \
        --sft_lr 1e-4 \
        --optimizer Adam \
        --eval_metrics Accuracy Precision Recall F1-score \
        --save_metric F1-score \
        --load_name sft \
        --save_name sft \

    python -u -m exp.test \
        --dataset ${dataset} \
        --model ${model} \
        --device cuda:3 \
        --valid_file valid \
        --test_file ${file_name} \
        --feature CIF \
        --seq_len 1000 \
        --batch_size 64 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --log_path ${log_path} \
        --checkpoints ${checkpoints} \
        --load_name sft \
        --result_file ${model}_${file_name}
done

model=TF

python -u -m exp.train \
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
  --save_name max_f1 \
  --log_path ${log_path} \
  --checkpoints ${checkpoints}

for file_name in day14 day30 day90 day150 day270
do
    rm -rf ${checkpoints}/${dataset}/${model}/sft.pth
    cp ${checkpoints}/${dataset}/${model}/max_f1.pth ${checkpoints}/${dataset}/${model}/sft.pth

    python -u -m exp.sft \
      --dataset ${dataset} \
      --model ${model} \
      --device cuda:2 \
      --tune_file ${file_name} \
      --test_file ${file_name} \
      --feature DIR \
      --seq_len 5000 \
      --batch_size 128 \
      --k_shot 10 \
      --sft_epochs 30 \
      --sft_lr 1e-4 \
      --optimizer Adam \
      --eval_method kNN \
      --eval_metrics Accuracy Precision Recall F1-score \
      --save_metric F1-score \
      --checkpoints ${checkpoints} \
      --load_name sft \
      --save_name sft \

    python -u -m exp.test \
      --dataset ${dataset} \
      --model ${model} \
      --device cuda:2 \
      --test_file ${file_name} \
      --feature DIR \
      --seq_len 5000 \
      --batch_size 256 \
      --eval_method kNN \
      --eval_metrics Accuracy Precision Recall F1-score \
      --log_path ${log_path} \
      --checkpoints ${checkpoints} \
      --load_name sft \
      --result_file ${model}_${file_name}
done
