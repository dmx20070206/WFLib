set -e

dataset=VersionDrift/048
id_model=Proteus
kplus1_model=RF
device=cuda:0
feature=TAM
seq_len=1800
test_file=tam_drift
extra_test_file=tam_bg_tune
energy_model_file=checkpoints/${dataset}/${id_model}/max_f1.pth
msp_model_file=checkpoints/${dataset}/${id_model}/max_f1.pth
entropy_model_file=checkpoints/${dataset}/${id_model}/max_f1.pth
kplus1_model_file=checkpoints/${dataset}/${kplus1_model}/max_f1.pth

[[ -f ${energy_model_file} ]] || { echo "Missing file: ${energy_model_file}"; exit 1; }
[[ -f ${msp_model_file} ]] || { echo "Missing file: ${msp_model_file}"; exit 1; }
[[ -f ${entropy_model_file} ]] || { echo "Missing file: ${entropy_model_file}"; exit 1; }
[[ -f ${kplus1_model_file} ]] || { echo "Missing file: ${kplus1_model_file}"; exit 1; }

python -u -m other_experiments.ood_summary \
    --id_model ${id_model} \
    --id_model_file ${energy_model_file} \
    --kplus1_model ${kplus1_model} \
    --kplus1_model_file ${kplus1_model_file} \
    --device ${device} \
    --test_file ${dataset}/${test_file} \
    --extra_test_file ${extra_test_file} \
    --feature ${feature} \
    --seq_len ${seq_len}