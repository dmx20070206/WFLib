bash scripts/OpenWorld/TemporalDrift/AWF.sh &
bash scripts/OpenWorld/TemporalDrift/BAPM.sh &
bash scripts/OpenWorld/TemporalDrift/ARES.sh &
bash scripts/OpenWorld/TemporalDrift/DF.sh &
bash scripts/OpenWorld/TemporalDrift/NetCLR.sh &
bash scripts/OpenWorld/TemporalDrift/Swallow.sh &
bash scripts/OpenWorld/TemporalDrift/TikTok.sh &
bash scripts/OpenWorld/TemporalDrift/VarCNN.sh &
bash scripts/OpenWorld/TemporalDrift/RF.sh

wait
echo "[DEBUG] All TemporalDrift bash commands have finished."
