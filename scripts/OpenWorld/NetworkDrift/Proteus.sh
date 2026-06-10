bash scripts/OpenWorld/NetworkDrift/AWF.sh &
bash scripts/OpenWorld/NetworkDrift/BAPM.sh &
bash scripts/OpenWorld/NetworkDrift/ARES.sh &
bash scripts/OpenWorld/NetworkDrift/DF.sh &
bash scripts/OpenWorld/NetworkDrift/NetCLR.sh &
bash scripts/OpenWorld/NetworkDrift/Swallow.sh &
bash scripts/OpenWorld/NetworkDrift/TikTok.sh &
bash scripts/OpenWorld/NetworkDrift/VarCNN.sh &
bash scripts/OpenWorld/NetworkDrift/RF.sh

wait
echo "[DEBUG] All NetworkDrift bash commands have finished."
