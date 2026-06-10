
# bash scripts/OpenSet/BehaviorDrift.sh &
bash scripts/OpenSet/NetworkDrift.sh &
bash scripts/OpenSet/MultiTab.sh &
bash scripts/OpenSet/TemporalDrift.sh &
bash scripts/OpenSet/VersionDrift.sh
# bash scripts/OpenSet/Defense.sh

wait
echo "[DEBUG] All OpenSet bash commands have finished."