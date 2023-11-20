cd ../../

DEVICE="cpu"

N_ROUNDS=100
N_ATTACK_ROUNDS=50
LOCAL_STEPS=1
LOG_FREQ=5

echo "Experiment with faces dataset"

echo "=> Generate data.."

cd data/ || exit

rm -r faces

python main.py \
  --dataset faces \
  --n_clients 10 \
  --iid \
  --save_dir faces \
  --seed 1234

cd ../


echo "==> Run experiment with faces dataset"

for seed in 12
do
  for lr in 0.05
  do
    for server_lr in 0.5
    do
      echo "=> experiment=ModelInversionAttack | lr=${lr} | server_lr=${server_lr} | seed=${seed}"
      python run_experiment.py \
        --experiment "faces" \
        --cfg_file_path data/faces/cfg.json \
        --objective_type weighted \
        --aggregator_type centralized \
        --n_rounds "${N_ROUNDS}" \
        --n_attack_rounds "${N_ATTACK_ROUNDS}" \
        --local_steps "${LOCAL_STEPS}" \
        --local_optimizer sgd \
        --local_lr "${lr}" \
        --server_optimizer sgd \
        --server_lr "${server_lr}" \
        --train_bz 64 \
        --test_bz 1024 \
        --device "${DEVICE}" \
        --log_freq "${LOG_FREQ}" \
        --verbose 1 \
        --logs_dir "logs_tuning/faces/faces_lr_${lr}_server_${server_lr}/seed_${seed}" \
        --seed "${seed}"
    done
  done
done