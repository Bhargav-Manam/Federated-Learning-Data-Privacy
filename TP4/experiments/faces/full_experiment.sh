cd ../../

DEVICE="cpu"

N_ROUNDS=10
N_ATTACK_ROUNDS=50
LOCAL_STEPS=1
LOG_FREQ=5

echo "Experiment with faces dataset"

echo "=> Generate data.."

cd data/ || exit

rm -r faces

python3 main.py \
  --dataset faces \
  --n_clients 10 \
  --by_labels_split \
  --alpha 0.13 \
  --save_dir faces \
  --seed 1234

cd ../


echo "==> Run experiment with faces dataset"

echo "=> experiment = Model Inversion Attack"
python3 run_experiment.py \
  --experiment "faces" \
  --cfg_file_path data/faces/cfg.json \
  --objective_type weighted \
  --aggregator_type centralized \
  --n_rounds "${N_ROUNDS}" \
  --n_attack_rounds "${N_ATTACK_ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --local_optimizer sgd \
  --local_lr 0.05 \
  --server_optimizer sgd \
  --server_lr 0.5 \
  --train_bz 64 \
  --test_bz 1024 \
  --device "${DEVICE}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs_tuning/faces/faces_lr_${lr}_server_${server_lr}/seed_${seed}" \
  --seed 12
