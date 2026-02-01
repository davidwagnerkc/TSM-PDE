export PSCRATCH=/scratch/dwcgt/tsm-pde
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PKGS/nvidia/cudnn/lib:$SITE_PKGS/nvidia_cudnn9/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PATH=$SITE_PKGS/nvidia/cuda_nvcc/bin:$PATH

export TRAINING_DATA_SEED=${TRAINING_DATA_SEED:-909}
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0
export STORAGE_PATH=$PSCRATCH/cfd
export MODEL_NAME=dns_2048x2048_train_${TRAINING_DATA_SEED}
export PREDICTDATA=models/${MODEL_NAME}/predict.nc

python -u tsm/dns_warmup.py \
  --model_predict_steps=16 \
  --delta_time=0.007012483601762931 \
  --num_samples=16 \
  --inner_steps=1 \
  --train_init_random_seed=$TRAINING_DATA_SEED \
  --model_input_size=2048 \
  --warmup_time=40.0 \
  --simulation_time=1.0 \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="tsm/configs/implicit_diffusion_dns_config.gin" \
  --gin_file="tsm/configs/kolmogorov_forcing.gin"
