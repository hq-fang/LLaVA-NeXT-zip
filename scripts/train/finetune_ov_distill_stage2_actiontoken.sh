


export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=ib
#export NCCL_SOCKET_IFNAME=bond0.2080
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
#export NCCL_IB_HCA=^=mlx5_bond_0
export NCCL_DEBUG=INFO

############### Pretrain ################

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

export RANK=$BEAKER_REPLICA_RANK
export ADDR=$BEAKER_LEADER_REPLICA_HOSTNAME
export PORT=29500
#RANK=0
#ADDR="127.0.0.1"
#PORT="29501"
#PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NNODES=16
NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH="/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH}"
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_FASTRAK_NUM_FLOWS=2
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_FASTRAK_USE_SNAP=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
export NCCL_TUNER_PLUGIN=libnccl-tuner.so
export NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto
export NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/var/lib/tcpxo/lib64/a3plus_guest_config.textproto
export NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export NCCL_FASTRAK_CTRL_DEV=enp0s12
export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
export NCCL_SOCKET_IFNAME=enp0s12
export NCCL_USE_SNAP=1
export NCCL_FASTRAK_USE_LLCM=1
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices
echo "NUM_GPUS: ${NUM_GPUS}"
echo "NNODES: ${NNODES}"
echo "RANK: ${RANK}"
echo "ADDR: ${ADDR}"
echo "PORT: ${PORT}"
export NUMACTL_CMD=""
export NCCL_PROTO=Simple,LL128
export NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto
export NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto
# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9" 
# PREV_STAGE_CHECKPOINT= "/data/input/jiafei/LLaVA-NeXT/checkpoints/onevision/lmms-lab/llava-onevision-qwen2-7b-ov" # replace it with your last checkpoint training from single image collection
# PREV_STAGE_CHECKPOINT= "/data/input/jiafei/LLaVA-NeXT/checkpoints/onevision/2mar_depth_mixed_aug_1/checkpoint-12000-1"
# PREV_STAGE_CHECKPOINT="jaslee20/llava-epoch3-qwen2-unified"
# PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov"
PREV_STAGE_CHECKPOINT="/data/input/jiafei/GroundedVLA/checkpoint/lmms-lab/llava-onevision-qwen2-7v-ov"
# PREV_STAGE_CHECKPOINT="/data/input/jiafei/GroundedVLA/checkpoint/mar20_full/checkpoint-11000"
# PREV_STAGE_CHECKPOINT="/data/input/jiafei/LLaVA-NeXT/checkpoints/onevision/new_7b_pose_tokenize3_simple_small_10epoch/checkpoint-85000"
# PREV_STAGE_CHECKPOINT="/net/nfs/prior/jiafei/unified_VLM/LLaVA-NeXT/checkpoints/backup/checkpoint-89000" # replace it with your last checkpoint training from single image collection
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 WANDB_API_KEY=b0161ce9ee3d3f5a6a2b28ffffcd098211e8376d torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    /data/input/jiafei/GroundedVLA/LLaVA-NeXT/llava/train/train_mem.py \
    --deepspeed /data/input/jiafei/GroundedVLA/LLaVA-NeXT/scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /data/input/jiafei/GroundedVLA/LLaVA-NeXT/scripts/train/onevision_distill_training_ak_a100.yaml \
    --image_folder gs://vision-jiafeid  \
    --video_folder gs://vision-jiafeid \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /data/input/jiafei/GroundedVLA/checkpoint/apr1_full_stage2_actiontoken \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --lora_enable False \
    --depth_data True
exit 0;


# You can delete the sdpa attn_implementation if you want to use flash attn




# #!/bin/bash

# # Environment setup
# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# # Optionally uncomment if needed:
# # export NCCL_SOCKET_IFNAME=ib
# # export TORCH_DISTRIBUTED_DEBUG=DETAIL

# LLM_VERSION="Qwen/Qwen2-7B-Instruct"
# LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
# VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
# VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# ############### Pretrain ################

# BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
# echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# export RANK=$BEAKER_REPLICA_RANK
# export ADDR=$BEAKER_LEADER_REPLICA_HOSTNAME
# # The PORT variable is no longer used in the torchrun command below, but we keep it for reference.
# export PORT=29500

# NNODES=32
# NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export LD_LIBRARY_PATH="/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH}"

# # NCCL configuration
# export NCCL_CROSS_NIC=0
# export NCCL_ALGO="Ring,Tree"
# # Note: Overwrite earlier NCCL_PROTO if needed:
# export NCCL_PROTO="Simple,LL128"
# export NCCL_MIN_NCHANNELS=4
# export NCCL_P2P_NET_CHUNKSIZE=524288
# export NCCL_P2P_PCI_CHUNKSIZE=524288
# export NCCL_P2P_NVL_CHUNKSIZE=1048576
# export NCCL_FASTRAK_NUM_FLOWS=2
# export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
# export NCCL_BUFFSIZE=8388608
# export NCCL_FASTRAK_USE_SNAP=1
# export NCCL_NET_GDR_LEVEL=PIX
# export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
# export NCCL_TUNER_PLUGIN="libnccl-tuner.so"
# export NCCL_TUNER_CONFIG_PATH="/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto"
# export NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE="/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto"
# export NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
# export NCCL_NVLS_ENABLE=0
# export NCCL_DEBUG="WARN"
# export NCCL_FASTRAK_CTRL_DEV="enp0s12"
# export NCCL_FASTRAK_IFNAME="enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0"
# export NCCL_SOCKET_IFNAME="enp0s12"
# export NCCL_USE_SNAP=1
# export NCCL_FASTRAK_USE_LLCM=1
# export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY="/dev/aperture_devices"

# echo "NUM_GPUS: ${NUM_GPUS}"
# echo "NNODES: ${NNODES}"
# echo "RANK: ${RANK}"
# echo "ADDR: ${ADDR}"
# echo "PORT: ${PORT}"

# # Stage 2 configuration
# PROMPT_VERSION="qwen_1_5"
# RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9"
# PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov"

# echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
# echo "MID_RUN_NAME: ${RUN_NAME}"

# # Launch training with static rendezvous configuration.
# # Notice we replaced --master_addr and --master_port with --rdzv_* options.
# ACCELERATE_CPU_AFFINITY=1 WANDB_MODE=offline torchrun \
#     --nproc_per_node="${NUM_GPUS}" \
#     --nnodes="${NNODES}" \
#     --node_rank="${RANK}" \
#     --rdzv_backend=static \
#     --rdzv_id=3379161.747210708 \
#     --rdzv_conf="read_timeout=600" \
#     --rdzv_endpoint="${ADDR}:29401" \
#     /data/input/jiafei/GroundedVLA/LLaVA-NeXT/llava/train/train_mem.py \
#     --deepspeed /data/input/jiafei/GroundedVLA/LLaVA-NeXT/scripts/zero2.json \
#     --model_name_or_path $PREV_STAGE_CHECKPOINT \
#     --version $PROMPT_VERSION \
#     --data_path /data/input/jiafei/GroundedVLA/LLaVA-NeXT/scripts/train/onevision_distill_training_ak_a100.yaml \
#     --image_folder gs://vision-jiafeid  \
#     --video_folder gs://vision-jiafeid \
#     --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
#     --mm_vision_tower_lr=2e-6 \
#     --vision_tower ${VISION_MODEL_VERSION} \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --group_by_modality_length True \
#     --image_aspect_ratio anyres_max_9 \
#     --image_grid_pinpoints  "(1x1),...,(6x6)" \
#     --mm_patch_merge_type spatial_unpad \
#     --bf16 True \
#     --run_name $RUN_NAME \
#     --output_dir /data/input/jiafei/GroundedVLA/checkpoint/mar25_full_stage2_actiontoken \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 32768 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --torch_compile True \
#     --torch_compile_backend "inductor" \
#     --dataloader_drop_last True \
#     --frames_upbound 32 \
#     --lora_enable False \
#     --depth_data True

# exit 0;
