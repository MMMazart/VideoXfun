export MODEL_NAME="/mnt/data/public_ckpt_1279/Wan-AI/Wan2.1-I2V-14B-480P"
export DATASET_NAME="datasets/kiss_and_hug_20250715"
export DATASET_META_NAME="datasets/kiss_and_hug_20250715/json_of_kiss_and_hug.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

cd /mnt/data/ssd/user_workspace/liuyu6/VideoX-Fun-ft

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard train_lora.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=632 \
  --video_sample_size=632 \
  --token_sample_size=632 \
  --video_sample_stride=1 \
  --video_sample_fps=16 \
  --video_sample_n_frames=81 \
  --train_batch_size=2 \
  --video_repeat=50 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=4e-05 \
  --seed=42 \
  --save_state \
  --output_dir="output_dir/kiss_hug" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --use_deepspeed \
  --train_mode="inpaint" \
  --low_vram \
  --weighting_scheme="logit_normal"