cd /mnt/data/ssd/user_workspace/liuyu6/VideoX-Fun-ft


torchrun --nproc_per_node=1 sample/predict_i2v_batch_mgpu.py \
    --benchmark_path=/mnt/data/hdd/user_workspace/wangqin2/video-tools/qwen_sever/output/kiss_hug_output.jsonl \
    --save_path=/mnt/data/ssd/user_workspace/liuyu6/outputs/i2v/kiss_hug/exp1-ft-1k-lora_weight=0.55 \
    --model_name=/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-I2V-14B-480P \
    --prompt=reprompt \
    --lora_path=/mnt/data/ssd/user_workspace/liuyu6/VideoX-Fun-ft/output_dir/kiss_hug/checkpoint-1000/lora_diffusion_pytorch_model.safetensors