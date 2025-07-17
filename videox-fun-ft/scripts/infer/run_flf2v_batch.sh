cd /mnt/data/ssd/user_workspace/liuyu6/VideoX-Fun-ft

# torchrun --nproc_per_node=1 --master_port=29511 /mnt/data/ssd/user_workspace/liuyu6/VideoX-Fun-ft/sample/predict_flf2v_batch_mgpu.py \
#     --benchmark_path=/mnt/data/ssd/user_workspace/liuyu6/tools/benchmark/imgs/flf2v_benchmark/pe_benchmark.jsonl \
#     --save_path=/mnt/data/ssd/user_workspace/liuyu6/outputs/flf2v/wan2.1-14B-flf2v-ft-1k-lora_weight=1-exp2 \
#     --model_name=/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/ \
#     --prompt=reprompt \
#     --lora_path='/mnt/thscc/workspace/dsw/904/interactive-ln4jc82pdo99/log/flf2v/flf2v_young2old_exp2/checkpoint-1000/lora_diffusion_pytorch_model.safetensors' \
#     --lora_weight=1.0
#     # --merge_model_path=/mnt/data/ssd/user_workspace/liuyu6/VideoX-Fun-ft/output_dir/flf2v_young2old_exp1/checkpoint-2000/merge_model-lw=1.0/merge_model.pt

torchrun --nproc_per_node=8 --master_port=29511 /mnt/data/ssd/user_workspace/liuyu6/VideoX-Fun-ft/sample/predict_flf2v_batch_mgpu.py \
    --benchmark_path=/mnt/data/hdd/user_workspace/wangqin2/Wan2.1-main/output/flf/pe_refine_1.jsonl \
    --save_path=/mnt/data/ssd/user_workspace/liuyu6/outputs/flf2v/wan2.1-14B-flf2v-ft-2k-lora_weight=1-exp2-wq \
    --model_name=/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-FLF2V-14B-720P/ \
    --lora_weight=1.0 \
    --prompt=reprompt \
    --lora_path='/mnt/thscc/workspace/dsw/904/interactive-ln4jc82pdo99/log/flf2v/flf2v_young2old_exp2/checkpoint-2000/lora_diffusion_pytorch_model.safetensors'