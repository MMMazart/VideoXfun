import os
import sys
import argparse

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                              WanTransformer3DModel)
from videox_fun.pipeline import WanFunInpaintPipeline
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent, get_image_to_480_video_latent,
                                   save_videos_grid)
import torch.distributed as dist
from tqdm import tqdm


def save_results(save_path, line, video_length, sample, fps):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    prefix = str(line['ID']) + '_' + str(line['seed']).zfill(8)
    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def main(args):
    rank, world_size = setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    print(f"rank:{rank}-local_rank:{local_rank}-world_size:{world_size}")

    # Config and model path
    config_path         = "config/wan2.1/wan_civitai.yaml"
    # model path
    model_name          = args.model_name 
    # Choose the sampler in "Flow"
    sampler_name        = "Flow"

    # Load pretrained model if need
    transformer_path    = None
    vae_path            = None
    lora_path           = args.lora_path

    # Other params
    video_length        = 81
    fps                 = 16
    weight_dtype        = torch.bfloat16
    
    negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    guidance_scale      = 6.0
    num_inference_steps = 40
    lora_weight         = args.lora_weight
    save_path           = args.save_path

    config = OmegaConf.load(config_path)

    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Get Clip Image Encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype)
    clip_image_encoder = clip_image_encoder.eval()

    # Get Scheduler
    Choosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
    }[sampler_name]
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = WanFunInpaintPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder
    )

    pipeline.to(device=device)

    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight)

    lines = open(args.benchmark_path).read().strip().split('\n')
    lines = [eval(line) for line in lines]

    for line in tqdm(lines[rank::world_size], disable = (rank != 0) ):
        seed = line['seed']
        generator = torch.Generator(device=device).manual_seed(seed)
        if 'img_path' in line:
            validation_image_start = line['img_path']
        else:
            validation_image_start = line['img_paths'][0]
        validation_image_end = None
        prompt = line[args.prompt]

        input_video, input_video_mask, clip_image, sample_size = get_image_to_480_video_latent(validation_image_start, validation_image_end, video_length=video_length)

        with torch.no_grad():
            video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

            input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)

            sample = pipeline(
                prompt, 
                num_frames = video_length,
                negative_prompt = negative_prompt,
                height      = sample_size[0],
                width       = sample_size[1],
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,
                video      = input_video,
                mask_video   = input_video_mask,
                clip_image = clip_image,
            ).videos

        save_results(save_path, line, video_length, sample, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--benchmark_path', type=str)
    parser.add_argument('--save_path', type=str, default='samples/wan-videos-fun-i2v')
    parser.add_argument('--model_name', type=str, default='/mnt/data/ssd/public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP')
    parser.add_argument('--prompt', type=str, default='prompt')
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--lora_weight', type=float, default=0.55)

    args = parser.parse_args()

    main(args)