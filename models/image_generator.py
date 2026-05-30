# models/image_generator.py
import argparse
import os
import sys
import time
import torch
from diffusers import DiffusionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model_id = "Lykon/dreamshaper-xl-v2-turbo"
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        print("Using CUDA with CPU offloading.", file=sys.stderr)
    elif torch.backends.mps.is_available():
        pipe = pipe.to("mps")
        print("Using Apple Silicon (MPS).", file=sys.stderr)
    else:
        pipe = pipe.to("cpu")
        print("Using CPU (slow).", file=sys.stderr)

    negative_prompt = (
        "blurry, low quality, bad anatomy, text, typography, watermark, logo, "
        "modern digital look, overexposed, human, face, portrait"
    )

    image = pipe(
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=10,
        guidance_scale=2.0,
        width=1024,
        height=1024,
    ).images[0]

    timestamp = int(time.time())
    file_path = os.path.join(args.out_dir, f"image_{timestamp}.png")
    image.save(file_path)

    # Print the saved path as the last stdout line — Go server reads this.
    print(file_path, flush=True)


if __name__ == "__main__":
    main()