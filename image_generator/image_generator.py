import torch
from diffusers import DiffusionPipeline

# 1. Definer hvilken modell du vil hente fra HuggingFace Hub
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_id = "Lykon/dreamshaper-xl-v2-turbo"
generator = torch.Generator(device="cuda").manual_seed(9)

# 2. Last inn hele pipelinen. 
# Vi bruker float16 (halv presisjon) for å spare tonnevis med VRAM.
pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    use_safetensors=True
)

# 3. Send modellen til GPU-en din
# Hvis du bruker Mac: endre "cuda" til "mps"
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
elif torch.backends.mps.is_available():
    pipe = pipe.to("mps")
else:
    print("Warning: Running on CPU will be EXTREMELY slow.")
    pipe = pipe.to("cpu")

prompt = (
    "A nostalgic 1980s synth-pop album cover, retro-futurism aesthetic. "
    "A lonely sports car driving towards a glowing neon sunset on a grid horizon. "
    "Calm, misty atmosphere, synthwave colors, pink and purple hues, vintage 35mm film grain, cinematic composition"
)


negative_prompt = (
    "blurry, low quality, bad anatomy, text, typography, watermark, logo, "
    "modern digital look, overexposed, human, face, portrait"
)
# num_inference_steps bestemmer hvor mange ganger U-Net skal rense støy (typisk 30-50)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=7.5,
    # generator=generator,
    width=1024,
    height=1024
).images[0]

# 5. Lagre resultatet lokalt
fileName = "tmp_images/image"
image.save(f"{fileName}.png")
print(f"{fileName}.png!")