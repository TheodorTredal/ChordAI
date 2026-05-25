import torch
from diffusers import DiffusionPipeline

# 1. Definer hvilken modell du vil hente fra HuggingFace Hub
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
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

# 4. Generer bildet
prompt = ("A sailboat on the ocean")

# num_inference_steps bestemmer hvor mange ganger U-Net skal rense støy (typisk 30-50)
image = pipe(
    prompt=prompt,
    negative_prompt="blurry, low quality, text, typography, distorted, human, person, face, portrait, eyes, woman, man, crowd, distorted anatomy",
    num_inference_steps=40,
    guidance_scale=2.5,
    # generator=generator,
    width=1024,
    height=1024
).images[0]

# 5. Lagre resultatet lokalt
fileName = "pop2"
image.save(f"{fileName}.png")
print(f"{fileName}.png!")