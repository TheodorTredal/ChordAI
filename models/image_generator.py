# models/image_generator.py
import argparse
import os
import time
import torch
from diffusers import DiffusionPipeline

def main():
    print("CREATING AN IMAGE")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    # Sørg for at mappen eksisterer
    os.makedirs(args.out_dir, exist_ok=True)

    model_id = "Lykon/dreamshaper-xl-v2-turbo"
    pipe = DiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        use_safetensors=True
    )

    # --- MINNEOPTIMALISERING FOR DELT SERVER/CUDA ---
    if torch.cuda.is_available():
        # Denne flytter automatisk modellagene mellom RAM og VRAM underveis.
        # Gjør at vi slipper den brutale "Out of Memory"-krasjen!
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        print("Bruker CUDA med CPU Offloading for å spare minne.")
    elif torch.backends.mps.is_available():
        pipe = pipe.to("mps")
        print("Bruker Apple Silicon (MPS).")
    else:
        pipe = pipe.to("cpu")
        print("Bruker CPU (Tregt!).")

    # 1. NY NEGATIV PROMPT: Blokkerer også utvaskede farger og kjedelig komposisjon
    negative_prompt = (
        "human, person, woman, girl, lady, female, man, boy, guy, male, face, portrait, "
        "eyes, close-up, photography of people, text, typography, title, font, watermark, "
        "logo, bad anatomy, blurry, low quality, overexposed, washed out colors, "
        "boring composition, cluttered background, amateur, draft, nudity,"
    )

    # 2. NY QUALITY MODIFIER: Trigger dype, episke og inspirerende kunststiler
    quality_modifiers = (
        "iconic music album cover, vinyl sleeve design, breathtaking conceptual art, "
        "epic scale, sublime atmosphere, divine lighting, hyper-detailed textures, "
        "vibrant yet moody color palette, professional graphic design, masterpiece, "
        "visually striking composition, trending on artstation"
    )

    enhanced_prompt = f"{args.prompt}, {quality_modifiers}"
    

    # Generer bildet
    image = pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=10,
        guidance_scale=2.0,
        width=1024,
        height=1024
    ).images[0]

    # Lagre med et unikt tidsstempel på samme måte som lyrikken din
    timestamp = int(time.time())
    file_path = os.path.join(args.out_dir, f"image_{timestamp}.png")
    
    # VIKTIG: Go-serveren din leser den aller Siste linjen med rå-print 
    # for å skjønne hva filnavnet ble. Derfor flyttet jeg rå-printen til bunnen.
    image.save(file_path)
    
    # Vi printer filbanen helt til slutt slik at Go plukker den opp korrekt
    print(file_path)

if __name__ == "__main__":
    main()