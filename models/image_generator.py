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

    negative_prompt = (
        "blurry, low quality, bad anatomy, text, typography, watermark, logo, "
        "modern digital look, overexposed, human, face, portrait"
    )

    # Generer bildet (Satt til 10 trinn siden det er en Turbo-modell)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=10,
        guidance_scale=2.0, # Turbo-modeller trives ofte best rundt 1.5 - 3.0
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