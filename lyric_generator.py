import torch
from transformers import pipeline

# 1. Definer hvilken tekstmodell du vil hente fra HuggingFace Hub
# Vi bruker en mindre Llama-3-modell (8 milliarder parametere) som er rå på tekst
model_id = "allenai/OLMo-2-1124-7B-Instruct"
# MERK: Hvis PC-en din har lite VRAM, kan du bytte til en mindre modell som: "Qwen/Qwen2.5-1.5B-Instruct"

# 2. Bestem enhet (Device) på samme måte som i bilde-koden din
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16 # Sparer VRAM på Nvidia-kort
elif torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.bfloat16 # Sparer VRAM på Mac (M1/M2/M3)
else:
    device = "cpu"
    torch_dtype = torch.float32

# 3. Last inn hele tekst-pipelinen
# Dette tilsvarer DiffusionPipeline.from_pretrained()
text_pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch_dtype,
    device_map=device
)

# 4. Definer prompten (System-instruksjon + Bruker-input)
prompt = [
    {"role": "system", "content": "Du er en låtskriver i ChordAI. Skriv en kort sangtekst tilpasset akkordene. Svar kun med sangteksten."},
    {"role": "user", "content": "Skriv en dreamy pop tekst til disse akkordene på svensk: Am - F - C - G"}
]

# 5. Generer tekst (Tilsvarer å kjøre pipe() på bildet)
# max_new_tokens bestemmer lengden på teksten (tilsvarer litt width/height)
# temperature bestemmer kreativitet (litt som guidance_scale)
outputs = text_pipe(
    prompt,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
)

# 6. Hent ut og print resultatet
sangtekst = outputs[0]["generated_text"][-1]["content"]
print("\n--- GENERERT SANGTEKST ---")
print(sangtekst)