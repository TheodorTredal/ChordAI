import torch
from transformers import pipeline

# # model_id="microsoft/Phi-3-mini-4k-instruct"
model_id="allenai/OLMo-2-1124-7B-Instruct"
class ChordAILyricGenerator:
    def __init__(self, model_id=model_id):
        print(f"Laster inn modell: {model_id}...")
        
        # 1. Automatisk enhets- og VRAM-håndtering
        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.bfloat16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
            
        # 2. Last inn tekst-pipelinen én gang for alle
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        print("Modell lastet inn suksessfullt!")

    def generer_sangtekst(self, vers_akkorder, refreng_akkorder, sjanger, sprak="norsk", kreativitet=0.7, maks_lengde=256):
        """
        Genererer en tilpasset sangtekst basert på innsendte parametere.
        
        kreativitet (temperature): 0.1 (veldig strukturert) til 1.0 (veldig kreativ)
        maks_lengde (max_new_tokens): Antall ord/tokens som skal genereres
        """
        
        # 3. System-prompten som definerer rollen (Denne kan også gjøres dynamisk!)
        system_instruksjon = (
            "Du er en profesjonell låtskriver i ChordAI-appen. Din oppgave er å skrive sangtekster "
            "som passer perfekt til den emosjonelle stemningen i akkordene. "
            "Del teksten tydelig inn i [Vers] og [Refreng], og plasser gjerne akkordene over tekstlinjene. "
            "Svar KUN med selve sangteksten, ingen introduksjon eller høflig prat."
        )
        
        # 4. Bruker-prompten som fylles ut dynamisk med variabler
        bruker_input = f"""
        Skriv en sangtekst på {sprak} i sjangeren {sjanger}.
        
        Følg denne strukturen:
        Vers-akkorder: {vers_akkorder}
        Refreng-akkorder: {refreng_akkorder}
        """
        
        prompt = [
            {"role": "system", "content": system_instruksjon},
            {"role": "user", "content": bruker_input}
        ]
        
        # 5. Generering med de tilpassede parameterne
        outputs = self.pipe(
            prompt,
            max_new_tokens=maks_lengde,
            temperature=kreativitet,
            do_sample=True if kreativitet > 0 else False,
        )
        
        return outputs[0]["generated_text"][-1]["content"]

# --- HIER TESTER VI DEN CUSTOMIZABLE KODEN ---
if __name__ == "__main__":
    # Initialiser generatoren (dette gjøres bare én gang når appen starter)
    generator = ChordAILyricGenerator()
    
    # Test 1: En trist norsk folketone/pop
    print("\n--- TEST 1: Norsk Melankoli ---")
    tekst_1 = generator.generer_sangtekst(
        vers_akkorder="Am - Em - F - C",
        refreng_akkorder="F - G - Am - C",
        sjanger="Melankolsk Pop",
        sprak="norsk",
        kreativitet=0.6 # Litt mer strukturert og emosjonell
    )
    print(tekst_1)
    
    # Test 2: En munter engelsk rockelåt
    print("\n--- TEST 2: Engelsk Rock ---")
    tekst_2 = generator.generer_sangtekst(
        vers_akkorder="G - C - D - G",
        refreng_akkorder="C - D - G - Em",
        sjanger="Upbeat Indie Rock",
        sprak="engelsk",
        kreativitet=0.9, # Høyere kreativitet for villere rocketekst
        maks_lengde=150  # Kort og konsist
    )
    print(tekst_2)