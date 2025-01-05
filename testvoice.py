import sys
import torchaudio
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice

# Initialize with the SFT model
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)

# Test with English Male voice
text = "Paul awoke to feel himself in the warmth of his bed—thinking ... thinking. This world of Castle Caladan, without play or companions his own age, perhaps did not deserve sadness in farewell. Dr. Yueh, his teacher, had hinted that the faufreluches class system was not rigidly guarded on Arrakis. The planet sheltered people who lived at the desert edge without caid or bashar to command them: will-o'-the-sand people called Fremen, marked down on no census of the Imperial Regate."
for i, output in enumerate(cosyvoice.inference_sft(text, '英文男')):  # English Male voice
    output_path = f'english_male_voice_{i}.wav'
    torchaudio.save(output_path, output['tts_speech'], cosyvoice.sample_rate)
    print(f"Saved output to {output_path}")