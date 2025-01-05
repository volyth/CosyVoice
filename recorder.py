import sys
import os
import time
import whisper
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf
import numpy as np
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice

def record_audio(duration=5, sample_rate=16000):
    print("Recording will start in 3 seconds...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    
    print("🔴 Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("✅ Done recording!")
    
    # Use absolute path
    temp_path = os.path.abspath("temp_recording.wav")
    print(f"Saving recording to: {temp_path}")
    sf.write(temp_path, recording, sample_rate)
    
    # Add delay to ensure file is saved
    time.sleep(1)  # Wait 1 second
    
    # Verify file exists
    attempts = 0
    while not os.path.exists(temp_path) and attempts < 5:
        print(f"Waiting for file to be saved... (attempt {attempts + 1})")
        time.sleep(1)
        attempts += 1
    
    if os.path.exists(temp_path):
        print(f"File successfully saved at {temp_path}")
        file_size = os.path.getsize(temp_path)
        print(f"File size: {file_size} bytes")
    else:
        raise FileNotFoundError(f"Failed to save file at {temp_path}")
    
    return temp_path

def main():
    try:
        # Initialize models
        print("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        
        print("Loading CosyVoice model...")
        cosyvoice_model = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)
        
        while True:
            print("\n=== Voice Chat Program ===")
            print("1. Start recording (5 seconds)")
            print("2. Quit")
            choice = input("Select option: ")
            
            if choice == '1':
                # Record
                audio_path = record_audio()
                
                # Double check file exists before transcription
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found at {audio_path}")
                
                print(f"\nTranscribing file: {audio_path}")
                result = whisper_model.transcribe(audio_path)
                print(f"Transcribed text: {result['text']}")
                
                # Synthesize and play response
                print("\nSynthesizing speech...")
                for i, output in enumerate(cosyvoice_model.inference_sft(result['text'], '英文男')):
                    output_path = os.path.abspath('response.wav')
                    torchaudio.save(output_path, output['tts_speech'], cosyvoice_model.sample_rate)
                    
                    print(f"\nPlaying response from: {output_path}")
                    data, samplerate = sf.read(output_path)
                    sd.play(data, samplerate)
                    sd.wait()
            
            elif choice == '2':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()