import os
import io
import soundfile as sf
from datasets import load_dataset, Audio

os.makedirs("test_samples", exist_ok=True)
os.makedirs("references", exist_ok=True)

print("Streaming Cebuano dataset from Hugging Face...")
# decode=False bypasses torchcodec and delivers raw audio bytes directly
dataset = load_dataset("SilencioNetwork/cebuano-speech", split="test", streaming=True)
dataset = dataset.cast_column("audio", Audio(decode=False))

# Save the first 5 samples
for i, sample in enumerate(dataset.take(5)):
    sample_name = f"ceb_sample_{i+1:03d}"
    audio_path = os.path.join("test_samples", f"{sample_name}.wav")
    ref_path = os.path.join("references", f"{sample_name}_reference.txt")

    # 1. Read audio bytes with soundfile and write as WAV
    audio_bytes = sample["audio"]["bytes"]
    with io.BytesIO(audio_bytes) as bio:
        data, sr = sf.read(bio)
        sf.write(audio_path, data, sr)

    # 2. Save ground truth transcript for automatic WER calculation
    transcript = (sample.get("transcript") or "").strip()
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"[OK] Saved {audio_path} and {ref_path}")
    print(f"   Transcript: {repr(transcript[:70])}...")

print("\nDone! You can now drop any file from test_samples/ into the BANAAG UI.")
