# Banaag

Cleans noisy Cebuano speech (band-pass, spectral subtraction, Wiener) and transcribes it with Whisper.
Runs only on your own computer at http://localhost:5000.

## 1. Install (once)

You need Python 3.10 or newer. In a terminal, inside this folder:

    python -m venv venv
    venv\Scripts\activate          (Windows)
    source venv/bin/activate       (Mac / Linux)
    pip install -r requirements.txt

What the packages are for:

| Package | Used for |
|---|---|
| flask | the local web server |
| numpy, scipy | the filters (STFT, Butterworth, resampling) |
| soundfile | reading and writing WAV/FLAC/MP3 |
| torch, transformers | running the Whisper model |
| jiwer | Word Error Rate |
| librosa (optional) | opening m4a and other formats; also install ffmpeg |

`torch` is large (a few GB). If the install is slow, that is normal.

## 2. Run

    python app.py

Wait for the terminal to settle (the first run downloads the Whisper model, about 1 GB, and needs
internet once). Then open **http://localhost:5000** in your browser.
Do not double-click index.html. It only shows a demo without the backend.

## 3. Use

1. Drop or choose an audio file, press START.
2. Watch the 7 processing stages.
3. On the results page: compare the waveforms, read the transcript, download the cleaned audio or transcript.

## Getting WER automatically

WER needs the correct text. Save it in the `references` folder as `<name>_reference.txt`.
Any upload whose file name starts with `<name>` uses it, for example
`CEB_006_reference.txt` is used for `CEB_006_train_15dB.wav`.
You can also attach a .txt on the upload screen.

## Files

| File | What it does |
|---|---|
| app.py | web server, job queue, runs the 7 stages |
| config.py | all filter and model settings (edit numbers here) |
| audio_io.py | load file, convert to mono 16 kHz |
| dsp.py | band-pass, spectral subtraction, Wiener filter |
| asr.py | Whisper model loading and transcription |
| metrics.py | WER, estimated SNR, reference lookup |
| index.html | the whole user interface (3 screens) |

## Problems

- **"Demo mode" banner or placeholder transcript:** app.py is not running, or you opened the file directly.
- **First upload is slow:** Whisper is still loading or downloading. Later uploads are faster.
- **Could not read that file:** try WAV or MP3; for m4a install librosa and ffmpeg.
- **SNR number:** it is an estimate from the quiet parts of the audio, not the notebook's clean-reference SNR.
