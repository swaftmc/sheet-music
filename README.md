# Sheet Music Transcriber

Local web app: upload an audio or video file, pick an instrument, and get sheet music on screen plus MusicXML, MIDI, and PDF (via the browser's print dialog).

Pipeline: bundled ffmpeg (`imageio-ffmpeg`) pulls the audio out, a note detector turns it into notes, and music21 snaps them to the detected beats and writes the score. OpenSheetMusicDisplay renders it in the page.

Pick the **Model** that matches the recording:

| Model | Use for | How it works |
|---|---|---|
| Piano | Solo piano | The piano model of Kong et al., "High-resolution Piano Transcription with Pedals" |
| Basic Pitch | One instrument or voice, no backing | Spotify's Basic Pitch |
| Demucs + Basic Pitch | The melody (violin, voice, flute) of a song with drums, bass, piano or guitar behind it | Demucs removes the backing, Basic Pitch scores every pitch frame by frame, and a Viterbi tracker follows one continuous line through it |

## Setup (Python 3.11; Basic Pitch doesn't support 3.14)

```
git clone https://github.com/swaftmc/sheet-music.git
cd sheet-music
py -3.11 -m venv .venv
.venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
```

Models download themselves the first time you pick them: the piano model (172 MB, saved in `models/`) and Demucs `htdemucs_6s` (about 50 MB, saved in the torch cache). If the piano download fails, save the file from [Zenodo](https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1) to `models/piano_note_pedal.pth`.

## Run

```
.venv\Scripts\python app.py
```

Open http://127.0.0.1:5000.

## Check

```
.venv\Scripts\python check_transcribe.py
```

## Limits

- Works best on one instrument or voice. For songs with backing, use the Demucs + Basic Pitch model. Demucs has no violin stem, so strings and pads that share the "other" stem can still leak into the melody.
- Demucs + Basic Pitch takes about 1-1.5x the song's length on CPU (separation plus note detection).
- With Demucs + Basic Pitch, the tracker penalises big jumps, and it writes rests only where the separated lead audio drops about 12 dB below its usual level. A very quiet passage can come out as rests, and a pad that keeps sounding through a real pause can still fill it.
- Notes more than a sixth above the surrounding melody are moved down an octave (this removes overtones). A real high note that far above its neighbours gets moved down too. Low notes are never moved up.
- A note that leaps 9 or more semitones away from both neighbours in the same direction (an up-down or down-up zigzag) is moved by an octave toward them; these come from the backing or an overtone. A real out-and-back leap that big is changed too, unless there's a pause of at least a quarter second on both sides.
- A held note is split wherever another pitch attacks and Basic Pitch rates it higher, so a fast figure over a sustained note isn't written as one long note. Figures faster than about 100 ms per note still get merged, and soft re-bows of the same pitch without a clear attack stay one note.
- Each beat uses either 16ths or triplets, never both, and gaps under half a beat are closed instead of written as rests. Very fast ornaments get simplified.
- Pedalled notes are notated only until the next note starts, so held bass octaves and melody notes over an arpeggio show up shorter than written.
- The grid follows the detected beats, so rubato stays aligned. Where the beat tracker slips, a few measures still get odd tuplets.
- The piano model runs on CPU at about 1.5x the recording's length (a 15-minute piece takes about 20 minutes). Trim long files first if you only need a section.
- Assumes 4/4 time. The tempo is auto-detected (you can override it), and the key is estimated.
- Non-piano instruments get a single melody line (the highest note at each moment), folded into the instrument's range. Bb/Eb instruments are written transposed.
- The page loads OpenSheetMusicDisplay from a CDN, so viewing needs internet. Downloads and transcription work offline.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE): free to use, change and share for non-commercial purposes. For commercial
use, ask first: open an issue at [swaftmc/sheet-music](https://github.com/swaftmc/sheet-music/issues).

Not included here, and under their own licences: the libraries from `requirements.txt` and the models that download on
first use (the piano model of Kong et al. and Demucs).
