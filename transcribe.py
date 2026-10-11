"""Audio/video file -> notes (piano model, Basic Pitch, or Demucs + Basic Pitch) -> sheet music (music21)."""
import math
import subprocess
import urllib.request
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
from music21 import chord, clef, instrument, key, layout, metadata, meter, note, stream, tempo

# name -> (music21 instrument class, lowest, highest sounding MIDI pitch, plays chords?)
INSTRUMENTS = {
    "piano": (instrument.Piano, 21, 108, True),
    "guitar": (instrument.AcousticGuitar, 40, 88, True),
    "violin": (instrument.Violin, 55, 100, False),
    "flute": (instrument.Flute, 60, 96, False),
    "cello": (instrument.Violoncello, 36, 76, False),
    "clarinet (Bb)": (instrument.Clarinet, 50, 91, False),
    "trumpet (Bb)": (instrument.Trumpet, 54, 82, False),
    "alto sax (Eb)": (instrument.AltoSaxophone, 49, 81, False),
}
# id -> (label shown in the model picker, message shown while it runs)
MODELS = {
    "piano": ("Piano: Kong et al. piano model (most accurate for solo piano)",
              "Transcribing with the piano model... about 1.5x the recording's length on CPU. "
              "The first run downloads the 172 MB model."),
    "basic-pitch": ("Basic Pitch: one instrument or voice, no backing",
                    "Transcribing... this can take a minute for long recordings."),
    "melody": ("Demucs + Basic Pitch: the melody of a song with backing",
               "Separating the melody from the backing, then transcribing... about 1-1.5x the recording's length. "
               "The first run downloads the Demucs model (about 50 MB)."),
}
GRID = 4  # subdivisions per quarter note, i.e. 16th-note quantization


def extract_audio(src, dst, channels=1, rate=22050):
    """Decode any audio/video file to WAV with the bundled ffmpeg."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src), "-vn", "-ac", str(channels), "-ar", str(rate), str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise ValueError("Could not read audio from this file: " + r.stderr.strip().splitlines()[-1])


PIANO_CKPT = Path(__file__).parent / "models" / "piano_note_pedal.pth"
PIANO_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
_piano_model = None


def _download_piano_checkpoint():
    """Fetch the piano model (about 172 MB) on first use; a partial download never replaces the real file."""
    if PIANO_CKPT.exists():
        return
    # ponytail: two simultaneous first runs would both download; fine for a one-user local app.
    PIANO_CKPT.parent.mkdir(parents=True, exist_ok=True)
    part = PIANO_CKPT.with_suffix(".part")
    try:
        urllib.request.urlretrieve(PIANO_URL, part)  # raises if the download is cut short
    except OSError as e:
        part.unlink(missing_ok=True)
        raise ValueError(f"Could not download the piano model ({e}). Save it to {PIANO_CKPT} by hand.") from e
    part.replace(PIANO_CKPT)


def _piano_events(y):
    """Piano-specific model (Kong et al. 2020); far fewer ghost notes than Basic Pitch on piano recordings."""
    global _piano_model
    import torch
    from piano_transcription_inference import PianoTranscription

    if _piano_model is None:
        _download_piano_checkpoint()
        _piano_model = PianoTranscription(checkpoint_path=str(PIANO_CKPT), device=torch.device("cpu"))
        # Pedal output is never notated, so run only the note network (about half the CPU time).
        _piano_model.model = _piano_model.model.note_model
    ev = _piano_model.transcribe(y, None)["est_note_events"]
    return [(e["onset_time"], e["offset_time"], e["midi_note"]) for e in ev]


def isolate_lead(stereo_wav, dst):
    """Strip drums, bass, guitar and piano with Demucs htdemucs_6s; write vocals+other (where lead lines land)."""
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs_6s")
    model.eval()
    wav, sr = sf.read(str(stereo_wav), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.T.copy())
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        src = apply_model(model, ((wav - mean) / std)[None], device="cpu", progress=False)[0] * std + mean
    lead = src[model.sources.index("vocals")] + src[model.sources.index("other")]
    sf.write(str(dst), lead.mean(0).numpy(), sr)


def _track_melody(model_output, y, sr):
    """Follow one pitch (or a rest) per frame through Basic Pitch's note probabilities with Viterbi, then cut notes.
    Jumps are penalised so the line stays off the accompaniment, and rests win only where the lead audio goes quiet."""
    from basic_pitch.note_creation import model_frames_to_time

    # ponytail: weights tuned on 8 synthetic mixes and one song; a trained melody-extraction model is the upgrade.
    lo, hi = 55, 100  # G3-E7: violin, flute and most voices
    sal = model_output["note"][:, lo - 21: hi - 20]
    ons = model_output["onset"][:, lo - 21: hi - 20]
    T, P = sal.shape
    times = model_frames_to_time(T)
    # Harmonic sum: score each pitch by the energy at its first 8 harmonics, so a low violin note with a weak
    # fundamental still beats its own first overtone. Basic Pitch alone often picks that overtone (an octave too high).
    top = min(122, int(librosa.hz_to_midi(0.45 * sr)))  # highest harmonic bin, kept under Nyquist
    M = np.abs(librosa.cqt(y, sr=sr, hop_length=256, fmin=librosa.midi_to_hz(lo), n_bins=top - lo + 1)) ** 0.5
    ct = librosa.frames_to_time(np.arange(M.shape[1]), sr=sr, hop_length=256)
    M = np.stack([np.interp(times, ct, row) for row in M], 1)
    H = np.zeros_like(sal)
    for i in range(P):
        hs = [(0.84 ** (h - 1), i + int(round(12 * np.log2(h)))) for h in range(1, 9)]
        hs = [(w, j) for w, j in hs if j < M.shape[1]]
        H[:, i] = sum(w * M[:, j] for w, j in hs) / sum(w for w, _ in hs)
    prob = sal  # Basic Pitch's own note probabilities, used to decide where notes change
    sal = sal * H / (H.max(1, keepdims=True) + 1e-9)
    # Rest likelihood: 0.2 while the lead plays, rising to 1 once it is 12-18 dB below its typical (75th pct) level.
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=2048, hop_length=256)[0] + 1e-9)
    db = np.interp(times, librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=256), db)
    rest = 0.2 + 0.8 * np.clip((np.percentile(db, 75) - 12 - db) / 6, 0, 1)
    # Staying on a pitch is ~65x likelier than a semitone step; each further semitone costs another e^-0.5.
    A = np.exp(-np.abs(np.subtract.outer(np.arange(P), np.arange(P))) / 2.0)
    np.fill_diagonal(A, 40.0)
    A = np.block([[A, np.full((P, 1), 0.5)], [np.full((1, P), 0.05), np.array([[40.0]])]])
    path = librosa.sequence.viterbi(np.vstack([sal.T, rest[None]]), A / A.sum(1, keepdims=True))
    notes, t = [], 0
    while t < T:
        u = t
        while u < T and path[u] == path[t]:
            u += 1
        q = path[t]
        if q < P:
            # The stay-put prior can hold one pitch straight through a fast figure (a "hanging" note). Split where
            # another pitch attacks, the held pitch is not re-attacked, and Basic Pitch rates the new pitch higher.
            # This uses Basic Pitch's own probabilities because the harmonic sum rates a note and the pitch whose
            # overtone it is (an octave or a twelfth below) about equally. A strong attack on the held pitch is a
            # new bow stroke.
            segs = [(t, q)]
            for k in range(t + 6, u - 6):
                start, cur = segs[-1]
                if k - start < 9:  # at most one split per ~100 ms
                    continue
                hit = [r for r in np.nonzero(ons[k] > 0.6)[0] if ons[k, r] >= ons[k - 1: k + 2, r].max()]
                now = prob[k:k + 6].mean(0)
                if cur in hit:
                    if ons[k, cur] > 0.7:
                        segs.append((k, cur))
                elif hit:
                    r = max(hit, key=lambda r: now[r])
                    if now[r] > now[cur]:
                        segs.append((k, r))
            bounds = [a for a, _ in segs] + [u]
            notes += [(float(times[a]), float(times[min(b, T - 1)]), lo + int(r))
                      for (a, r), b in zip(segs, bounds[1:]) if b - a >= 6]  # drop blips under ~70 ms
        t = u
    return _unspike(_fold_down(notes))


def _unspike(notes):
    """Move a note (or run of repeats) by octaves toward its neighbours when it leaps 9+ semitones away from both
    in the same direction. Such out-and-back leaps come from the backing or an overtone, not the melody."""
    gap, pause = 9, 0.25  # semitones; seconds of silence that mark a deliberate leap
    runs = []  # [first, last] index of each run of same-pitch notes
    for i, (_, _, p) in enumerate(notes):
        if runs and notes[runs[-1][1]][2] == p:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    ps = [notes[a][2] for a, _ in runs]
    out = list(notes)
    for j in range(1, len(runs) - 1):
        a, p, b = ps[j - 1], ps[j], ps[j + 1]  # left neighbour is already fixed, so back-to-back spikes resolve
        first, last = runs[j]
        # ponytail: a real out-and-back leap over a sixth is left alone only when a pause separates it.
        near = (notes[first][0] - notes[first - 1][1] < pause) and (notes[last + 1][0] - notes[last][1] < pause)
        if near and (min(p - a, p - b) >= gap or min(a - p, b - p) >= gap):
            ps[j] = min((p + 12 * k for k in range(-3, 4)), key=lambda q: (max(abs(q - a), abs(q - b)), abs(q - p)))
            out[first:last + 1] = [(s, e, ps[j]) for s, e, _ in notes[first:last + 1]]
    return out


def _fold_down(notes):
    """Move a note down an octave while it sits more than a sixth above the local median pitch (an overtone).
    Never moves notes up: low phrase endings are real, and folding them up wrote them an octave too high."""
    # ponytail: a real high note more than a sixth above the surrounding 17 notes gets folded down too.
    ps = [p for _, _, p in notes]
    out = []
    for i, (s, e, p) in enumerate(notes):
        med = float(np.median(ps[max(0, i - 8): i + 9]))
        while p - med > 9:
            p -= 12
        out.append((s, e, p))
    return out


def _fold_beats(beats):
    """Halve or double the beat track until it sits in 60-180 BPM."""
    # ponytail: beat trackers often lock onto half/double time; folding into 60-180 is a heuristic.
    while len(beats) > 2 and np.median(np.diff(beats)) < 60 / 180:
        beats = beats[::2]
    while len(beats) > 1 and np.median(np.diff(beats)) > 1.0:
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
    return beats


def detect_notes(wav, model="piano", lead_wav=None):
    """Return ([(start_s, end_s, midi_pitch)], beat_times_s); beats always come from the full mix in wav."""
    y, sr = librosa.load(str(wav), sr=22050)
    if model == "piano":
        events = _piano_events(librosa.resample(y, orig_sr=sr, target_sr=16000))
    else:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict

        # Stricter than defaults (0.5/0.3/127 ms) to cut ghost notes.
        model_output, _, events = predict(str(lead_wav or wav), ICASSP_2022_MODEL_PATH, onset_threshold=0.6,
                                          frame_threshold=0.4, minimum_note_length=100)
        if model == "melody":
            events = _track_melody(model_output, librosa.load(str(lead_wav), sr=sr)[0], sr)
    beats = _fold_beats(np.asarray(librosa.beat.beat_track(y=y, sr=sr, units="time")[1], dtype=float))
    return sorted((float(s), float(e), int(p)) for s, e, p, *_ in events), beats


def _beat_clock(beats, t_end):
    """Map seconds -> beat position along the detected beats, so rubato doesn't drift off the grid."""
    iv = float(np.median(np.diff(beats)))
    pre = beats[0] - iv * np.arange(int(np.ceil(beats[0] / iv)), 0, -1)
    post = beats[-1] + iv * np.arange(1, int(np.ceil(max(t_end - beats[-1], 0) / iv)) + 2)
    grid = np.concatenate([pre, beats, post])
    idx = np.arange(len(grid), dtype=float)
    return lambda t: float(np.interp(t, grid, idx)), round(60 / iv)


def _to_range(p, lo, hi):
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def _beat_of(b):
    """Beat index a beat position belongs to; onsets up to a 32nd early count as the next beat."""
    return math.floor(b + 1 / 8)


def _grids(onsets):
    """Pick one grid per beat (16ths or 8th-triplets) so a beat never mixes both; triplets must fit clearly better."""
    groups = {}
    for a in onsets:
        groups.setdefault(_beat_of(a), []).append(a)
    err = lambda xs, g: sum(abs(x - round(x * g) / g) for x in xs)
    return {k: 3 if err(xs, 3) < 0.7 * err(xs, GRID) else GRID for k, xs in groups.items()}


def _quantize(notes, to_beats, lo, hi, poly):
    """Seconds -> quarter-length grid, fold into range; monophonic instruments keep the top note."""
    starts = [to_beats(s) for s, _, _ in notes]
    grids = _grids(starts)
    out = []
    for (_, e, p), a in zip(notes, starts):
        g = grids[_beat_of(a)]
        s, b = round(a * g) / g, to_beats(e)
        ge = grids.get(_beat_of(b), g)
        out.append((s, max(round(b * ge) / ge, s + 1 / g), _to_range(p, lo, hi)))
    if poly:
        return out
    # ponytail: "melody = highest note at each onset" is naive; a real melody extractor would be the upgrade.
    top = {}
    for s, e, p in out:
        if s not in top or p > top[s][1]:
            top[s] = (e, p)
    starts = sorted(top)
    line = []
    for i, s in enumerate(starts):
        e, p = top[s]
        if i + 1 < len(starts):
            nxt = starts[i + 1]
            e = nxt if nxt - e <= 0.5 else min(e, nxt)  # close gaps under half a beat instead of writing tiny rests
        line.append((s, e, p))
    return line


def _part(notes, inst, cl, ts, ks, cls=stream.Part):
    """Build one staff; chordify splits overlapping notes into tied chords so the staff has a single voice."""
    raw = stream.Stream()
    for s, e, p in notes:
        n = note.Note(p)
        n.quarterLength = e - s
        raw.insert(s, n)
    part = cls()
    for obj in (inst, cl, ts, ks):
        part.insert(0, obj)
    if notes:
        for el in raw.chordify().flatten().notes:
            if isinstance(el, chord.Chord) and len(el.pitches) == 1:  # a one-pitch chord is written as a note
                n = note.Note(el.pitches[0], duration=el.duration)
                n.tie = el.tie
                part.insert(el.offset, n)
            else:
                part.insert(el.offset, el)
    else:
        part.insert(0, note.Rest(quarterLength=4))
    return part


def build_score(notes, bpm, name="piano", title="Transcription", to_beats=None):
    """Turn [(start_s, end_s, midi)] into a Score; to_beats maps seconds to beats (default: steady bpm)."""
    cls, lo, hi, poly = INSTRUMENTS[name]
    notes = _quantize(notes, to_beats or (lambda t: t * bpm / 60), lo, hi, poly)
    # Pedal makes recorded notes ring into the next ones; notate each note only until the next onset (any staff).
    # ponytail: this also shortens a held melody note under a moving arpeggio; voice separation is the upgrade.
    onsets = sorted({s for s, _, _ in notes})
    nxt = dict(zip(onsets, onsets[1:]))
    notes = sorted({(s, min(e, nxt.get(s, e)), p) for s, e, p in notes})
    probe = stream.Stream([note.Note(p) for _, _, p in notes])
    # ponytail: assumes 4/4 and uses music21's default sharp spelling; the key signature is a best guess.
    ks = probe.analyze("key") if notes else key.Key("C")
    ts = meter.TimeSignature("4/4")

    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    if name == "piano":
        rh = _part([n for n in notes if n[2] >= 60], instrument.Piano(), clef.TrebleClef(),
                   ts, ks, stream.PartStaff)
        lh = _part([n for n in notes if n[2] < 60], instrument.Piano(), clef.BassClef(),
                   meter.TimeSignature("4/4"), key.KeySignature(ks.sharps), stream.PartStaff)
        score.insert(0, rh)
        score.insert(0, lh)
        score.insert(0, layout.StaffGroup([rh, lh], symbol="brace", barTogether=True))
    else:
        c = {"guitar": clef.Treble8vbClef(), "cello": clef.BassClef()}.get(name, clef.TrebleClef())
        part = _part(notes, cls(), c, ts, ks)
        part.atSoundingPitch = True
        score.insert(0, part)
    score.parts[0].insert(0, tempo.MetronomeMark(number=bpm))
    return score


def _tempo_map(notes, beats):
    """Return (seconds -> beats function, bpm) that follows the detected beats; (None, 120) if there are too few."""
    if len(beats) < 2:
        return None, 120
    clock, bpm = _beat_clock(beats, max(e for _, e, _ in notes))
    # Beat trackers place beats a little early or late; shift so most onsets land on the beat.
    # ponytail: median phase assumes most notes start on beats; syncopated music may shift wrongly.
    phase = sorted(clock(s) - round(clock(s)) for s, _, _ in notes)[len(notes) // 2]
    shift = phase if phase <= 0 else phase - 1  # never push early notes below offset 0
    return (lambda t: clock(t) - shift), bpm


def transcribe(src, out_dir, name="piano", bpm=None, model="piano"):
    """Full pipeline; writes score.musicxml and score.mid into out_dir and returns a summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav, stereo, lead = out_dir / "audio.wav", out_dir / "stereo.wav", out_dir / "lead.wav"
    try:
        extract_audio(src, wav)
        if model == "melody":
            extract_audio(src, stereo, channels=2, rate=44100)  # Demucs expects 44.1 kHz stereo
            isolate_lead(stereo, lead)
        notes, beats = detect_notes(wav, model, lead if model == "melody" else None)
    finally:
        for f in (wav, stereo, lead):
            f.unlink(missing_ok=True)
    if not notes:
        raise ValueError("No notes were detected in this file.")
    to_beats = None
    if bpm is None:  # follow the performer's tempo; a user-typed BPM means a steady grid instead
        to_beats, bpm = _tempo_map(notes, beats)
    score = build_score(notes, bpm, name, Path(src).stem, to_beats)
    score.write("midi", fp=out_dir / "score.mid")  # MIDI stays at sounding pitch
    if name != "piano":  # Bb/Eb instruments are written transposed
        score = score.toWrittenPitch()
    score.write("musicxml", fp=out_dir / "score.musicxml")
    return {"notes": len(notes), "bpm": bpm, "instrument": name}
