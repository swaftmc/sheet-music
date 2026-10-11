"""Self-check: score building for every instrument, the melody tracker, then full pipeline on a synthetic C-major scale."""
import tempfile
import wave
from pathlib import Path

import numpy as np
from music21 import converter

from transcribe import INSTRUMENTS, _beat_clock, _fold_down, _track_melody, _unspike, build_score, transcribe

SCALE = [60, 62, 64, 65, 67, 69, 71, 72]
LEAD = [(1, 1.0), (2, 0.5), (3, 0.3)]  # harmonic profile of a plain lead tone


def tones(parts, n):
    """Audio of (start_frame, end_frame, midi, amp, harmonics) tones, at 256 samples per frame."""
    y = np.zeros(n * 256)
    for a, b, p, amp, prof in parts:
        t = np.arange((b - a) * 256) / 22050
        y[a * 256:b * 256] += amp * sum(w * np.sin(2 * np.pi * 440 * 2 ** ((p - 69) / 12) * k * t) for k, w in prof)
    return y


# 1. Score building: a C-major chord plus a bass note, at 120 BPM (0.5 s = 1 quarter).
notes = [(0.0, 1.0, 48), (0.0, 1.0, 60), (0.0, 1.0, 64), (0.0, 1.0, 67), (1.0, 1.5, 72)]
for name, (_, lo, hi, poly) in INSTRUMENTS.items():
    s = build_score(notes, 120, name)
    with tempfile.TemporaryDirectory() as d:
        s.write("musicxml", fp=Path(d) / "x.musicxml")
        back = converter.parse(Path(d) / "x.musicxml")
    sounding = [p.midi for p in back.toSoundingPitch().flatten().pitches]
    assert sounding, name
    assert all(lo <= p <= hi for p in sounding), (name, sounding)
    if name == "piano":
        assert len(back.parts) == 2 and 48 in sounding and 72 in sounding, sounding
    if not poly:
        assert not back.flatten().getElementsByClass("Chord"), name
print("score building ok for", len(INSTRUMENTS), "instruments")

# 2. Pedal overlap: an arpeggio whose notes each ring 2 s must not turn into stacked chords.
arp = [(i * 0.25, i * 0.25 + 2.0, p) for i, p in enumerate([60, 64, 67, 72])]
flat = build_score(arp, 120, "piano").parts[0].flatten()
assert not flat.getElementsByClass("Chord"), [c.pitches for c in flat.getElementsByClass("Chord")]
# 3. Triplets snap to the triplet grid instead of 16ths (1/3 s at 60 BPM = 1/3 quarter).
trip = build_score([(i / 3, (i + 1) / 3, 64) for i in range(6)], 60, "piano").parts[0].flatten().notes
assert [round(float(n.offset), 3) for n in trip] == [round(i / 3, 3) for i in range(6)], [n.offset for n in trip]
# 4. Rubato: beats that slow from 0.5 s to 0.7 s apart still put one note on each beat.
beats = np.cumsum(np.linspace(0.5, 0.7, 12))
clock, _ = _beat_clock(beats, beats[-1])
rub = build_score([(a, b, 64) for a, b in zip(beats, beats[1:])], 60, "piano", to_beats=clock).parts[0].flatten().notes
offs = [float(n.offset) for n in rub]
assert offs == [offs[0] + i for i in range(11)], offs
assert all(n.quarterLength == 1 or n.tie for n in rub), [(n.quarterLength, n.tie) for n in rub]
# 5. Pedal trim works across staves: bass G#3 onset cuts the ringing treble E4.
tr = build_score([(0, 2, 64), (0.5, 2, 56)], 120, "piano")
assert tr.parts[0].flatten().notes[0].quarterLength == 1, tr.parts[0].flatten().notes[0].quarterLength
# 6. One grid per beat: 16ths in beat 1 and triplets in beat 2 each stay exact, with no rests between.
mix = [(i * 0.25, (i + 1) * 0.25, 67) for i in range(4)] + [(1 + i / 3, 1 + (i + 1) / 3, 69) for i in range(3)]
got = [(round(float(n.offset), 3), n.quarterLength) for n in build_score(mix, 60, "violin").parts[0].flatten().notesAndRests]
want = [(0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25), (1.0, 1 / 3), (1.333, 1 / 3), (1.667, 1 / 3)]
assert [o for o, _ in got[:7]] == [o for o, _ in want] and all(abs(float(a) - b) < 1e-6 for (_, a), (_, b) in zip(got, want)), got

# 7. Melody tracker on a hand-built note-probability grid (~86 frames/s): lead E5, D5, D5 (new bow stroke),
# a real pause, then G5 with a 3-frame flicker to G#5; a quieter pad on E4 sounds the whole time.
T = 320
note_p, onset_p = np.full((T, 88), 0.01), np.zeros((T, 88))
note_p[:, 64 - 21] = 0.5
for a, b, p in [(0, 60, 76), (60, 120, 74), (120, 180, 74), (260, 320, 79)]:
    note_p[a:b, p - 21] = 0.8
    onset_p[a, p - 21] = 0.9
note_p[280:283, 79 - 21], note_p[280:283, 80 - 21] = 0.3, 0.9
y = tones([(0, 60, 76, 0.3, LEAD), (60, 180, 74, 0.3, LEAD), (260, T, 79, 0.3, LEAD), (0, T, 64, 0.02, [(1, 1.0)])], T)
mel = _track_melody({"note": note_p, "onset": onset_p}, y, 22050)
assert [p for *_, p in mel] == [76, 74, 74, 79], mel
assert mel[3][0] - mel[2][1] > 0.5, mel  # the pause stays a rest instead of following the pad
# 8. A low A3 whose first overtone outweighs its fundamental (a violin's G string) is written A3, even though
# Basic Pitch rates the overtone A4 higher; the harmonic sum hears A3's odd harmonics that A4 cannot explain.
note_p = np.full((172, 88), 0.01)
note_p[:, 69 - 21], note_p[:, 57 - 21] = 0.75, 0.6
y = tones([(0, 172, 57, 0.2, [(1, 0.15), (2, 1.0), (3, 0.5), (4, 0.3)])], 172)
mel = _track_melody({"note": note_p, "onset": np.zeros((172, 88))}, y, 22050)
assert [p for *_, p in mel] == [57], mel
# 9. No hanging note: a sustained A3 drone's steady probability would keep the tracker on it through a fast
# A4-E5-A4-C5 figure; each attack must start a new note instead.
T = 172
note_p, onset_p = np.full((T, 88), 0.01), np.zeros((T, 88))
note_p[:, 57 - 21] = 0.7
fig = [(40 + 24 * i, 64 + 24 * i, p) for i, p in enumerate([69, 76, 69, 72])]
for a, b, p in fig:
    note_p[a:b, p - 21] = 0.8
    onset_p[a, p - 21] = 0.9
y = tones([(a, b, p, 0.3, LEAD) for a, b, p in fig] + [(0, T, 57, 0.08, [(1, 1.0), (2, 0.5)])], T)
mel = _track_melody({"note": note_p, "onset": onset_p}, y, 22050)
assert [p for *_, p in mel if p != 57] == [69, 76, 69, 72], mel
assert max(e - s for s, e, _ in mel) < 1.0, mel
# 10. Octave fold only goes down: an overtone jump folds back, a low phrase ending stays low.
assert _fold_down([(0, 1, 76), (1, 2, 88), (2, 3, 74)]) == [(0, 1, 76), (1, 2, 76), (2, 3, 74)]
assert _fold_down([(0, 1, 74), (1, 2, 76), (2, 3, 57), (3, 4, 74)])[2] == (2, 3, 57)
# 11. Zigzag fix: stray A3s between A4/E5 notes move up an octave; a low-med-low phrase ending stays low, and an
# out-and-back leap separated by pauses is left alone.
zig = [(0, .3, 69), (.3, .5, 57), (.5, .8, 76), (.8, 1, 57), (1, 1.3, 69)]
assert [p for *_, p in _unspike(zig)] == [69, 69, 76, 69, 69], _unspike(zig)
end = [(0, .5, 72), (.5, 1, 71), (1, 1.5, 59), (1.5, 2, 64), (2, 2.5, 59)]
assert _unspike(end) == end, _unspike(end)
apart = [(0, .5, 76), (1, 1.5, 57), (2, 2.5, 76)]
assert _unspike(apart) == apart, _unspike(apart)
# 12. Piano model download: a fetch lands at the checkpoint path; a failed one raises a clear error and leaves no
# partial file that a later run would mistake for the model.
import transcribe as tmod
with tempfile.TemporaryDirectory() as d:
    src_file = Path(d) / "src.bin"
    src_file.write_bytes(b"weights")
    tmod.PIANO_CKPT, tmod.PIANO_URL = Path(d) / "models" / "m.pth", src_file.as_uri()
    tmod._download_piano_checkpoint()
    assert tmod.PIANO_CKPT.read_bytes() == b"weights"
    tmod.PIANO_CKPT.unlink()
    tmod.PIANO_URL = (Path(d) / "missing.bin").as_uri()
    try:
        tmod._download_piano_checkpoint()
        raise AssertionError("a failed download must raise")
    except ValueError as e:
        assert "piano model" in str(e), e
    assert not list((Path(d) / "models").iterdir()), "no .part or .pth may be left behind"
print("notation, melody tracker and model download checks ok")

# 13. End to end: synthesize a scale (0.5 s per note) and transcribe it.
sr = 22050
t = np.arange(int(sr * 0.5)) / sr
env = np.minimum(1, np.minimum(t, t[::-1]) * 40)
audio = np.concatenate([np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t) * env for m in SCALE])
with tempfile.TemporaryDirectory() as d:
    wav = Path(d) / "scale.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((audio * 20000).astype("<i2").tobytes())
    info = transcribe(wav, Path(d) / "out", "piano", bpm=120, model="basic-pitch")  # sine tones aren't piano
    got = [p.midi for p in converter.parse(Path(d) / "out" / "score.musicxml").flatten().pitches]
    assert (Path(d) / "out" / "score.mid").stat().st_size > 0
missing = set(SCALE) - set(got)
assert not missing, (missing, got)
print("pipeline ok:", info, "pitches:", got)
