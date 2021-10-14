import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from chorder import Dechorder

from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi.containers import Marker

from .event import MusicEvent
from .const import Constants

def sort_events(events):
    return sorted(events, key=lambda x: (x.bar, x.position, x.note_pitch, x.note_duration, x.note_velocity))

def remove_empty_events(events):
    return list(filter(lambda x: not x.is_empty(), events))

def sort_and_remove_identical_events(events):
    return sort_events(set(events))

def remove_duplicate_metrics_from_remi(remi : str):
    tokens = remi.split()
    res = []
    prev_beat = 0
    prev_tempo = 0
    prev_chord = 0
    for tok in tokens:
        if tok.startswith('Bar'):
            res += [tok]
            prev_beat = 0
        elif tok.startswith('Beat'):
            beat = int(tok[4:])
            if beat != prev_beat:
                res += [tok]
                prev_beat = beat
        elif tok.startswith('Tempo'):
            tempo = int(tok[5:])
            if tempo != prev_tempo:
                res += [tok]
                prev_tempo = tempo
        elif tok.startswith('Chord'):
            chord = int(tok[5:])
            if chord != prev_chord:
                res += [tok]
                prev_chord = chord
        else:
            res += [tok]
    return res

def flatten(ls):
    res = []
    for l in ls:
        res += l
    return res

def classify_events_by_attr(events, attrs):
    res = {}
    for e in events:
        val = tuple([e.__getattribute__(attr) for attr in attrs])
        if len(val) == 1:
            val = val[0]
        if val not in res:
            res[val] = []
        res[val] += [e]
    return res


def analyze_midi(midi):
    const = Constants()
    midi = deepcopy(midi)
    chords = Dechorder.dechord(midi)
    markers = []
    prev = None
    for cidx, chord in enumerate(chords):
        if chord.is_complete():
            chord_text = 'Chord_' + const.pitch_classes[chord.root_pc] + '_' + chord.quality
            if chord_text != prev:
                markers.append(Marker(time=int(cidx*midi.ticks_per_beat), text=chord_text))
                prev = chord_text
    midi.markers = markers
    return midi

def edit_time(time, offset, step):
    time = max(0, time - offset)
    return int(np.round(time/step)*step)

def quantize_midi(
    midi, 
    unit=12,
    min_tempo=30,
    max_tempo=300,
    num_tempo_bins=30, 
    num_velocity_bins=30):

    midi = deepcopy(midi)
    ## load notes
    instr_notes = []
    for instr in midi.instruments:
        notes = []
        for note in instr.notes:
            notes += [note]
        instr_notes += [sorted(notes, key=lambda x: (x.start, x.pitch))]
    
    offset = min([notes[0].start for notes in instr_notes])
    tick_resol = midi.ticks_per_beat
    const = Constants(unit, tick_resol, min_tempo, max_tempo, num_tempo_bins, num_velocity_bins)

    for i,notes in enumerate(instr_notes):
        for note in notes:
            note.start = edit_time(note.start, offset, const.step)
            note.end = edit_time(note.end, offset, const.step)
            note.end = min(note.end, edit_time(note.start + const.bar_resol, offset, const.step))
            note.velocity = const.velocity_bins[
                np.argmin(abs(const.velocity_bins - note.velocity))
            ]
        midi.instruments[i].notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # load chords
    chords = []
    for marker in midi.markers:
        if marker.text.startswith('Chord'):
            marker.time = edit_time(marker.time, offset, const.step)
            chords.append(marker)
    chords.sort(key=lambda x: x.time)
    midi.markers = chords
    

    # load tempos
    tempos = midi.tempo_changes
    for tempo in tempos:
        tempo.time = edit_time(tempo.time, offset, const.step)
        tempo.tempo = const.tempo_bins[
            np.argmin(abs(const.tempo_bins - tempo.tempo))
        ]
    tempos.sort(key=lambda x: x.time)
    midi.tempo_changes = tempos

    midi.max_tick = edit_time(midi.max_tick, offset, const.step)
    return midi

def process_midi(
    file_path, 
    save_path, 
    unit=12,
    min_tempo=30,
    max_tempo=300,
    num_tempo_bins=30, 
    num_velocity_bins=30):

    try:
        midi = mid_parser.MidiFile(file_path)
        times = midi.time_signature_changes
        if len(times):
            t = times[0]
            if t.numerator != 4 or t.denominator != 4:
                return
        midi = analyze_midi(midi)
        midi = quantize_midi(midi, unit, min_tempo, max_tempo, num_tempo_bins, num_velocity_bins)
        midi.dump(save_path + file_path.split('/')[-1])
    except Exception as e:
        print(file_path, 'caused error', e)
