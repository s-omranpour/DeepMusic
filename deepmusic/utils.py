from typing import List
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from chorder import Dechorder

from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi.containers import Marker, Note

from .event import MusicEvent, NoteEvent, TempoEvent, ChordEvent
from .conf import CHORDS, PITCH_CLASSES, MusicConfig


def validate_tempo(config : MusicConfig, tempo : TempoEvent):
    return (0 <= tempo.bar) and (0 <= tempo.beat < config.n_bar_steps) and (0 <= tempo.tempo < config.num_tempo_bins)

def validate_chord(config : MusicConfig, chord : ChordEvent):
    return (0 <= chord.bar) and (0 <= chord.beat < config.n_bar_steps) and (0 <= chord.chord < len(CHORDS)) and (chord.chord_name in CHORDS)

def validate_note(config : MusicConfig, note : NoteEvent):
    return (0 <= note.pitch < 128) and (0 < note.duration <= config.n_bar_steps) and (0 <= note.velocity < config.num_velocity_bins)

def sort_events(events : List[MusicEvent]):
    return sorted(events, key=lambda x: (x.bar, x.beat))

def sort_and_remove_identical_events(events : List[MusicEvent]):
    return sort_events(set(events))

def remove_duplicate_beats_from_tokens(tokens : List):
    res = []
    prev_beat = ''
    for tok in tokens:
        if tok.startswith('Beat'):
            if tok != prev_beat:
                res += [tok]
                prev_beat = tok
        else:
            res += [tok]
    return res

def flatten(ls):
    res = []
    for l in ls:
        res += l
    return res

def organize_events_by_attr(events : List[MusicEvent], attrs):
    res = {}
    for e in events:
        val = tuple([e.__getattribute__(attr) for attr in attrs])
        if len(val) == 1:
            val = val[0]
        if val not in res:
            res[val] = []
        res[val] += [deepcopy(e)]
    return res


def analyze_midi(midi):
    midi = deepcopy(midi)
    chords = Dechorder.dechord(midi)
    markers = []
    prev = None
    for cidx, chord in enumerate(chords):
        if chord.is_complete():
            chord_text = 'Chord_' + PITCH_CLASSES[chord.root_pc] + '_' + chord.quality
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
    config : MusicConfig = None,
    unit : int = 12,
    min_tempo : int = 30,
    max_tempo: int = 300,
    num_tempo_bins : int = 30, 
    num_velocity_bins : int = 30):

    assert len(midi.time_signature_changes) == 1
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
    config = config if config is not None else MusicConfig(unit, (4,4), tick_resol, min_tempo, max_tempo, num_tempo_bins, num_velocity_bins)

    for i,notes in enumerate(instr_notes):
        for note in notes:
            note.start = edit_time(note.start, offset, config.step)
            note.end = edit_time(note.end, offset, config.step)
            note.end = min(note.end, edit_time(note.start + config.bar_resol, offset, config.step))
            note.velocity = config.velocity_bins[
                np.argmin(abs(config.velocity_bins - note.velocity))
            ]
        midi.instruments[i].notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # load chords
    chords = []
    for marker in midi.markers:
        if marker.text.startswith('Chord'):
            marker.time = edit_time(marker.time, offset, config.step)
            chords.append(marker)
    chords.sort(key=lambda x: x.time)
    midi.markers = chords
    
    # load tempos
    tempos = midi.tempo_changes
    for tempo in tempos:
        tempo.time = edit_time(tempo.time, offset, config.step)
        tempo.tempo = config.tempo_bins[
            np.argmin(abs(config.tempo_bins - tempo.tempo))
        ]
    tempos.sort(key=lambda x: x.time)
    midi.tempo_changes = tempos

    midi.max_tick = edit_time(midi.max_tick, offset, config.step)
    return midi

def process_midi(
    file_path : str, 
    save_path : str, 
    config : MusicConfig = None,
    unit : int = 12,
    min_tempo : int = 30,
    max_tempo: int = 300,
    num_tempo_bins : int = 30, 
    num_velocity_bins : int = 30):

    try:
        midi = mid_parser.MidiFile(file_path)
        if len(midi.time_signature_changes) > 1:
            return
        midi = analyze_midi(midi)
        midi = quantize_midi(midi, config, unit, min_tempo, max_tempo, num_tempo_bins, num_velocity_bins)
        midi.dump(save_path + file_path.split('/')[-1])
    except Exception as e:
        print(file_path, 'caused error', e)
