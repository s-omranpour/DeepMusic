from typing import List
import numpy as np
import itertools

from deepmusic.event import NoteEvent

PITCHES = np.arange(0, 128)
PIRCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_QUALITIES = ['M', 'm', 'o', '+', '7', 'M7', 'm7', 'o7', '/o7', 'sus2', 'sus4']
CHORDS = ['_'.join(e) for e in itertools.product(PIRCH_CLASSES, CHORD_QUALITIES)]
INSTRUMENT_PROGRAMS = np.arange(0, 256)
INSTRUMENT_FAMILIES = [
    'piano',
    'percussion',
    'organ',
    'guitar',
    'bass',
    'strings',
    'ensemble',
    'brass',
    'reed',
    'pipe',
    'synth-lead',
    'synth-pad',
    'synth-effects',
    'ethnic',
    'percussive',
    'sound-effects',
    'drums'
]

class MusicConfig:
    def __init__(
        self, 
        unit=12, 
        tick_resol=120, 
        min_tempo=30,
        max_tempo=300,
        num_tempo_bins=30, 
        num_velocity_bins=30):

        ## time
        self.unit = unit
        self.n_bar_steps = 4 * unit

        ## resolution
        self.tick_resol = tick_resol
        self.step = tick_resol // unit
        self.bar_resol = 4*tick_resol

        ## position
        self.position_bins = np.arange(0, 4*unit)

        ## tempo
        self.min_tempo = min_tempo
        self.max_tempo = max_tempo
        self.num_tempo_bins = num_tempo_bins
        self.tempo_bins = np.linspace(min_tempo, max_tempo, num_tempo_bins, dtype=int)

        ## duration
        self.duration_bins = np.arange(1, 4*unit+1)

        ## velocity
        self.num_velocity_bins = num_velocity_bins
        self.velocity_bins = np.linspace(1, 127, num_velocity_bins, dtype=int)

        ## tokens
        self.special_tokens = ['BOS', 'EOS', 'MASK']
        self.position_tokens = ['Bar'] + ['Beat_'+str(p) for p in self.position_bins]
        self.tempo_tokens = ['Tempo_'+str(t) for t in self.tempo_bins]
        self.chord_tokens = ['Chord_'+str(c) for c in CHORDS]
        self.pitch_tokens = ['NotePitch_'+str(i) for i in PITCHES]
        self.duration_tokens = ['NoteDuration_'+str(d) for d in self.duration_bins]
        self.velocity_tokens = ['NoteVelocity_'+str(v)for v in self.velocity_bins]
        self.program_tokens = ['NoteInstrument_'+str(inst) for inst in INSTRUMENT_PROGRAMS]
        self.all_tokens =  self.special_tokens + self.position_tokens + self.tempo_tokens +\
            self.chord_tokens + self.pitch_tokens + self.duration_tokens + self.velocity_tokens + self.program_tokens

    def update_resolution(self, tick_resol):
        self.tick_resol = tick_resol
        self.step = tick_resol // self.unit
        self.bar_resol = 4*tick_resol

    def __repr__(self):
        return f'MusicConfig(unit={self.unit}, tick_resol={self.tick_resol}, min_tempo={self.min_tempo}, max_tempo={self.max_tempo}, num_tempo_bins={self.num_tempo_bins}, num_velocity_bins={self.num_velocity_bins})'

    def __eq__(self, o: object):
        if isinstance(o, MusicConfig):
            return self.unit == o.unit and\
                self.tick_resol == o.tick_resol and \
                    self.min_tempo == o.min_tempo and \
                        self.max_tempo == o.max_tempo and \
                            self.num_tempo_bins == o.num_tempo_bins and \
                                self.num_velocity_bins == o.num_velocity_bins
        return False

    def encode(self, tokens : List):
        return [self.all_tokens.index(tok) for tok in tokens]

    def decode(self, indices : List):
        return [self.all_tokens[idx] for idx in indices]