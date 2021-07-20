from copy import deepcopy
from typing import List
import numpy as np

from deepnote.repr import MusicRepr
from deepnote.modules import Note, Metric

class Scale:
    def __init__(self):
        self.major = [1,0,1,0,1,1,0,1,0,1,0,1]
        self.minor = [1,0,1,1,0,1,0,1,1,0,1,0]
        self.dominant = [1,0,1,0,1,1,0,1,0,1,1,0]
        self.diminished = [1,0,1,1,0,1,1,0,1,1,0,1]
        self.half_diminished = [1,0,1,1,0,1,1,0,1,0,1,0]
        self.augmented = [1,0,0,1,1,0,0,1,1,0,0,1]
        self.type_map = {
            'M' : self.major,
            'M7' : self.major,
            'sus2' : self.major,
            'sus4' : self.major,
            'm' : self.minor,
            'm7': self.minor,
            'o' : self.diminished,
            'o7': self.diminished,
            '/o7': self.half_diminished,
            '+' : self.augmented,
            '7' : self.dominant
        }
        self.pitch_map = {
            'C' : 0,
            'C#' : 1,
            'D' : 2,
            'D#' : 3,
            'E' : 4,
            'F' : 5,
            'F#' : 6,
            'G' : 7,
            'G#' : 8,
            'A' : 9,
            'A#' : 10,
            'B' : 11
        }

    def make_mask(self, chord):
        root, type = chord.split('_')
        root = self.pitch_map[root]
        scale = self.type_map[type]
        mask = scale[-root:] + scale*11
        return np.array(mask[:128])

    def pitch_in_scale(self, pianoroll : np.array, chord : str):
        assert pianoroll.shape[0] == 128
        mask = self.make_mask(chord)
        mask_repeat = np.array([mask]*pianoroll.shape[1]).T
        return np.sum(mask_repeat * pianoroll)

    def correct_seq_pitches(self, seq):
        seq = deepcopy(seq)
        res = []
        prev_chord = None
        scale = None
        for e in seq.events:
            if isinstance(e, Metric) and e.chord is not None:
                prev_chord = e.chord
                scale = self.make_whole_scale(prev_chord)
            elif isinstance(e, Note) and prev_chord is not None and scale is not None:
                e.pitch = scale[np.argmin(np.abs(e.pitch - scale))]
            res += [e]
        return MusicRepr(res, tick_resol=seq.tick_resol, unit=seq.unit)
