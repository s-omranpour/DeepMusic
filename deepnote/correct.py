from copy import deepcopy
import numpy as np

from deepnote.repr import MusicRepr
from deepnote.modules import Note, Metric

class Corrector:
    def __init__(self):
        self.major_scale = [1,0,1,0,1,1,0,1,0,1,0,1]
        self.minor_scale = [1,0,1,1,0,1,0,1,1,0,1,0]
        self.dominant_scale = [1,0,1,0,1,1,0,1,0,1,1,0]
        self.diminished_scale = [1,0,1,1,0,1,1,0,1,1,0,1]
        self.half_diminished_scale = [1,0,1,1,0,1,1,0,1,0,1,0]
        self.augmented_scale = [1,0,0,1,1,0,0,1,1,0,0,1]
        self.scale_map = {
            'M' : self.major_scale,
            'M7' : self.major_scale,
            'sus2' : self.major_scale,
            'sus4' : self.major_scale,
            'm' : self.minor_scale,
            'm7': self.minor_scale,
            'o' : self.diminished_scale,
            'o7': self.diminished_scale,
            '/o7': self.half_diminished_scale,
            '+' : self.augmented_scale,
            '7' : self.dominant_scale
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

    def make_whole_scale(self, chord):
        root, type = chord.split('_')
        root = self.pitch_map[root]
        scale = self.scale_map[type]
        mask = scale[:root] + scale*11
        mask = np.array(mask[:128])
        return np.where(mask > 0)[0]


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
