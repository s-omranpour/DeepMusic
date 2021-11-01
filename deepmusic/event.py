from typing import List
import numpy as np
from .conf import CHORDS, MusicConfig


class MusicEvent:
    """
        bar : 0~...
        beat: 0~len(beats)-1  (0 = stating of a bar) 
    """
    def __init__(self, bar : int = 0, beat : int = 0):
        self.bar = bar
        self.beat = beat

    def set_metric_attributes(self, bar : int = None, beat : int = None):
        if bar is not None:
            self.bar = bar
        if beat is not None:
            self.beat = beat

    def __eq__(self, o: object):
        if isinstance(o, MusicEvent):
            return self.bar == o.bar and self.beat == o.beat

    def to_tokens(self):
        return ['Beat_' + str(self.beat)]

    def update_metric_attributes_with_config(self, new_config : MusicConfig, old_config : MusicConfig):
        beat = int(np.round(self.beat * new_config.n_bar_steps / old_config.n_bar_steps))
        self.beat = beat


class TempoEvent(MusicEvent):
    """
        tempo : 0~len(tempo_bins)-1  (index of value in tempo bins)
    """
    def __init__(self, bar : int = 0, beat : int = 0, tempo : int = 0):
        super().__init__(bar, beat)
        self.set_tempo(tempo)

    @staticmethod
    def from_tokens(tokens : List[str], bar : int):
        assert tokens[0].startswith('Beat')
        assert tokens[1].startswith('Tempo')
        beat = int(tokens[0][5:])
        tempo = int(tokens[1][6:])
        return TempoEvent(bar, beat, tempo)

    def __eq__(self, o: object):
        if isinstance(o, TempoEvent):
            return super().__eq__(o) and self.tempo == o.tempo

    def __repr__(self):
        return f'TempoEvent(bar={self.bar}, beat={self.beat}, tempo={self.tempo})'

    def __hash__(self):
        return hash((self.bar, self.beat, self.tempo))

    def set_tempo(self, tempo : int = 0):
        self.tempo = tempo

    def to_tokens(self, include_metrics=False):
        res = []
        if include_metrics:
            res += super().to_tokens()
        res += ['Tempo_' + str(self.tempo)]
        return res


class ChordEvent(MusicEvent):
    def __init__(self, bar : int = 0, beat : int = 0, chord : int = 0):
        super().__init__(bar, beat)
        self.set_chord(chord)
    
    @staticmethod
    def from_tokens(tokens : List[str], bar : int):
        assert tokens[0].startswith('Beat')
        assert tokens[1].startswith('Chord')
        beat = int(tokens[0][5:])
        chord = int(tokens[1][6:])
        return ChordEvent(bar, beat, chord)

    def __eq__(self, o: object):
        if isinstance(o, ChordEvent):
            return super().__eq__(o) and self.chord == o.chord

    def __repr__(self):
        return f'ChordEvent(bar={self.bar}, beat={self.beat}, chord={self.chord_name})'

    def __hash__(self):
        return hash((self.bar, self.beat, self.chord))

    def set_chord(self, chord : int = 0):
        self.chord = chord
        self.chord_name =  CHORDS[self.chord - 1]

    def to_tokens(self, include_metrics=False):
        res = []
        if include_metrics:
            res += super(ChordEvent, self).to_tokens()
        res += ['Chord_' + str(self.chord)]
        return res
        

class NoteEvent(MusicEvent):

    """
        bar : 0~...
        beat: 0~len(beats)-1  (0 = stating of a bar) 
        pitch : 0~128 (0 to 127 show midi pitch )
        duration : 1~len(beats) ( show duration in number of subbeats - 1)
        velocity : 0~len(velocity_bins)-1 (0,... show index of the value in velocity bins)
    """

    def __init__(self,
        bar : int = 0, 
        beat : int = 0, 
        pitch : int = 0,
        duration : int = 0,
        velocity : int = 0):

        super().__init__(bar, beat)
        self.pitch = pitch
        self.duration = duration
        self.velocity = velocity

    @staticmethod
    def from_tuple(cp):
        return NoteEvent(*cp)

    @staticmethod
    def from_tokens(tokens : List, bar : int):
        assert tokens[0].startswith('Beat')
        beat = int(tokens[0][5:])
        values = []
        for idx, prefix in enumerate(['NotePitch_', 'NoteDuration_']):
            assert tokens[idx+1].startswith(prefix)
            values += [int(tokens[idx+1][len(prefix):])]
        if tokens[3].startswith('NoteVelocity_'):
            values += [int(tokens[3][13:])]
        else:
            values += [None]       ## in melodies that have no velocity token, we use a default value of None
        return NoteEvent(bar, beat, *values)

    def set_attributes(self, pitch : int = None, duration : int = None, velocity : int = None):
        if pitch is not None:
            self.pitch = pitch
        if duration is not None:
            self.duration = duration
        if velocity is not None:
            self.velocity = velocity

    def __repr__(self):
        return f"NoteEvent(bar={self.bar}, beat={self.beat}, pitch={self.pitch}, duration={self.duration}, velocity={self.velocity})"

    def __eq__(self, other):
        if isinstance(other, MusicEvent):
            return super(NoteEvent, self).__eq__(other) and\
                self.pitch == other.pitch and\
                    self.duration == other.duration and\
                        self.velocity == other.velocity
        return False

    def __hash__(self):
        return hash((self.bar, self.beat, self.pitch, self.duration, self.velocity))

    def to_tuple(self):
        return [self.bar, self.beat, self.pitch, self.duration, self.velocity]

    def to_tokens(self, include_metrics=False, include_velocity=True):
        res = []
        if include_metrics:
            res += super().to_token()
        res += [
            'NotePitch_' + str(self.pitch), 
            'NoteDuration_' + str(self.duration)
        ]
        if include_velocity and self.velocity is not None:
            res += ['NoteVelocity_' + str(self.velocity)]
        return res

    def update_note_with_config(self, new_config : MusicConfig, old_config : MusicConfig):
        self.update_metric_attributes_with_config(new_config, old_config)
        self.duration = int(np.round(self.duration * new_config.n_bar_steps / old_config.n_bar_steps))
        if self.velocity is not None:
            self.velocity = np.argmin(np.abs(new_config.velocity_bins - old_config.velocity_bins[self.velocity]))
