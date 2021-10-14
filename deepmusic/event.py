from operator import pos
from typing import List, Dict
import numpy as np

from .const import Constants, CHORDS


class MusicEvent:

    """
    *** extended compund word representation ***
        bar : 0~...
        beat: 0~len(beats)-1  (0 = stating of a bar) 
        tempo : 0~len(tempo_bins)  (0 = IGNORE, 1,... show index of value in tempo bins)
        chord : 0~132 (0 = IGNORE, 1 to 132 show index of name in chords)
        note_pitch : 0~128 (0 = IGNORE, 1 to 128 show midi pitch + 1)
        note_duration : 0~len(beats)-1 (0 = IGNORE, 1,... show duration in number of subbeats)
        note_velocity : 0~len(velocity_bins) (0 = IGNORE, 1,... show index of the value in velocity bins)
    """

    def __init__(self,
        bar : int = 0, 
        beat : int = 0, 
        tempo : int = 0, 
        chord : int = 0,
        note_pitch : int = 0,
        note_duration : int = 0,
        note_velocity : int = 0, 
        const : Constants = None):

        self.const = Constants() if const is None else const
        assert 0 <= bar
        assert 0 <= beat < self.const.n_bar_steps
        assert 0 <= tempo < self.const.num_velocity_bins
        assert 0 <= chord < len(CHORDS)
        assert 0 <= note_pitch < 128
        assert 0 <= note_duration < self.const.n_bar_steps
        assert 0 <= note_velocity < self.const.num_velocity_bins

        self.bar = bar
        self.beat = beat
        self.tempo = tempo
        self.chord = chord
        self.note_pitch = note_pitch
        self.note_duration = note_duration
        self.note_velocity = note_velocity

    @staticmethod
    def from_tuple(cp, const : Constants = None):
        return MusicEvent(*cp, const=const)

    @staticmethod
    def from_tokens(tokens : List, bar : int, const : Constants = None):
        beat = int(tokens[0][4:]) if tokens[1].startswith('Beat') else 0
        values = [0 for _ in range(5)]
        idx = 1
        for i, attr in enumerate(['Tempo', 'Chord', 'NotePitch', 'NoteDuration', 'NoteVelocity']):
            if idx >= len(tokens):
                break
            if tokens[idx].startswith(attr):
                values[i] = int(tokens[idx][len(attr):]) + 1 ## indices start from 1 (0 = ignore)
                idx += 1
            else:
                values[i] = 0
        return MusicEvent(bar, beat, *values)

    def set_metric_attributes(self, bar : int = None, beat : int = None, tempo : int = None, chord : int = None):
        if bar is not None:
            self.bar = bar
        if beat is not None:
            self.beat = beat
        if tempo is None:
            self.tempo = tempo
        if chord is not None:
            self.chord = chord

    def set_note_attributes(self, note_pitch, note_duration, note_velocity):
        self.note_pitch = note_pitch
        self.note_duration = note_duration
        self.note_velocity = note_velocity

    def change_const(self, const: Constants):
        if const is not None and self.const != const:
            self.beat = int(np.round(self.beat * const.n_bar_steps / self.const.n_bar_steps))
            if self.tempo > 0:
                self.tempo = np.argmin(np.abs(const.tempo_bins - self.const.tempo_bins[self.tempo]))
            self.note_duration = int(np.round(self.note_duration * const.n_bar_steps / self.const.n_bar_steps))
            if self.note_velocity > 0:
                self.note_velocity = np.argmin(np.abs(const.velocity_bins - self.const.velocity_bins[self.note_velocity]))
            self.const = const

    def __repr__(self, pretty=False):
        prop = [f'bar={self.bar}, beat={self.beat}']
        if self.tempo:
            prop += [f'tempo={self.get_actual_tempo() if pretty else self.tempo}']
        if self.chord:
            prop += [f'chord={self.get_actual_chord() if pretty else self.chord}']
        if self.note_pitch:
            prop += [f'note_pitch={self.note_pitch}']
        if self.note_duration:
            prop += [f'note_duration={self.note_duration}']
        if self.note_velocity:
            prop += [f'note_velocity={self.get_actual_velocity() if pretty else self.note_velocity}'] 
        return f"MusicEvent({', '.join(prop)})"

    def __eq__(self, other):
        if isinstance(other, MusicEvent):
            return self.bar == other.bar and\
                self.beat == other.beat and\
                    self.tempo == other.tempo and\
                        self.chord == other.chord and\
                            self.note_pitch == other.note_pitch and\
                                self.note_duration == other.note_duration and\
                                    self.note_velocity == other.note_velocity
        return False

    def __hash__(self):
        return hash((self.bar, self.beat, self.tempo, self.chord, self.note_pitch, self.note_duration, self.note_velocity))

    def get_position_in_ticks(self):
        return self.beat*self.const.step + self.bar*self.const.bar_resol

    def get_actual_tempo(self):
        return self.const.tempo_bins[self.tempo - 1] if self.tempo else None

    def get_actual_chord(self):
        return CHORDS[self.chord - 1] if self.chord else None

    def get_duration_in_ticks(self):
        return self.duration*self.const.step

    def get_actual_velocity(self):
        return self.const.velocity_bins[self.note_velocity - 1]

    def to_tuple(self):
        return [self.bar, self.beat, self.tempo, self.chord, self.note_pitch, self.note_duration, self.note_velocity]

    def to_tokens(self):
        res = []
        if self.beat > 0:
            res += ['Beat' + str(self.beat)]
        else:
            res += ['Bar']
        if self.tempo > 0:
            res += ['Tempo'+str(self.tempo)]
        if self.chord > 0:
            res += ['Chord' + str(self.chord)]
        if self.note_pitch > 0:
            res += [
                'NotePitch' + str(self.note_pitch - 1), ## indices start from 1
                'NoteDuration' + str(self.note_duration - 1),
                'NoteVelocity' + str(self.note_velocity - 1)
            ]
        return res
    
    def is_empty(self):
        return self.note_pitch == 0
