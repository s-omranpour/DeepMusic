from operator import pos
from typing import List, Dict
import numpy as np

from .const import Constants, CHORDS


class MusicEvent:

    """
    *** extended compund word representation ***

        position: 0~len(positions)-1  (0 = stating of a bar) 
        tempo : 0~len(tempo_bins)  (0 = IGNORE, 1,... show index of value in tempo bins)
        chord : 0~132 (0 = IGNORE, 1 to 132 show index of name in chords)
        note_pitch : 0~128 (0 = IGNORE, 1 to 128 show midi pitch + 1)
        note_duration : 0~len(positions)-1 (0 = IGNORE, 1,... show duration in number of subbeats)
        note_velocity : 0~len(velocity_bins) (0 = IGNORE, 1,... show index of the value in velocity bins)
    """

    def __init__(self, 
        position : int = 0, 
        tempo : int = 0, 
        chord : int = 0,
        note_pitch : int = 0,
        note_duration : int = 0,
        note_velocity : int = 0, 
        const : Constants = None):

        self.const = Constants() if const is None else const
        assert 0 <= position < self.const.n_bar_steps
        assert 0 <= tempo < self.const.num_velocity_bins
        assert 0 <= chord < len(CHORDS)
        assert 0 <= note_pitch < 128
        assert 0 <= note_duration < self.const.n_bar_steps
        assert 0 <= note_velocity < self.const.num_velocity_bins

        self.position = position
        self.tempo = tempo
        self.chord = chord
        self.note_pitch = note_pitch
        self.note_duration = note_duration
        self.note_velocity = note_velocity

    @staticmethod
    def from_cp(cp, const : Constants = None):
        return MusicEvent(*cp, const=const)

    @staticmethod
    def from_remi(remi : str, const : Constants = None):
        tokens = remi.split()
        pos = 0 if tokens[0] == 'Bar' else int(tokens[0][11:])
        values = [0 for _ in range(6)]
        idx = 1
        for i, attr in enumerate(['Tempo', 'Chord', 'NotePitch', 'NoteDuration', 'NoteVelocity']):
            if idx >= len(tokens):
                break
            if tokens[idx].startswith(attr):
                values[i] = int(tokens[idx][len(attr):]) + 1
                idx += 1
            else:
                values[i] = 0
        return MusicEvent(pos, *values)

    def set_metric_attributes(self, position, tempo, chord):
        self.position = position
        self.tempo = tempo
        self.chord = chord

    def set_note_attributes(self, note_pitch, note_duration, note_velocity):
        self.note_pitch = note_pitch
        self.note_duration = note_duration
        self.note_velocity = note_velocity

    def change_const(self, const: Constants):
        if const is not None and self.const != const:
            self.position = int(np.round(self.position * const.n_bar_steps / self.const.n_bar_steps))
            if self.tempo > 0:
                self.tempo = np.argmin(np.abs(const.tempo_bins - self.const.tempo_bins[self.tempo]))
            self.note_duration = int(np.round(self.note_duration * const.n_bar_steps / self.const.n_bar_steps))
            if self.note_velocity > 0:
                self.note_velocity = np.argmin(np.abs(const.velocity_bins - self.const.velocity_bins[self.note_velocity]))
            self.const = const

    def __repr__(self, pretty=False):
        cls = 'Bar' if self.position == 0 else 'Beat'
        prop = ['position=' + str(self.position)]
        if self.tempo:
            prop += ["tempo=" + str(self.const.tempo_bins[self.tempo]) if pretty else str(self.tempo)]
        if self.chord:
            prop += ['chord=' + CHORDS[self.chord] if pretty else str(self.chord)]
        if self.note_pitch:
            prop += ["note_pitch=" + str(self.note_pitch)]
        if self.note_duration:
            prop += ['note_duration=' + str(self.note_duration)]
        if self.note_velocity:
            prop += ['note_velocity=' + str(self.const.velocity_bins[self.note_velocity]) if pretty else str(self.note_velocity)] 
        return f"{cls}({', '.join(prop)})"

    def __eq__(self, other):
        if isinstance(other, MusicEvent):
            return self.position == other.position and\
                self.tempo == other.tempo and\
                    self.chord == other.chord and\
                        self.note_pitch == other.note_pitch and\
                            self.note_duration == other.note_duration and\
                                self.note_velocity == other.note_velocity
        return False

    def __hash__(self):
        return hash((self.position, self.tempo, self.chord, self.note_pitch, self.note_duration, self.note_velocity))

    def to_cp(self):
        return [self.position, self.tempo, self.chord, self.note_pitch, self.note_duration, self.note_velocity]

    def to_remi(self):
        res = ['Bar' if self.position == 0 else 'BeatPosition' + str(self.position)]
        if self.tempo > 0:
            res += ['Tempo'+str(self.tempo)]
        if self.chord > 0:
            res += ['Chord' + str(self.chord)]
        if self.note_pitch > 0:
            res += [
                'NotePitch' + str(self.note_pitch - 1),
                'NoteDuration' + str(self.note_duration - 1),
                'NoteVelocity' + str(self.note_velocity - 1)
            ]
        return res
    
    def has_note(self):
        return self.note_pitch > 0

    def has_metric(self):
        return bool(self.tempo + self.chord)
