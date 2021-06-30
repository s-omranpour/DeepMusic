from typing import List
import numpy as np

from .const import Constants


class Metric:
    """
    *** compund word representation ***

        type :  0 = metric
        position: 0~len(positions)-1  (0 means stating of a bar) 
        tempo : 0~len(tempo_bins)  (0 means no change, 1 to 20 show index of tempo in tempo bins)
        chord : 1~132 (0 means no change, 1 to 132 show index of chord in chords)
    """

    def __init__(
        self, 
        position : int = 0, 
        tempo : int = None,
        chord : str = None):
        
        self.position = position
        self.tempo = tempo
        self.chord = chord
        

    def __repr__(self):
        cls = 'Bar' if self.position == 0 else 'Beat'
        prop = ['position='+str(self.position)]
        if self.tempo:
            prop += ["tempo=" + str(self.tempo)]
        if self.chord:
            prop += ['chord='+self.chord]
        return f"{cls}({', '.join(prop)})"

    def to_remi(self):
        res = [Event('BeatPosition', self.position) if self.position > 0 else Event('Bar')]
        if self.tempo:
            res += [Event('BeatTempo', self.tempo)]
        if self.chord:
            res += [Event('BeatChord', self.chord)]
        return res

    def to_cp(self, const : Constants):
        return [
            0, 
            self.position, 
            np.where(const.tempo_bins == self.tempo)[0][0] + 1 if self.tempo else 0,
            const.chords.index(self.chord) + 1 if self.chord else 0,
            0,
            0, 
            0, 
            0
        ]

    @staticmethod
    def from_cp(cp, const : Constants):
        assert len(cp) == 8, "incorrect size"
        assert cp[0] == 0,   "incorrect type"
        assert 0 <= cp[1] < const.unit*4,           "incorrect position"
        assert 0 <= cp[2] <= len(const.tempo_bins), "incorrect tempo index"
        assert 0 <= cp[3] <= len(const.chords),     "incorrect chord index"
        
        return Metric(
            position=cp[1], 
            tempo=const.tempo_bins[cp[2] - 1] if cp[2] > 0 else None, 
            chord=const.chords[cp[3] - 1] if cp[3] > 0 else None
        )

    def __eq__(self, other):
        if isinstance(other, Metric):
            return self.position == other.position and self.tempo == other.tempo and self.chord == other.chord
        return False

    def __hash__(self):
        return hash((self.position, self.tempo, self.chord))



class Note:
    """
    *** compund word representation ***
    
        type :  1 = note
        inst_family: 0~16  (general midi instrument types, 16 = drums) 
        pitch : 0~127  (standard midi pitch)
        duration : 0~len(positions)-1  (duration in time units minus 1)
        velocity : 0~len(vel_bins)-1  (index of velocity in velocity bins)
    """

    def __init__(self, inst_family : str = 'piano', pitch : int = 0, duration : int = 1, velocity : int = 0):
        self.inst_family = inst_family
        self.pitch = pitch
        self.duration = duration
        self.velocity = velocity
    
    def __repr__(self):
        return "Note(inst_family={}, pitch={}, duration={}, velocity={})".format(
            self.inst_family, self.pitch, self.duration, self.velocity
        ) 

    def to_remi(self):
        return [
            Event('NoteInstFamily', self.inst_family),
            Event('NotePitch', self.pitch),
            Event('NoteDuration', self.duration),
            Event('NoteVelocity', self.velocity)
        ]

    def to_cp(self, const : Constants):
        return [
            1, 
            0, 
            0,
            0, 
            const.instruments.index(self.inst_family),
            self.pitch, 
            self.duration - 1,  
            np.where(const.velocity_bins == self.velocity)[0][0]
        ]

    @staticmethod
    def from_cp(cp, const : Constants):
        assert len(cp) == 8, "incorrect size"
        assert cp[0] == 1,   "incorrect type"
        assert 0 <= cp[4] < len(const.instruments), "incorrect instrument family"
        assert 0 <= cp[5] < 128, "incorrect pitch"
        assert 0 <= cp[6] < const.unit*4, "incorrect duration"
        assert 0 <= cp[7] < len(const.velocity_bins), "incorrect velocity"
        return Note(
            inst_family=const.instruments[cp[4]],
            pitch=cp[5], 
            duration=cp[6] + 1, 
            velocity=const.velocity_bins[cp[7]]
        )

    def __eq__(self, other):
        if isinstance(other, Note):
            return self.inst_family == other.inst_family and \
                   self.pitch == other.pitch and \
                   self.duration == other.duration and\
                   self.velocity == other.velocity
        return False

    def __hash__(self):
        return hash((self.inst_family, self.pitch, self.duration, self.velocity))


class Event:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

    def __repr__(self):
        res = 'type=' + self.type
        if self.value:
            res += ', value=' + str(self.value)
        return "Event({})".format(res)

    def to_token(self):
        return self.type+'_'+str(self.value) if self.value is not None else self.type

    def __eq__(self, other):
        if isinstance(other, Event):
            return self.type == other.type and self.value == other.value
        return False