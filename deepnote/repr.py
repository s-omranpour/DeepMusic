import os
from miditoolkit.midi import parser
from miditoolkit.midi import containers as ct
from collections import defaultdict
import numpy as np
from copy import deepcopy
import itertools
import warnings
from joblib import wrap_non_picklable_objects

from deepnote.const import *
from deepnote.modules import *
from deepnote import utils


class MusicRepr:
    def __init__(self, events=[], tick_resol : int = DEFAULT_TICK_RESOL, unit : int = DEFAULT_UNIT):
        self.events = events
        self.tick_resol = tick_resol
        self.unit = unit
        self.step = tick_resol // unit
        self.bar_resol = unit * 4

    def __repr__(self):
        return 'MusicRepr(tick_resol={}, unit={})'.format(self.tick_resol, self.unit)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return self.events[index]

    def __eq__(self, other):
        if isinstance(other, MusicRepr):
            if self.__len__() == len(other) and self.unit == other.unit:
                bars = self.get_bars()
                other_bars = other.get_bars()
                if len(bars) == len(other_bars):
                    for bar1, bar2 in zip(bars, other_bars):
                        if not utils.compare_bars(bar1, bar2):
                            print(bar1, bar2)
                            return False
                    return True
        return False

    def find_beat_index(self, beat):
        query = beat * self.unit
        prev_bar = -self.bar_resol
        t = 0
        for i, e in enumerate(self.events):
            if isinstance(e, Metric) and e.position == 0:
                prev_bar += self.bar_resol
                t = prev_bar
            if isinstance(e, Metric) and e.position > 0:
                t = prev_bar + e.position
            if query < t:
                return i - 1
        return -1

    def slice_by_index(self, start, end):
        return MusicRepr(self.events[start:end], tick_resol=self.tick_resol, unit=self.unit)

    def slice_by_beat(self, start, end):
        start = self.find_beat_index(start)
        end = self.find_beat_index(end)
        return MusicRepr(self.events[start:end])

    def slice_by_bar(self, start, end):
        start = self.find_beat_index(start*4)
        end = self.find_beat_index(end*4)
        return MusicRepr(self.events[start:end])

    def get_bars(self):
        res = []
        for e in self.events:
            if isinstance(e, Metric) and e.position == 0:
                res += [[e]]
            else:
                res[-1] += [e]
        for i,bar in enumerate(res):
            res[i] = MusicRepr(bar, tick_resol=self.tick_resol, unit=self.unit)
        return res

    def get_bar_count(self):
        return len(list(filter(lambda x: isinstance(x, Metric) and x.position == 0, self.events)))

    def separate_tracks(self):
        existing_insts = set()
        tracks = dict([[inst, []] for inst in DEFAULT_INSTRUMENTS])
        for e in self.events:
            if isinstance(e, Note):
                tracks[e.inst_family] += [e]
                existing_insts.update([e.inst_family])
            elif isinstance(e, Metric):
                for inst in tracks:
                    tracks[inst] += [e]
        res = {}
        for inst in existing_insts:
            res[inst] = MusicRepr(utils.remove_excess_pos(tracks[inst]))
        return res

    def get_instruments(self):
        insts = set([e.inst_family for e in filter(lambda x: isinstance(x, Note), self.events)])
        return list(insts)


    ## Input
    @staticmethod
    def from_file(file_path, unit=DEFAULT_UNIT):
        midi = parser.MidiFile(file_path)
        return MusicRepr.from_midi(midi, unit)

    @staticmethod
    def from_midi(midi, unit=DEFAULT_UNIT):
        midi = deepcopy(midi)
        tick_resol = midi.ticks_per_beat
        step = tick_resol // unit
        bar_resol = unit * 4

        events = defaultdict(list)
        for inst in midi.instruments:
            inst_family = DEFAULT_INSTRUMENTS[16 if inst.is_drum else inst.program // 8]
            for note in inst.notes:
                events[note.start // step] += [
                    Note(inst_family=inst_family, 
                         pitch=note.pitch, 
                         duration=min(max((note.end - note.start) // step, 1), bar_resol), 
                         velocity=DEFAULT_VEL_BINS[np.argmin(np.abs(DEFAULT_VEL_BINS - int(note.velocity)))])
                ]

        for tempo in midi.tempo_changes:
            tempo.tempo = DEFAULT_TEMPO_BINS[np.argmin(np.abs(DEFAULT_TEMPO_BINS - int(tempo.tempo)))]
            events[tempo.time // step] += [tempo]
        
        for marker in midi.markers:
            if marker.text.startswith('Chord'):
                marker.text = marker.text[6:]
                events[marker.time // step] += [marker]

        res = []
        pos_indices = sorted(list(events.keys()))
        bar_idx = -1
        for pos_idx in pos_indices:
            while pos_idx // bar_resol > bar_idx:
                res += [Metric()]  ## bar
                bar_idx += 1
            
            notes = []
            beat = Metric(position=(pos_idx % bar_resol) + 1)
            for e in events[pos_idx]:
                if isinstance(e, Note):
                    notes += [e]
                if isinstance(e, ct.Marker):
                    beat.chord = e.text
                if isinstance(e, ct.TempoChange):
                    beat.tempo = e.tempo
            res += [beat] + notes
        return MusicRepr(res, tick_resol=tick_resol, unit=unit)


    @staticmethod
    def from_cp(cp, unit=DEFAULT_UNIT):
        cp = utils.clean_cp(np.array(cp))
        bars = []
        for c in cp:
            if c[0] == 0: ## metric
                if c[1] == 0:
                    bars += [[Metric()]]
                else:
                    bars[-1] += [Metric.from_cp(c, unit=unit)]
            
            elif c[0] == 1:  ## note
                bars[-1] += [Note.from_cp(c, unit=unit)]
        res = []
        for bar in bars:
            res += utils.sort_bar_beats(bar)
        return MusicRepr(res)

    @staticmethod
    def from_string(text):
        tokens = text.split()
        n = len(tokens)
        i = 0
        bars = []
        while i < n:
            if tokens[i] == 'Bar':
                bars += [[Metric()]]

            elif tokens[i].startswith('BeatPosition'):
                bars[-1] += [Metric(position=int(tokens[i].split('_')[1]))]

            elif tokens[i].startswith('BeatTempo'):
                assert isinstance(bars[-1][-1], Metric) and bars[-1][-1].position > 0
                bars[-1][-1].tempo = int(tokens[i].split('_')[1])

            elif tokens[i].startswith('BeatChord'):
                assert isinstance(bars[-1][-1], Metric) and bars[-1][-1].position > 0
                bars[-1][-1].chord = tokens[i].split('_')[1]

            elif tokens[i].startswith('NoteInstFamily'):
                assert tokens[i+1].startswith('NotePitch') and \
                    tokens[i+2].startswith('NoteDuration') and \
                        tokens[i+3].startswith('NoteVelocity')

                bars[-1] += [
                    Note(pitch=int(tokens[i+1].split('_')[1]),
                         duration=int(tokens[i+2].split('_')[1]),
                         inst_family=tokens[i].split('_')[1],
                         velocity=int(tokens[i+3].split('_')[1])
                    )
                ]
                i += 3
            i += 1
        res = []
        for bar in bars:
            res += utils.sort_bar_beats(bar)
        return MusicRepr(res)

    @staticmethod
    def from_indices(indices):
        return MusicRepr.from_string(' '.join([DEFAULT_TOKENS[idx] for idx in indices]))

    @staticmethod
    def merge_tracks(tracks : dict, key_inst: str = None):
        insts = list(tracks.keys())
        bars = [track.get_bars() for track in tracks.values()]
        
        ### making sure all tracks have the same length by adding empty bars to the end
        max_len = max([len(bar) for bar in bars])
        for bar in bars:
            if len(bar) < max_len:
                bar.events += [Metric() for _ in range(max_len - len(bar))]

        key_inst = insts[0] if key_inst is None else key_inst
        res = utils.flatten([utils.merge_bars(dict(zip(insts, bar)), key_inst) for bar in zip(*bars)])
        return MusicRepr(res, tick_resol=tracks[key_inst].tick_resol, unit=tracks[key_inst].unit)


    ## Output
    def to_remi(self, ret='token'):
        assert ret in ['token', 'index', 'event']
        res = []
        for e in self.events:
            res += e.to_remi()
        if ret != 'event':
            res = [r.to_token() for r in res]
        if ret == 'index':
            res = [DEFAULT_TOKENS.index(tok) for tok in res]
        return res

    def to_cp(self):
        res = []
        for e in self.events:
            res += [e.to_cp()]
        return np.array(res)

    def to_midi(self, output_path=None):
        midi = parser.MidiFile()

        midi.ticks_per_beat = self.tick_resol
        tempos = []
        chords = []
        instr_notes = defaultdict(list)

        n_bars = 0
        prev_pos = 0
        for e in self.events:
            if isinstance(e, Metric):
                if e.position == 0:
                    n_bars += 1
                if e.tempo:
                    tempos += [ct.TempoChange(e.tempo, e.position*self.step + (n_bars-1)*self.bar_resol)]
                if e.chord:
                    chords += [ct.Marker('Chord_'+e.chord, e.position*self.step + (n_bars-1)*self.bar_resol)]
                prev_pos = e.position
            
            if isinstance(e, Note):
                s = self.step*prev_pos + (n_bars-1)*self.bar_resol
                instr_notes[e.inst_family] += [
                    ct.Note(velocity=e.velocity, 
                            pitch=e.pitch, 
                            start=s, 
                            end=s + e.duration*self.step)
                ]
                
        tempos.sort(key=lambda x: x.time)
        chords.sort(key=lambda x: x.time)
        
        max_tick = 0
        instruments = []
        for k, v in instr_notes.items():
            k = DEFAULT_INSTRUMENTS.index(k)
            inst = ct.Instrument(k * 8 if k < 16 else 0, k == 16)
            inst.notes = sorted(v, key=lambda x: x.start)
            max_tick = max(max_tick, inst.notes[-1].end)
            instruments += [inst]

        midi.instruments = instruments
        midi.max_tick = max_tick
        midi.tempo_changes = tempos
        midi.markers = chords
        midi.key_signature_changes = []
        midi.time_signature_changes = [ct.TimeSignature(4, 4, 0)]
        if output_path:
            midi.dump(output_path)
        return midi

    def to_audio(self, audio_path, sf2_path=None):
        self.to_midi('test.mid')
        utils.midi_to_audio('test.mid', audio_path, sf2_path=sf2_path)
        os.remove('test.mid')
        return