import os
from miditoolkit.midi import parser
from miditoolkit.midi import containers as ct
from collections import defaultdict
import numpy as np
from copy import deepcopy

from deepnote.const import Constants
from deepnote.modules import *
from deepnote import utils


class MusicRepr:
    def __init__(self, events: list = [], const : Constants = None):
        self.events = events
        self.const = const

    def __repr__(self):
        return 'MusicRepr(\n num_events={},\n const={}\n)'.format(len(self.events), self.const.__repr__())

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return self.events[index]

    def __eq__(self, other):
        if isinstance(other, MusicRepr):
            if self.__len__() == len(other) and self.const == other.const:
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
        query = beat * self.const.unit
        prev_bar = -self.const.n_bar_steps
        t = 0
        for i, e in enumerate(self.events):
            if isinstance(e, Metric) and e.position == 0:
                prev_bar += self.const.n_bar_steps
                t = prev_bar
            if isinstance(e, Metric) and e.position > 0:
                t = prev_bar + e.position
            if query < t:
                return i - 1
        return -1

    def slice_by_index(self, start, end):
        return MusicRepr(self.events[start:end], const=self.const)

    def slice_by_beat(self, start, end):
        start = self.find_beat_index(start)
        end = self.find_beat_index(end)
        return MusicRepr(self.events[start:end], const=self.const)

    def slice_by_bar(self, start, end):
        start = self.find_beat_index(start*4)
        end = self.find_beat_index(end*4)
        return MusicRepr(self.events[start:end], const=self.const)

    def get_bars(self):
        res = []
        for e in self.events:
            if isinstance(e, Metric) and e.position == 0:
                res += [[e]]
            else:
                res[-1] += [e]
        for i,bar in enumerate(res):
            res[i] = MusicRepr(bar, const=self.const)
        return res

    def get_bar_count(self):
        return len(list(filter(lambda x: isinstance(x, Metric) and x.position == 0, self.events)))

    def separate_tracks(self):
        existing_insts = set()
        tracks = dict([[inst, []] for inst in self.const.instruments])
        for e in self.events:
            if isinstance(e, Note):
                tracks[e.inst_family] += [e]
                existing_insts.update([e.inst_family])
            elif isinstance(e, Metric):
                for inst in tracks:
                    tracks[inst] += [e]
        res = {}
        for inst in existing_insts:
            res[inst] = MusicRepr(utils.remove_excess_pos(tracks[inst]), const=self.const)
        return res

    def get_instruments(self):
        insts = set([e.inst_family for e in filter(lambda x: isinstance(x, Note), self.events)])
        return list(insts)


    ## Input
    @staticmethod
    def from_file(
        file_path, 
        unit=12,
        min_tempo=30,
        max_tempo=300,
        num_tempo_bins=30, 
        num_velocity_bins=30):

        midi = parser.MidiFile(file_path)
        return MusicRepr.from_midi(midi, unit, min_tempo, max_tempo, num_tempo_bins, num_velocity_bins)

    @staticmethod
    def from_midi(
        midi, 
        unit=12,
        min_tempo=30,
        max_tempo=300,
        num_tempo_bins=30, 
        num_velocity_bins=30):

        midi = deepcopy(midi)
        tick_resol = midi.ticks_per_beat
        const = Constants(unit, tick_resol, min_tempo, max_tempo, num_tempo_bins, num_velocity_bins)

        events = defaultdict(list)
        for inst in midi.instruments:
            inst_family = const.instruments[16 if inst.is_drum else inst.program // 8]
            for note in inst.notes:
                events[note.start // const.step] += [
                    Note(
                        inst_family=inst_family, 
                        pitch=note.pitch, 
                        duration=min(max((note.end - note.start) // const.step, 1), const.n_bar_steps), 
                        velocity=const.velocity_bins[
                            np.argmin(np.abs(const.velocity_bins - int(note.velocity)))
                        ]
                    )
                ]

        for tempo in midi.tempo_changes:
            tempo.tempo = const.tempo_bins[
                np.argmin(np.abs(const.tempo_bins - int(tempo.tempo)))
            ]
            events[tempo.time // const.step] += [tempo]
        
        for marker in midi.markers:
            if marker.text.startswith('Chord'):
                marker.text = marker.text[6:]
                events[marker.time // const.step] += [marker]

        res = []
        pos_indices = sorted(list(events.keys()))
        bar_idx = -1
        for pos_idx in pos_indices:
            while pos_idx // const.n_bar_steps > bar_idx:
                res += [Metric()]  ## bar
                bar_idx += 1
            
            notes = []
            beat = Metric(position=(pos_idx % const.n_bar_steps) + 1)
            for e in events[pos_idx]:
                if isinstance(e, Note):
                    notes += [e]
                if isinstance(e, ct.Marker):
                    beat.chord = e.text
                if isinstance(e, ct.TempoChange):
                    beat.tempo = e.tempo
            res += [beat] + notes
        return MusicRepr(res, const=const)


    @staticmethod
    def from_cp(cp, const : Constants):
        cp = utils.clean_cp(np.array(cp))
        bars = []
        for c in cp:
            if c[0] == 0: ## metric
                if c[1] == 0:
                    bars += [[Metric()]]
                else:
                    bars[-1] += [Metric.from_cp(c, const=const)]
            
            elif c[0] == 1:  ## note
                bars[-1] += [Note.from_cp(c, const=const)]
        res = []
        for bar in bars:
            res += utils.sort_bar_beats(bar)
        return MusicRepr(res, const=const)

    @staticmethod
    def from_string(text, const : Constants):
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
                tempo = int(tokens[i].split('_')[1])
                assert tempo in const.tempo_bins
                bars[-1][-1].tempo = tempo

            elif tokens[i].startswith('BeatChord'):
                assert isinstance(bars[-1][-1], Metric) and bars[-1][-1].position > 0
                chord = tokens[i][9:]
                assert chord in const.chords
                bars[-1][-1].chord = chord

            elif tokens[i].startswith('NoteInstFamily'):
                assert tokens[i+1].startswith('NotePitch') and \
                    tokens[i+2].startswith('NoteDuration') and \
                        tokens[i+3].startswith('NoteVelocity')
                
                inst_family = tokens[i].split('_')[1]
                assert inst_family in const.instruments
                pitch = int(tokens[i+1].split('_')[1])
                assert 0 <= pitch < 128
                duration = int(tokens[i+2].split('_')[1])
                assert duration in const.duration_bins
                velocity = int(tokens[i+3].split('_')[1])
                assert velocity in const.velocity_bins

                bars[-1] += [
                    Note(
                        inst_family=inst_family,
                        pitch=pitch,
                        duration=duration,
                        velocity=velocity
                    )
                ]
                i += 3
            i += 1
        res = []
        for bar in bars:
            res += utils.sort_bar_beats(bar)
        return MusicRepr(res, const=const)

    @staticmethod
    def from_indices(indices, const : Constants):
        return MusicRepr.from_string(' '.join([const.all_tokens[idx] for idx in indices]))

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
        return MusicRepr(res, const=tracks[key_inst].const)

    @staticmethod
    def concatenate(seq_list : list):
        consts = [seq.const for seq in seq_list]
        for i in range(len(consts) - 1):
            assert consts[i] == consts[i+1], f"constants of seq {i} and {i+1} are not equal."
        
        events = []
        for seq in seq_list:
            events += seq.events
        return MusicRepr(events, const=consts[0])


    ## Output
    def to_remi(self, ret='token'):
        assert ret in ['token', 'index', 'event']
        res = []
        for e in self.events:
            res += e.to_remi()
        if ret != 'event':
            res = [r.to_token() for r in res]
        if ret == 'index':
            res = [self.const.all_tokens.index(tok) for tok in res]
        return res

    def to_cp(self):
        res = []
        for e in self.events:
            res += [e.to_cp(const=self.const)]
        return np.array(res)

    def to_midi(self, output_path=None):
        midi = parser.MidiFile()

        midi.ticks_per_beat = self.const.tick_resol
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
                    tempos += [ct.TempoChange(e.tempo, e.position*self.const.step + (n_bars-1)*self.const.bar_resol)]
                if e.chord:
                    chords += [ct.Marker('Chord_'+e.chord, e.position*self.const.step + (n_bars-1)*self.const.bar_resol)]
                prev_pos = e.position
            
            if isinstance(e, Note):
                s = self.const.step * prev_pos + (n_bars-1)*self.const.bar_resol
                instr_notes[e.inst_family] += [
                    ct.Note(velocity=e.velocity, 
                            pitch=e.pitch, 
                            start=s, 
                            end=s + e.duration*self.const.step)
                ]
                
        tempos.sort(key=lambda x: x.time)
        chords.sort(key=lambda x: x.time)
        
        max_tick = 0
        instruments = []
        for k, v in instr_notes.items():
            k = self.const.instruments.index(k)
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