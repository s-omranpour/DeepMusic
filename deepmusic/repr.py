from operator import pos
import os
from miditoolkit.midi import parser
from miditoolkit.midi import containers as ct
from collections import defaultdict
import numpy as np
from copy import deepcopy

from deepmusic.modules import utils, Constants, Note, Metric, Event


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
                            return False
                    return True
        return False

    def find_beat_index(self, beat):
        query = beat * self.const.unit
        n_bars = 0
        prev_idx = 0
        for i, e in enumerate(self.events):
            if isinstance(e, Metric):
                prev_pos = e.position
                if e.position == 0:
                    n_bars += 1
                if query < (n_bars-1)*self.const.n_bar_steps + prev_pos:
                    return prev_idx
                prev_idx = i
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
        const: Constants = None, 
        unit: int = 12,
        min_tempo : int = 30,
        max_tempo : int = 300,
        num_tempo_bins : int = 30, 
        num_velocity_bins : int = 30):

        midi = parser.MidiFile(file_path)
        return MusicRepr.from_midi(midi, const, unit, min_tempo, max_tempo, num_tempo_bins, num_velocity_bins)

    @staticmethod
    def from_midi(
        midi, 
        const: Constants = None, 
        unit: int = 12,
        min_tempo : int = 30,
        max_tempo : int = 300,
        num_tempo_bins : int = 30, 
        num_velocity_bins : int = 30):

        midi = deepcopy(midi)
        tick_resol = midi.ticks_per_beat
        
        if const is None:
            const = Constants(
                unit, 
                tick_resol, 
                min_tempo, 
                max_tempo, 
                num_tempo_bins, 
                num_velocity_bins
            )
        else:
            const.update_resolution(tick_resol)

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

        for pos in range(0, sorted(events)[-1] + 1, const.n_bar_steps):
            if pos not in events:
                events[pos] = []

        res = []
        for pos_idx in sorted(events):
            notes = []
            beat = Metric(position=pos_idx % const.n_bar_steps)
            for e in events[pos_idx]:
                if isinstance(e, Note):
                    notes += [e]
                if isinstance(e, ct.Marker):
                    beat.chord = e.text
                if isinstance(e, ct.TempoChange):
                    beat.tempo = e.tempo
            res += [beat] + sorted(notes, key=lambda x: x.pitch)
        return MusicRepr(res, const=const)


    @staticmethod
    def from_cp(cp, const : Constants = None):
        if const is None:
            const = Constants()

        cp = utils.clean_cp(np.array(cp))
        bars = []
        for c in cp:
            if c[0] == 0: ## metric
                if c[1] == 0:
                    bars += [[Metric.from_cp(c, const=const)]]
                else:
                    bars[-1] += [Metric.from_cp(c, const=const)]
            elif c[0] == 1:  ## note
                bars[-1] += [Note.from_cp(c, const=const)]
        res = []
        for bar in bars:
            res += utils.sort_bar_beats(bar)
        res = utils.remove_excess_pos(res)
        return MusicRepr(res, const=const)

    @staticmethod
    def from_string(text, const : Constants = None):
        if const is None:
            const = Constants()

        tokens = text.split()
        bars = []
        for token in tokens:
            if token == 'Bar':
                bars += [[Metric()]]

            elif token.startswith('BeatPosition'):
                bars[-1] += [Metric(position=int(token.split('_')[1]))]

            elif token.startswith('BeatTempo'):
                ## traverse backwords to find the recent metric
                for e in bars[-1][::-1]:
                    if isinstance(e, Metric):
                        tempo = int(token.split('_')[1])
                        assert tempo in const.tempo_bins, f"tempo={tempo} is not in the defined tempo bins"
                        e.tempo = tempo
                        # bars[-1][-(i+1)].tempo = tempo
                        break

            elif token.startswith('BeatChord'):
                ## traverse backwords to find the recent metric
                for e in bars[-1][::-1]:
                    if isinstance(e, Metric):
                        chord = token[10:]
                        assert chord in const.chords, f"chord={chord} is not in the defined chords"
                        e.chord = chord
                        break

            elif token.startswith('NoteInstFamily'):
                inst_family = token.split('_')[1]
                assert inst_family in const.instruments
                bars[-1] += [Note(inst_family=inst_family)]

            elif token.startswith('NotePitch'):
                ## traverse backwords to find the recent note
                for e in bars[-1][::-1]:
                    if isinstance(e, Note):
                        pitch = int(token.split('_')[1])
                        assert 0 <= pitch < 128, "wrong pitch number"
                        e.pitch = pitch
                        break

            elif token.startswith('NoteDuration'):
                ## traverse backwords to find the recent note
                for e in bars[-1][::-1]:
                    if isinstance(e, Note):
                        duration = int(token.split('_')[1])
                        assert duration in const.duration_bins, f"duration={duration} is not in the defined duration bins"
                        e.duration = duration
                        break

            elif token.startswith('NoteVelocity'):
                ## traverse backwords to find the recent note
                for e in bars[-1][::-1]:
                    if isinstance(e, Note):
                        velocity = int(token.split('_')[1])
                        assert velocity in const.velocity_bins, f"velocity={velocity} is not in the velocity bins"
                        e.velocity = velocity
                        break

        res = []
        for bar in bars:
            res += utils.sort_bar_beats(bar)
        return MusicRepr(utils.remove_excess_pos(res), const=const)

    @staticmethod
    def from_indices(indices, const : Constants = None):
        if const is None:
            const = Constants()
            
        return MusicRepr.from_string(' '.join([const.all_tokens[idx] for idx in indices]), const=const)

    @staticmethod
    def from_single_pianoroll(pianoroll : np.array, inst : str, const : Constants = None):
        if const is None:
            const = Constants()
        events = []
        for i in range(0, pianoroll.shape[1], const.n_bar_steps):
            events += utils.pianoroll_bar_to_events(pianoroll[:, i:i+const.n_bar_steps], inst, const)
        return MusicRepr(events, const)


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

    def remove_instruments(self, instruments_to_remove : list):
        assert len(instruments_to_remove) > 0, "please specify at least one instrument to remove."
        tracks = self.separate_tracks()
        tracks = dict([(inst, tracks[inst]) for inst in tracks if inst not in instruments_to_remove])
        return MusicRepr.merge_tracks(tracks)

    def keep_instruments(self, instruments_to_keep : list):
        assert len(instruments_to_keep) > 0, "please specify at least one instrument to keep."
        tracks = self.separate_tracks()
        tracks = dict([(inst, tracks[inst]) for inst in tracks if inst in instruments_to_keep])
        return MusicRepr.merge_tracks(tracks)

    ## Output
    def to_remi(self, ret='token', add_eos=False):
        assert ret in ['token', 'index', 'event']
        res = []
        for e in self.events:
            res += e.to_remi()
        if add_eos:
            res += [Event('EOS')]
        if ret != 'event':
            res = [r.to_token() for r in res]
        if ret == 'index':
            res = [self.const.all_tokens.index(tok) for tok in res]
        return res

    def to_cp(self, add_eos=False):
        res = []
        for e in self.events:
            res += [e.to_cp(const=self.const)]
        if add_eos:
            res += [[2] + [0]*7]
        return np.array(res)

    def to_pianoroll(self, separate_tracks=True, binarize=False, add_tempo_chord=False):
        def to_single_pianoroll(track : MusicRepr):
            roll = np.zeros(shape=(130, track.get_bar_count()*track.const.n_bar_steps))
            prev_pos = 0
            prev_bar = -1
            for e in track.events:
                if isinstance(e, Metric):
                    if e.position == 0:
                        prev_bar += 1
                    prev_pos = e.position
                    offset = prev_bar*track.const.n_bar_steps + prev_pos
                    roll[128:, offset] = e.to_cp(track.const)[2:4]
                else:
                    offset = prev_bar*track.const.n_bar_steps + prev_pos
                    roll[e.pitch, offset:offset+e.duration] = 1 if binarize else e.velocity
            if add_tempo_chord:
                return roll
            return roll[:128]

        if separate_tracks:
            tracks = self.separate_tracks()
            return dict([(inst, to_single_pianoroll(tracks[inst])) for inst in tracks])
        return to_single_pianoroll(self)
        
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
                prev_pos = e.position
                if e.position == 0:
                    n_bars += 1
                if e.tempo:
                    tempos += [ct.TempoChange(e.tempo, prev_pos*self.const.step + (n_bars-1)*self.const.bar_resol)]
                if e.chord:
                    chords += [ct.Marker('Chord_'+e.chord, prev_pos*self.const.step + (n_bars-1)*self.const.bar_resol)]
                
            
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
        max_tick = max(
            chords[-1].time if len(chords) > 0 else 0, 
            tempos[-1].time if len(tempos) > 0 else 0
        )
        
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
