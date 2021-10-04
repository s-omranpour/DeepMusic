from typing import List, Dict, Union
from collections import Counter
from mido.midifiles import tracks
import numpy as np
import os
import logging
from copy import deepcopy

from miditoolkit.midi.parser import MidiFile
from miditoolkit.midi import containers as ct
from midi2audio import FluidSynth
from numpy.lib.function_base import insert

from .const import INSTRUMENT_FAMILIES, CHORDS, Constants
from .event import MusicEvent
import utils


class Music:
    def __init__(self, tracks : List, name : str = ''):
        assert len(tracks) > 0, "Empty list of tracks."
        consts = list(set([t.const for t in tracks]))
        assert len(consts) == 1, "Inconsistant constants among inserted tracks."
        self.const = consts[0]
        self.name = name

        n_bars = [t.get_bar_count() for t in tracks]
        max_bars = max(n_bars)
        for i,t in enumerate(tracks):
            if t.pad_with_bar(right=max_bars - n_bars[i]):
                logging.warn(f'Track no. {i+1} is padded to the max_n_bars={max_bars}.')
        self.tracks = tracks

    @staticmethod
    def from_file(
        file_path : str, 
        const: Constants = None, 
        unit: int = 12,
        min_tempo : int = 30,
        max_tempo : int = 300,
        num_tempo_bins : int = 30, 
        num_velocity_bins : int = 30):

        midi = MidiFile(file_path)
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
        tracks = [Track.create_track_from_midi_instrument(inst, tempo_changes=midi.tempo_changes, markers=midi.markers) for inst in midi.instruments]
        return Music(tracks, file_path)

    @staticmethod
    def from_pianoroll():
        pass

    @staticmethod
    def from_indices(indices : List, const : Constants = None):
        if const is None:
            const = Constants()
        return Music.from_remi([const.remi_tokens[idx] for idx in indices], const)


    @staticmethod
    def from_remi(remi : List, const : Constants = None):
        if const is None:
            const = Constants()
        prev_pos = 'Bar'
        pass

    @staticmethod
    def from_cp(cp : Union[List[List], np.array], const : Constants = None, use_program=True):
        cp = np.array(cp)
        assert len(cp.shape) == 2 and cp.shape[1] == 7, "Invalid cp shape."
        if const is None:
            const = Constants()
        
        tracks = []
        programs = set(cp[:, -1])
        for program in programs:
            cp_track = cp[cp[:, -1] == program][:, :-1]
            events = [MusicEvent.from_cp(c, const) for c in cp_track]
            tracks += [Track(program*8 if const.use_program else program, events).reorder_beats()]
        return Music(tracks)

    @staticmethod
    def merge(musics : List, merge_programs=True):
        """
            if merge == True:
                merge tracks with same program and different names
            else:
                merge tracks with same program and name
        """
        tracks = utils.flatten([m.get_tracks() for m in musics])
        res = Music(tracks, name = 'Merged '+', '.join([m.name for m in musics]))
        res.merge_similar_tracks(~merge_programs)
        return res      

    @staticmethod
    def concatenate(musics : List, merge_programs=True):
        """
            if merge == True:
                merge tracks with same program and different names
            else:
                merge tracks with same program and name
        """
        assert len(musics) > 1, "Not enough musics to concatenate."
        all_tracks = musics[0].get_tracks()
        n_bar = musics[0].get_bar_count()
        for music in musics[1:]:
            music.pad_with_bar(right=n_bar)
            n_bar = music.get_bar_count()
            all_tracks += music.get_tracks()
        res = Music(all_tracks, 'Concatenated '+', '.join([music.name for music in musics]))
        res.pad_tracks_to_same_length()
        res.merge_similar_tracks(~merge_programs)
        return res

    def merge_similar_tracks(self, use_name=True):
        ids = set([(t.program, t.name if use_name else '') for t in tracks])
        super_tracks = []
        for id in ids:
            super_tracks += [Track.merge_tracks(list(filter(lambda x: x.program == id[0] and x.name == id[1], self.tracks)))]
        self.tracks = super_tracks

    def pad_with_bar(self, left : int = 0, right : int = 0):
        for track in self.tracks:
            track.pad_with_bar(left, right)

    def pad_tracks_to_same_length(self):
        n_bars = [t.get_bar_count() for t in self.tracks]
        max_bars = max(n_bars)
        for i,t in enumerate(self.tracks):
            t.pad_with_bar(right=max_bars - n_bars[i])
    

    def remove_tracks(self, indices:List=None, programs:List=None, inst_families:List=None):
        tracks = deepcopy(self.tracks)
        if indices is not None:
            assert isinstance(indices, list), "Indices should be a list of integers."
            for idx in indices:
                assert isinstance(idx, int), "Indices should be a list of integers."
                tracks.pop(idx)

        if programs is not None:
            assert isinstance(programs, list), "Programs should be a list of integers."
            for t in tracks:
                if t.program in programs:
                    tracks.remove(t)

        if inst_families is not None:
            assert isinstance(inst_families, list), "Inst_families should be a list of strings."
            for t in tracks:
                if t.inst_family in inst_families:
                    tracks.remove(t)
        return Music(tracks)

    def keep_tracks(self, indices:List=None, programs:List=None, inst_families:List=None):
        tracks = []
        if indices is not None:
            assert isinstance(indices, list), "Indices should be a list of integers."
            for idx in indices:
                assert isinstance(idx, int), "Indices should be a list of integers."
                tracks += [self.tracks[idx]]

        if programs is not None:
            assert isinstance(programs, list), "Programs should be a list of integers."
            for t in self.tracks:
                if t.program in programs:
                    tracks += [t]

        if inst_families is not None:
            assert isinstance(inst_families, list), "Inst_families should be a list of strings."
            for t in self.tracks:
                if t.inst_family in inst_families:
                    tracks += [t]
        return Music(tracks)

    def get_tracks(self):
        return deepcopy(self.tracks)

    def get_bars(self):
        bars = [t.get_bars() for t in self.tracks]
        return [Music(tracks=list(b), name='bar '+str(i)) for i,b in enumerate(zip(*bars))]

    def get_bar_count(self):
        return self.tracks[0].get_bar_count()

    def get_instruments(self, family=False):
        return list(set([t.inst_family if family else t.program for t in self.tracks]))

    def to_remi(self, ret='token', add_eos=False, use_program=True):
        pass

    def to_cp(self, add_eos=False, use_program=True):
        pass

    def to_pianoroll(self, separate_tracks=True, binarize=False, add_tempo_chord=False):
        pass

    def to_midi(self, output_path : str = None):
        instruments, tempo_changes, markers, max_ticks = list(zip(*[t.to_midi_instrument() for t in self.tracks]))
        midi = MidiFile()
        midi.ticks_per_beat = self.const.tick_resol
        midi.instruments = instruments
        midi.max_tick = max(max_ticks)
        tempo_changes = [set(t) for t in tempo_changes]
        midi.tempo_changes = sorted(tempo_changes[0].intersection(*tempo_changes), key=lambda x: x.time)
        markers = [set(m) for m in markers]
        midi.markers = sorted(markers[0].intersection(*markers), key=lambda x: x.time)
        midi.key_signature_changes = []
        midi.time_signature_changes = [ct.TimeSignature(4, 4, 0)]
        if output_path:
            midi.dump(output_path)
        return midi

    def to_audio(self, audio_path, sf2_path=None):
        self.to_midi('test.mid')
        FluidSynth(sf2_path).midi_to_audio('test.mid', audio_path, sf2_path=sf2_path)
        os.remove('test.mid')


class Track:
    def __init__(self, program : int, events: List, name: str = ''):
        """
            if 0 <= program < 128:
                instrument is not a drum
            elif 128 <= program < 256:
                instrument is drum
        """
        self.events = []
        self.const = Constants()
        self.set_program(program)
        self.add_events(events)
        self.set_name(name)

    @staticmethod
    def create_track_from_midi_instrument(const : Constants, instrument : ct.Instrument, tempo_changes : List = None, markers : List = None):
        assert const is not None, "Please enter valid constants."
        assert instrument is not None and isinstance(instrument, ct.Instrument), "Invalid instrument."

        timeline = {}
        for note in instrument.notes:
            time = note.start // const.step
            if time not in timeline:
                timeline[time] = {'notes' : [], 'tempo' : None, 'chord' : None}
            timeline[time] = note

        for tempo in tempo_changes:
            time = tempo.time // const.step
            if time not in timeline:
                timeline[time] = {'notes' : [], 'tempo' : 0, 'chord' : 0}
            tempo.tempo = const.tempo_bins[np.argmin(np.abs(const.tempo_bins - int(tempo.tempo)))] ## quantize tempo
            timeline[time]['tempo'] = tempo
        
        for marker in markers:
            if marker.text.startswith('Chord_'):
                time = marker.time // const.step
                if time not in timeline:
                    timeline[time] = {'notes' : [], 'tempo' : 0, 'chord' : 0}
                marker.text = marker.text[6:]
                timeline[time]['chord'] = marker

        for pos in range(0, sorted(timeline)[-1] + 1, const.n_bar_steps):
            if pos not in timeline:
                timeline[pos] = []
        
        events = []
        for pos_idx in sorted(timeline):
            pos_events = []
            for note in timeline[pos_idx]['notes']:
                pos_events += [
                    MusicEvent(
                        position = pos_idx % const.n_bar_steps,
                        note_pitch=note.pitch,
                        note_duration=min(max((note.end - note.start) // const.step, 1), const.n_bar_steps), 
                        note_velocity=np.argmin(np.abs(const.velocity_bins - int(note.velocity)))
                    )
                ]
                if len(pos_events) == 0:
                    pos_events += [MusicEvent(position = pos_idx % const.n_bar_steps)]
                pos_events[0].set_metric_attributes(
                    tempo = np.argmin(np.abs(const.tempo_bins - timeline[pos_idx]['tempo'])) if timeline[pos_idx]['tempo'] is not None else 0,
                    chord = CHORDS.index(timeline[pos_idx]['chord'] if timeline[pos_idx]['chord'] is not None else 0)
                )
                events += pos_events
        return Track(instrument.program + (128 if instrument.is_drum else 0), events, instrument.name)

    @staticmethod
    def merge_tracks(tracks : List):
        consts = list(set([t.get_const() for t in tracks]))
        assert len(consts) == 1, "Inconsistent constants among tracks."
        program = Counter([t.program for t in tracks]).most_common(1)[0][0]
        name = 'Merged :' + ', '.join([t.name for t in tracks])

        n_bars = [t.get_bar_count() for t in tracks]
        max_bars = max(n_bars)
        for i,t in enumerate(tracks):
            if t.pad_with_bar(right=max_bars - n_bars[i]):
                logging.warn(f'Track no. {i+1} is padded to the max_n_bars={max_bars}.')
        
        events = []
        tracks = [t.organize() for t in tracks]
        for bar_idx in range(max_bars):
            bar = {}
            for t in tracks:
                for k,v in t[bar_idx].items():
                    if k not in bar:
                        bar[k] = []
                    bar[k] += v

            for pos_idx in sorted(bar):
                tempo = Counter([e.tempo for e in bar[pos_idx]]).most_common(1)[0][0]
                chord = Counter([e.chord for e in bar[pos_idx]]).most_common(1)[0][0]
                for e in bar[pos_idx]:
                    e.tempo = 0
                    e.chord = 0
                bar[pos_idx][0].tempo = tempo
                bar[pos_idx][0].chord = chord
                events += utils.remove_identical_events_in_bar(utils.remove_empty_events(bar[pos_idx]))

        return Track(program, events, name)
                
    @staticmethod
    def concatente(tracks: List):
        consts = list(set([t.get_const() for t in tracks]))
        assert len(consts) == 1, "Inconsistent constants among tracks."
        
        program = Counter([t.program for t in tracks]).most_common(1)[0][0]
        name = 'Concatenated :' + ', '.join([t.name for t in tracks])
        events = []
        for t in tracks:
            events += t.events
        return Track(program, events, name)

    def set_program(self, program : int):
        assert 0 <= program < 256, "Invalid program."
        self.program = program
        if program < 128:
            self.inst_family = INSTRUMENT_FAMILIES[program // 8]
            self.is_drum = False
        else:
            self.inst_family = INSTRUMENT_FAMILIES[-1]
            self.is_drum = True

    def add_events(self, events : List):
        if len(events):
            consts = list(set([e.const for e in events]))
            assert len(consts) == 1, "Inconsistent constants among inserted events."
            if len(self.events) == 0:
                self.const = consts[0]
            assert consts[0] == self.const, "Inconsistent constants between inserted events and current track."
            self.events += events

    def set_name(self, name : str):
        assert isinstance(name, str), "Invalid name."
        self.name = name

    def change_const(self, const : Constants):
        if const is not None and self.const != const:
            for e in self.events:
                e.change_const(const)
            self.const = const

    def pad_with_bar(self, left : int = 0, right : int = 0):
        if left > 0:
            self.events = [MusicEvent(const=self.const)] * left + self.events
        if right > 0:
            self.events += [MusicEvent(const=self.const)] * right
        return bool(left + right)

    def find_beat_index(self, beat):
        query = beat * self.const.unit
        n_bars = 0
        prev_idx = 0
        for i, e in enumerate(self.events):
            if e.position == 0:
                n_bars += 1
            if query < (n_bars-1)*self.const.n_bar_steps + e.position:
                return prev_idx
            prev_idx = i
        return -1

    def slice_by_index(self, start, end):
        return Track(self.program, self.events[start:end], const=self.const, desc=self.desc)

    def slice_by_beat(self, start, end):
        start = self.find_beat_index(start)
        end = self.find_beat_index(end)
        return Track(self.program, self.events[start:end], const=self.const, desc=self.desc)

    def slice_by_bar(self, start, end):
        start = self.find_beat_index(start*4)
        end = self.find_beat_index(end*4)
        return Track(self.program, self.events[start:end], const=self.const, desc=self.desc)

    def organize(self):
        def organize_bar(bar):
            res = {}
            beats = bar.get_beats()
            for beat in beats:
                res[beat[0].position] = beat
            return res
        return [organize_bar(bar) for bar in self.get_bars()]

    def reorder_beats(self):
        bars = self.get_bars()
        events = []
        for bar in bars:
            events += sorted(bar.events, key=lambda x: (x.position, int(x.tempo > 0) + int(x.chord > 0)))
        return Track(self.program, events, self.name)

    def get_beats(self):
        res = [[]]
        prev_pos = 0
        for e in self.events:
            if e.position == prev_pos:
                res[-1] += [e]
            else:
                res += [[e]]
                prev_pos = e.position
        for i,beat in enumerate(res):
            res[i] = Track(self.program, beat, const=self.const, desc=self.desc)
        return res

    def get_bars(self):
        res = []
        for e in self.events:
            if e.position == 0:
                res += [[e]]
            else:
                res[-1] += [e]
        for i,bar in enumerate(res):
            res[i] = Track(self.program, bar, const=self.const, desc=self.desc)
        return res

    def get_bar_count(self):
        return len(list(filter(lambda x: x.position == 0, self.events)))

    def get_instrument(self, family=False):
        return self.inst_family if family else self.program

    def get_events(self):
        return self.events

    def get_const(self):
        return self.const

    def get_name(self):
        return self.name

    def to_midi_instrument(self):
        tempos = []
        markers = []
        notes = []
        n_bars = 0
        for e in self.events:
            if e.position == 0:
                n_bars += 1
            if e.tempo:
                tempos += [ct.TempoChange(e.tempo, e.position*self.const.step + (n_bars-1)*self.const.bar_resol)]
            if e.chord:
                markers += [ct.Marker('Chord_'+e.chord, e.position*self.const.step + (n_bars-1)*self.const.bar_resol)]
            if e.note_pitch:
                s = self.const.step * e.position + (n_bars-1)*self.const.bar_resol
                notes += [
                    ct.Note(
                        velocity=self.const.velocity_bins[e.velocity - 1], 
                        pitch=e.pitch - 1, 
                        start=s, 
                        end=s + e.duration*self.const.step
                    )
                ]
        instrument = ct.Instrument(self.program % 128, self.program > 127)
        instrument.notes = sorted(notes, key=lambda x: x.start)
        max_tick = instrument.notes[-1].end
        return instrument, sorted(tempos, key=lambda x: x.time), sorted(markers, key=lambda x: x.time), max_tick

    def to_pianoroll(self, binarize=False, add_tempo_chord=True):
        roll = np.zeros(shape=(130, self.get_bar_count()*self.const.n_bar_steps))
        prev_pos = 0
        prev_bar = -1
        for e in self.events:
            if e.position == 0:
                prev_bar += 1
            prev_pos = e.position
            offset = prev_bar*self.const.n_bar_steps + prev_pos
            roll[128:, offset] = [e.tempo, e.chord]
            roll[e.pitch, offset:offset+e.duration] = 1 if binarize else e.velocity
        if add_tempo_chord:
            return roll
        return roll[:128]

    def to_remi(self, add_instrument_token=True, use_program=True):
        res = []
        prev_pos = 0
        for e in self.events:
            if e.position == prev_pos:
                res += e.to_remi()[1:]
            else:
                res += e.to_remi()
                prev_pos = e.position
            if add_instrument_token and e.has_note():
                res += ['NoteInstrument_'+str(self.program) if use_program else 'NoteInstrumentFamily_'+str(self.inst_family)]
        return res

    def to_cp(self, add_instrument_token=True, use_program=True):
        res = []
        has_note = []
        for e in self.events:
            res += [e.to_cp()]
            has_note += [e.has_note()]
        res = np.array(res)
        if add_instrument_token: 
            inst_token = np.array(has_note).astype(int) *\
                (self.program if use_program else INSTRUMENT_FAMILIES.index(self.inst_family))
            res = np.concatenate([res, inst_token[:, None]], axis=1)
        return res

    def __repr__(self):
        return f'Track(inst_family={self.inst_family}, program={self.program}, desc={self.desc})'
    
    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return self.events[index]

    def __eq__(self, other):
        if isinstance(other, Track):
            if self.program == other.program and len(self) == len(other) and self.const == other.const:
                bars = self.get_bars()
                other_bars = other.get_bars()
                if len(bars) == len(other_bars):
                    for bar1, bar2 in zip(bars, other_bars):
                        if not utils.compare_bars(bar1, bar2):
                            return False
                    return True
        return False