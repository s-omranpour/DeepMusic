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
from numpy.core.fromnumeric import shape

from deepmusic.utils import flatten

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
            t.clean()
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
    def from_pianoroll(rolls : Dict[int, np.array], const : Constants):
        if const is None:
            const = Constants()
        tracks = [Track.from_pianoroll(v, k, const) for k,v in rolls.items()]
        return Music(tracks)

    @staticmethod
    def from_indices(indices : List, const : Constants = None):
        if const is None:
            const = Constants()
        return Music.from_tokens(const.decode(indices), const)

    @staticmethod
    def from_tokens(tokens : List, const : Constants = None):
        if const is None:
            const = Constants()

        tracks = {}
        prev_pos_tok = None
        prev_tempo_tok = None
        prev_chord_tok = None
        nb = -1
        for idx,tok in enumerate(tokens):
            if tok.startswith('Bar'):
                nb += 1
                prev_pos_tok = tok
            elif tok.startswith('Beat'):
                prev_pos_tok = tok
            elif tok.startswith('Tempo'):
                prev_tempo_tok = tok
            elif tok.startswith('Chord'):
                prev_chord_tok = tok
            elif tok.startswith('NotePitch'):
                assert tokens[idx+1].startswith('NoteDuration') and tokens[idx+2].startswith('NoteVelocity') and tokens[idx+3].startswith('NoteInstrument')
                program = int(tokens[idx+3][14:])
                if program not in tracks:
                    tracks[program] = []
                tracks[program] += [MusicEvent.from_tokens([prev_pos_tok, prev_tempo_tok, prev_chord_tok] + tokens[idx:idx+3], bar=nb, const=const)] 
        tracks = [Track(k, v) for k,v in tracks.items()]
        return Music(tracks)

    @staticmethod
    def from_tuples(tuples : Union[List[List], np.array], const : Constants = None):
        tuples = np.array(tuples)
        assert len(tuples.shape) == 2 and tuples.shape[1] == 7, "Invalid tuples shape."
        if const is None:
            const = Constants()
        
        tracks = []
        programs = set(tuples[:, -1])
        for program in programs:
            track = tuples[tuples[:, -1] == program][:, :-1]
            events = [MusicEvent.from_tuple(t, const) for t in track]
            tracks += [Track(program, events)]
        return Music(tracks)

    @staticmethod
    def merge(musics : List, merge_similar_programs=True):
        """
            if merge == True:
                merge tracks with same program and different names
            else:
                merge tracks with same program and name
        """
        tracks = utils.flatten([m.get_tracks() for m in musics])
        res = Music(tracks, name = 'Merged from '+', '.join([m.name for m in musics]))
        if merge_similar_programs:
            programs = set([t.program for t in res.tracks])
            for program in programs:
                indices = [idx for idx in range(len(res.tracks)) if res.tracks[idx].program == program]
                res.merge_tracks(indices)
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
        res.merge_tracks(~merge_programs)
        return res

    def merge_tracks(self, indices):
        selected_tracks = [self.tracks[idx] for idx in indices]
        for idx in indices:
            self.tracks.pop(idx)
        self.tracks += [Track.merge_tracks(selected_tracks)]

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

    def to_tokens(self, return_indices=True, add_eos=False):
        res = []
        tracks = [t.organize() for t in self.tracks]
        for bars in zip(*tracks):
            for beat in range(self.const.n_bar_steps):
                for bar in bars:
                    if beat in bar:
                        res += bar[beat].to_tokens(remove_duplicate_metrics=True, add_instrument_token=True)
        res = utils.remove_duplicate_metrics_from_remi(res)
        if add_eos:
            res += [self.const.special_tokens[1]]
        return self.const.encode(res) if return_indices else res

    def to_tuples(self, add_eos=False):
        res = []
        tracks = [t.organize() for t in self.tracks]
        for bars in zip(*tracks):
            for beat in range(self.const.n_bar_steps):
                for bar in bars:
                    if beat in bar:
                        res += [bar[beat].to_tuples(add_instrument_token=True)]

    def to_pianoroll(self, binarize=False, add_tempo_chord=False):
        rolls = [t.to_pianoroll(binarize=binarize, add_tempo_chord=add_tempo_chord) for t in self.tracks]
        programs = [t.program for t in self.tracks]
        return dict(zip(programs, rolls))

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
    def __init__(self, program : int, events: List, const : Constants = None, name: str = ''):
        """
            if 0 <= program < 128:
                instrument is not a drum
            elif 128 <= program < 256:
                instrument is drum
        """
        self.events = []
        self.const = Constants() if const is None else const
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
            while time not in timeline: ## ensure tempo change happens only in presence of a note
                time += 1
            tempo.tempo = const.tempo_bins[np.argmin(np.abs(const.tempo_bins - int(tempo.tempo)))] ## quantize tempo
            timeline[time]['tempo'] = tempo
        
        for marker in markers:
            if marker.text.startswith('Chord_'):
                time = marker.time // const.step
                while time not in timeline: ## ensure chord change happens only in presence of a note
                    time += 1
                marker.text = marker.text[6:]
                timeline[time]['chord'] = marker
        
        events = []
        prev_tempo = 0
        prev_chord = 0
        for pos_idx in sorted(timeline):
            pos_events = []
            if timeline[pos_idx]['tempo'] is not None:
                prev_tempo = np.argmin(np.abs(const.tempo_bins - timeline[pos_idx]['tempo']))
            if timeline[pos_idx]['chord'] is not None:
                prev_chord = CHORDS.index(timeline[pos_idx]['chord'])
            for note in timeline[pos_idx]['notes']:
                pos_events += [
                    MusicEvent(
                        bar = pos_idx // const.n_bar_steps,
                        position = pos_idx % const.n_bar_steps,
                        tempo=prev_tempo,
                        chord=prev_chord,
                        note_pitch=note.pitch,
                        note_duration=min(max((note.end - note.start) // const.step, 1), const.n_bar_steps), 
                        note_velocity=np.argmin(np.abs(const.velocity_bins - int(note.velocity)))
                    )
                ]
            events += pos_events
        return Track(instrument.program + (128 if instrument.is_drum else 0), events, instrument.name)

    @staticmethod
    def from_pianoroll(roll : np.array, program : int, const : Constants = None, name : str = ''):
        padded = np.pad(roll[:128], ((0, 0), (1, 1)))
        diff = np.diff(padded.astype(np.int8), axis=1)
        pitches, note_ons = np.where(diff < 0)
        note_offs = np.where(diff > 0)[1]

        poses = {}
        for pitch, on, off in zip(pitches, note_ons, note_offs):
            if on not in poses:
                poses[on] = []
            poses[on] += [
                MusicEvent(
                    bar=on // const.n_bar_steps, 
                    beat=on % const.n_bar_steps, 
                    tempo=roll[128, on] if roll.shape[0] > 128 else 0,
                    chord=roll[129, on] if roll.shape[0] > 128 else 0,
                    note_pitch=pitch, 
                    note_duration=off - on, 
                    note_velocity=int(roll[pitch, on]),
                    const=const
                )
            ]
        return Track(program, flatten([poses[t] for t in sorted(poses)]))

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
        return Track(program, utils.flatten([t.events for t in tracks]), name)
                
    @staticmethod
    def concatente(tracks: List):
        consts = list(set([t.get_const() for t in tracks]))
        assert len(consts) == 1, "Inconsistent constants among tracks."
        
        program = Counter([t.program for t in tracks]).most_common(1)[0][0]
        name = 'Concatenated :' + ', '.join([t.name for t in tracks])
        n_prev_bars = [0] + [t.get_bar_count() for t in tracks[:-1]]
        events = []
        for nb, t in zip(n_prev_bars ,tracks):
            for e in t.events:
                e = deepcopy(e)
                e.set_metric_attributes(bar=e.bar + nb) ## update bar
                events += [e]
        return Track(program, events, name)

    def add_events(self, events : List):
        if len(events):
            consts = list(set([e.const for e in events]))
            assert len(consts) == 1, "Inconsistent constants among inserted events."
            if len(self.events) == 0:
                self.const = consts[0]
            assert consts[0] == self.const, "Inconsistent constants between inserted events and current track."
            self.events += events
            self.clean()
        logging.warn('Empty list of events entered.')

    def set_program(self, program : int):
        assert 0 <= program < 256, "Invalid program."
        self.program = program
        if program < 128:
            self.inst_family = INSTRUMENT_FAMILIES[program // 8]
            self.is_drum = False
        else:
            self.inst_family = INSTRUMENT_FAMILIES[-1]
            self.is_drum = True    
        

    def set_name(self, name : str):
        assert isinstance(name, str), "Invalid name."
        self.name = name

    def change_const(self, const : Constants):
        if const is not None and self.const != const:
            for e in self.events: ## update constants for each event
                e.change_const(const)
            self.const = const

    def pad_with_bar(self, left : int = 0, right : int = 0):
        if left > 0:
            for e in self.events:
                e.set_metric_attributes(bar=e.bar + left)
            self.events = [MusicEvent(const=self.const) for _ in range(left)] + self.events
        if right > 0:
            nb = self.get_bar_count()
            self.events += [MusicEvent(bar=nb+i+1, const=self.const) for i in range(right)]
        return bool(left + right)

    def find_beat_index(self, beat):
        query = beat * self.const.unit
        n_bars = 0
        for i, e in enumerate(self.events):
            if e.position == 0:
                n_bars += 1
            if query <= (n_bars-1)*self.const.n_bar_steps + e.position:
                return i
        return -1

    def slice_by_index(self, start, end):
        return Track(self.program, self.events[start:end], const=self.const, desc=self.desc)

    def slice_by_beat(self, start, end):
        start = self.find_beat_index(start)
        end = self.find_beat_index(end + 1)
        return Track(self.program, self.events[start:end], const=self.const, desc=self.desc)

    def slice_by_bar(self, start, end):
        start = self.find_beat_index(start*4)
        end = self.find_beat_index((end + 1)*4)
        return Track(self.program, self.events[start:end], const=self.const, desc=self.desc)

    def organize(self):
        res = [utils.classify_events_by_attr(bar.events, ['position']) for bar in self.get_bars()]
        for idx,bar in enumerate(res):
            for k,v in bar.items():
                bar[k] = Track(self.program, v, self.const, f'Bar No. {idx} - Beat No. {k}')
            res[idx] = bar
        return res

    def clean(self): 
        # 1. remove empty events 
        # 2. remove identical events 
        # 3. sort events
        # 4. synchronize metrics

        self.events = utils.sort_and_remove_identical_events(utils.remove_empty_events(self.events))
        bar_beats = utils.classify_events_by_attr(self.events, ['bar', 'beat'])
        self.events = []
        for pos in sorted(bar_beats):
            tempo = Counter([e.tempo for e in bar_beats[pos]]).most_common(1)[0][0] ## use voting to determine the tempo for this position
            chord = Counter([e.chord for e in bar_beats[pos]]).most_common(1)[0][0] ## use voting to determine the chord for this position
            for e in bar_beats[pos]: ## all events in a position should have same metrics
                e.tempo = tempo
                e.chord = chord
            self.events += bar_beats[pos]

    def get_bars(self):
        res = utils.classify_events_by_attr(self.events, ['bar'])
        return [Track(self.program, res[i], const=self.const, desc=self.desc + ' - Bar No. ' + str(i)) for i in sorted(res)]

    def get_bar_count(self):
        return len(set([e.bar for e in self.events]))

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
        prev_values = [None] * 2
        for e in self.events:
            tempo = e.get_actual_tempo()
            if tempo != prev_values[0]:
                tempos += [ct.TempoChange(tempo, e.get_position_in_ticks())]
            chord = e.get_actual_chord()
            if chord != prev_values[1]:
                markers += [ct.Marker('Chord_'+chord, e.get_position_in_ticks())]
            if e.note_pitch:
                notes += [
                    ct.Note(
                        velocity=e.get_actual_velocity(), 
                        pitch=e.pitch - 1, 
                        start=e.get_position_in_ticks(), 
                        end=e.get_position_in_ticks() + e.get_duration_in_ticks()
                    )
                ]
        instrument = ct.Instrument(self.program % 128, self.program > 127, name=self.name)
        instrument.notes = sorted(notes, key=lambda x: x.start)
        max_tick = instrument.notes[-1].end
        return instrument, sorted(tempos, key=lambda x: x.time), sorted(markers, key=lambda x: x.time), max_tick

    def to_pianoroll(self, binarize=False, add_tempo_chord=True):
        roll = np.zeros(shape=(130, self.get_bar_count()*self.const.n_bar_steps))
        for e in self.events:
            offset = e.bar*self.const.n_bar_steps + e.beat
            roll[128:, offset] = [e.tempo, e.chord]
            roll[e.pitch, offset:offset+e.duration] = 1 if binarize else e.velocity
        if add_tempo_chord:
            return roll
        return roll[:128]

    def to_tokens(self, remove_duplicate_metrics=True, add_instrument_token=True):
        res = []
        prev_beat = 0
        prev_tempo = 0
        prev_chord = 0
        for e in self.events:
            toks = e.to_tokens()
            selected = []
            if remove_duplicate_metrics:
                if e.beat != prev_beat:
                    selected = toks[1]
                if e.tempo > 0 and e.tempo != prev_tempo:
                    selected += [toks[2]]
                if e.chord > 0 and e.chord != prev_chord:
                    selected += [toks[3]]
            else:
                selected = toks
            if add_instrument_token and e.has_note():
                selected += ['NoteInstrument_'+str(self.program)]
            
            res += selected
            prev_beat = e.beat
            prev_tempo = e.tempo
            prev_chord = e.chord
        return res

    def to_tuples(self, add_instrument_token=True):
        res = []
        has_note = []
        for e in self.events:
            res += [e.to_tuple()]
            has_note += [e.has_note()]
        res = np.array(res)
        if add_instrument_token: 
            inst_token = np.array(has_note).astype(int) * self.program
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
            return self.program == other.program and self.const == other.const and len(self) == len(other) and set(self.events) == set(other.events)   
        return False