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

from .conf import INSTRUMENT_FAMILIES, CHORDS, MusicConfig
from .event import MusicEvent, NoteEvent, TempoEvent, ChordEvent
import utils


class Music:
    def __init__(self, tracks : List[Track], tempos : List[TempoEvent], chords : List[ChordEvent], name : str = ''):
        assert len(tracks) > 0, "Empty list of tracks."
        configs = list(set([t.config for t in tracks]))
        assert len(configs) == 1, "Inconsistant MusicConfig among inserted tracks."
        self.config = configs[0]
        self.name = name
        self.tracks = tracks
        self.tempos = tempos
        self.chords = chords

    @staticmethod
    def from_file(
        file_path : str, 
        config: MusicConfig = None, 
        unit: int = 12,
        min_tempo : int = 30,
        max_tempo : int = 300,
        num_tempo_bins : int = 30, 
        num_velocity_bins : int = 30):

        midi = MidiFile(file_path)
        midi = deepcopy(midi)
        tick_resol = midi.ticks_per_beat
        
        if config is None:
            config = MusicConfig(unit, tick_resol, min_tempo, max_tempo, num_tempo_bins, num_velocity_bins)
        else:
            config.update_resolution(tick_resol)

        tracks = [Track.create_track_from_midi_instrument(config, inst) for inst in midi.instruments]
        tempos = []
        for tempo in midi.tempo_changes:
            time = tempo.time // config.step
            tempo = np.argmin(np.abs(config.tempo_bins - int(tempo.tempo)))
            tempos += [TempoEvent(bar=time // config.n_bar_steps, beat=time % config.n_bar_steps, tempo=tempo)]

        chords = []
        for marker in midi.markers:
            if marker.text.startswith('Chord_'):
                time = marker.time // config.step
                chord = CHORDS.index(marker.text[6:]) 
                chords += [ChordEvent(bar=time // config.n_bar_steps, beat=time % config.n_bar_steps, chord=chord)]

        return Music(tracks, tempos, chords, file_path)

    @staticmethod
    def from_pianoroll(rolls : Dict[int, np.array], config : MusicConfig):
        if config is None:
            config = MusicConfig()
        tracks = [Track.from_pianoroll(v, k, config) for k,v in rolls.items()]
        return Music(tracks)

    @staticmethod
    def from_indices(indices : List, config : MusicConfig = None):
        if config is None:
            config = MusicConfig()
        return Music.from_tokens(config.decode(indices), config)

    @staticmethod
    def from_tokens(tokens : List, config : MusicConfig = None):
        if config is None:
            config = MusicConfig()

        tracks = {}
        tempos = []
        chords = []
        prev_pos_tok = None
        nb = -1
        for idx,tok in enumerate(tokens):
            if tok.startswith('Bar'):
                nb += 1
            elif tok.startswith('Beat'):
                prev_pos_tok = tok
            elif tok.startswith('Tempo'):
                tempos += [TempoEvent.from_tokens([prev_pos_tok, tok], bar=nb)]
            elif tok.startswith('Chord'):
                chords += [ChordEvent.from_tokens([prev_pos_tok, tok], bar=nb)]
            elif tok.startswith('NotePitch'):
                program = int(tokens[idx+3][14:])
                if program not in tracks:
                    tracks[program] = []
                tracks[program] += [NoteEvent.from_tokens([prev_pos_tok] + tokens[idx:idx+3], bar=nb)] 
        tracks = [Track(k, v, config) for k,v in tracks.items()]
        return Music(tracks, tempos, chords)

    @staticmethod
    def from_tuples(tuples : Union[List[List], np.array], config : MusicConfig = None):
        tuples = np.array(tuples)
        assert len(tuples.shape) == 2 and tuples.shape[1] == 7, "Invalid tuples shape."
        if config is None:
            config = MusicConfig()
        
        tracks = []
        programs = set(tuples[:, -1])
        for program in programs:
            track = tuples[tuples[:, -1] == program][:, :-1]
            events = [NoteEvent.from_tuple(t) for t in track]
            tracks += [Track(program, events, config)]

        tempos = []
        for time in np.where(tuples[:,2] > 0)[0]:
            bar, beat, tempo = tuples[time, :3]
            tempos += [TempoEvent(bar, beat, tempo-1)]

        chords = []
        for time in np.where(tuples[:,3] > 0)[0]:
            bar, beat, chord = tuples[time, :3]
            chords += [ChordEvent(bar, beat, chord-1)]
        return Music(tracks, tempos, chords)

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
    def concatenate(musics : List, merge_similar_programs=True):
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
            music.pad_left(n_bar)
            n_bar += music.get_bar_count()
            all_tracks += music.get_tracks()
        res = Music(all_tracks, 'Concatenated '+', '.join([music.name for music in musics]))
        if merge_similar_programs:
            programs = set([t.program for t in res.tracks])
            for program in programs:
                indices = [idx for idx in range(len(res.tracks)) if res.tracks[idx].program == program]
                res.merge_tracks(indices)
        return res

    def change_config(self, config : MusicConfig):
        for t in self.tracks:
            t.change_config(config)

        for tempo in self.tempos:
            tempo.tempo = np.argmin(np.abs(config.tempo_bins - self.config.tempo_bins[tempo.tempo]))
            

    def merge_tracks(self, indices):
        selected_tracks = [self.tracks[idx] for idx in indices]
        for idx in indices:
            self.tracks.pop(idx)
        self.tracks += [Track.merge_tracks(selected_tracks)]

    def pad_left(self, n : int):
        for track in self.tracks:
            track.pad_left(n)

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
        return Music(tracks, self.tempos, self.chords)

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
        return Music(tracks, self.tempos, self.chords)

    def get_tracks(self):
        return deepcopy(self.tracks)

    def get_bars(self):
        bars = [t.get_bars() for t in self.tracks]
        return [Music(tracks=list(b), tempos=self.tempos, chords=self.chords, name='bar '+str(i)) for i,b in enumerate(zip(*bars))]

    def get_bar_count(self):
        return max([t.get_bar_count() for t in self.tracks])

    def get_instruments(self, family=False):
        return list(set([t.inst_family if family else t.program for t in self.tracks]))

    def to_tokens(self, return_indices=True, add_eos=False):
        res = []
        tracks = [t.organize() for t in self.tracks]
        tempos = utils.organize_events_by_attr(self.tempos, ['bar', 'beat'])
        chords = utils.organize_events_by_attr(self.chords, ['bar', 'beat'])
        for bar_idx, bars in enumerate(zip(*tracks)):
            res += ['Bar']
            for beat in range(self.const.n_bar_steps):
                if (bar_idx,beat) in tempos:
                    res += ['Beat'+str(beat), 'Tempo'+str(tempos[bar_idx, beat][-1].tempo)]
                if (bar_idx,beat) in chords:
                    res += ['Beat'+str(beat), 'Chord'+str(chords[bar_idx, beat][-1].chord)]
                for bar in bars:
                    if beat in bar:
                        res += bar[beat].to_tokens(add_instrument_token=True)[1:] ## each track starts with a bar
        res = utils.remove_duplicate_beats_from_tokens(res)
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
        instruments, max_ticks = list(zip(*[t.to_midi_instrument() for t in self.tracks]))
        midi = MidiFile()
        midi.ticks_per_beat = self.const.tick_resol
        midi.instruments = instruments
        midi.max_tick = max(max_ticks)
        tempo_changes = []
        for e in self.tempos:
            tempo = self.config.tempo_bins[e.tempo] 
            tempo_changes += [ct.TempoChange(tempo, e.beat*self.config.step + e.bar*self.config.bar_resol)]
        chords = []
        for e in self.chords:
            chords += [ct.Marker('Chord_'+e.chord_name, e.beat*self.config.step + e.bar*self.config.bar_resol)]

        midi.tempo_changes = sorted(tempo_changes, key=lambda x: x.time)
        midi.markers = sorted(chords, key=lambda x: x.time)
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
    def __init__(self, program : int, notes: List, config : MusicConfig = None, name: str = ''):
        """
            if 0 <= program < 128:
                instrument is not a drum
            elif 128 <= program < 256:
                instrument is drum
        """
        self.notes = []
        self.config = MusicConfig() if config is None else config
        self.set_program(program)
        self.add_notes(notes)
        self.set_name(name)

    @staticmethod
    def create_track_from_midi_instrument(config : MusicConfig, instrument : ct.Instrument):
        assert config is not None, "Please enter a valid config."
        assert instrument is not None and isinstance(instrument, ct.Instrument), "Invalid instrument."

        notes = []
        for note in instrument.notes:
            pos = note.start // config.step
            notes += [
                NoteEvent(
                    bar = pos // config.n_bar_steps,
                    position = pos % config.n_bar_steps,
                    note_pitch=note.pitch,
                    note_duration=min(max((note.end - note.start) // config.step, 1), config.n_bar_steps), 
                    note_velocity=np.argmin(np.abs(config.velocity_bins - int(note.velocity)))
                )
            ]
        return Track(instrument.program + (128 if instrument.is_drum else 0), notes, config, instrument.name)

    @staticmethod
    def from_pianoroll(roll : np.array, program : int, config : MusicConfig = None, name : str = ''):
        config = config if config is not None else MusicConfig()
        padded = np.pad(roll[:128], ((0, 0), (1, 1)))
        diff = np.diff(padded.astype(np.int8), axis=1)
        pitches, note_ons = np.where(diff < 0)
        note_offs = np.where(diff > 0)[1]

        poses = {}
        for pitch, on, off in zip(pitches, note_ons, note_offs):
            if on not in poses:
                poses[on] = []
            poses[on] += [
                NoteEvent(
                    bar=on // config.n_bar_steps, 
                    beat=on % config.n_bar_steps, 
                    note_pitch=pitch, 
                    note_duration=off - on, 
                    note_velocity=int(roll[pitch, on]),
                    config=config
                )
            ]
        return Track(program, flatten([poses[t] for t in sorted(poses)], config, name))

    @staticmethod
    def merge_tracks(tracks : List):
        configs = list(set([t.get_config() for t in tracks]))
        assert len(configs) == 1, "Inconsistent MusicConfig among tracks."
        config = configs[0]
        program = Counter([t.program for t in tracks]).most_common(1)[0][0]
        name = 'Merged :' + ', '.join([t.name for t in tracks])
        return Track(program, utils.flatten([t.get_notes() for t in tracks]), config, name)
                
    @staticmethod
    def concatente(tracks: List):
        configs = list(set([t.get_config() for t in tracks]))
        assert len(configs) == 1, "Inconsistent MusicConfig among tracks."
        
        config = configs[0]
        program = Counter([t.program for t in tracks]).most_common(1)[0][0]
        name = 'Concatenated :' + ', '.join([t.name for t in tracks])
        n_prev_bars = [0] + [t.get_bar_count() for t in tracks[:-1]]
        notes = []
        for nb, track in zip(n_prev_bars ,tracks):
            t = deepcopy(track)
            t.pad_left(nb)
            notes += t.get_notes()
        return Track(program, notes, config, name)

    def add_notes(self, notes : List):
        if len(notes):
            for note in notes:
                if utils.validate_note(self.config, note):
                    self.notes += [note]
                else:
                    logging.warn('Invalid note filtered: ' + str(note))
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

    def change_config(self, config : MusicConfig):
        if config is not None and self.config != config:
            for note in self.notes:
                note = utils.update_note_with_config(note, config, self.config)
            self.config = config
        logging.warn('Config unchanged.')

    def pad_left(self, n : int = 0):
        if n > 0:
            for note in self.notes:
                note.set_metric_attributes(bar=note.bar + n)
        return bool(n)

    def find_beat_index(self, beat):
        query = beat * self.config.unit
        n_bars = 0
        for i, n in enumerate(self.notes):
            if n.beat == 0:
                n_bars += 1
            if query <= (n_bars-1)*self.config.n_bar_steps + n.beat:
                return i
        return -1

    def slice_by_index(self, start, end):
        return Track(self.program, self.notes[start:end], config=self.config, name=self.name)

    def slice_by_beat(self, start, end):
        start = self.find_beat_index(start)
        end = self.find_beat_index(end + 1)
        return Track(self.program, self.notes[start:end], config=self.config, name=self.name)

    def slice_by_bar(self, start, end):
        start = self.find_beat_index(start*4)
        end = self.find_beat_index((end + 1)*4)
        return Track(self.program, self.notes[start:end], config=self.config, name=self.name)

    def clean(self): 
        # 1. remove identical notes 
        # 2. sort notes
        self.notes = utils.sort_and_remove_identical_notes(self.notes)

    def organize(self):
        res = [utils.classify_events_by_attr(bar.get_notes(), ['beat']) for bar in self.get_bars()]
        for idx,bar in enumerate(res):
            for k,v in bar.items():
                bar[k] = Track(self.program, v, self.config, f'Bar No. {idx} - Beat No. {k}')
            res[idx] = bar
        return res

    def get_bars(self):
        res = utils.classify_events_by_attr(self.notes, ['bar'])
        for i in range(max(res) + 1):
            if i not in res:
                res[i] = []
        return [Track(self.program, res[i], config=self.config, desc=self.desc + ' - Bar No. ' + str(i)) for i in sorted(res)]

    def get_bar_count(self):
        return self.notes[-1].bar + 1

    def get_instrument(self, family=False):
        return self.inst_family if family else self.program

    def get_notes(self):
        return deepcopy(self.notes)

    def get_config(self):
        return self.config

    def get_name(self):
        return self.name

    def to_midi_instrument(self):
        notes = []
        for note in self.notes:
            s = note.beat*self.config.step + note.bar*self.config.bar_resol
            d = note.duration*self.config.step
            v = self.config.velocity_bins[note.velocity]
            notes += [
                ct.Note(
                    velocity=v, 
                    pitch=note.pitch, 
                    start=s, 
                    end=s + d
                )
            ]
        instrument = ct.Instrument(self.program % 128, self.program > 127, name=self.name)
        instrument.notes = sorted(notes, key=lambda x: x.start)
        max_tick = instrument.notes[-1].end
        return instrument, max_tick

    def to_pianoroll(self, binarize=False, add_tempo_chord=True):
        roll = np.zeros(shape=(128, self.get_bar_count()*self.config.n_bar_steps))
        for note in self.notes:
            offset = note.bar*self.config.n_bar_steps + note.beat
            roll[note.pitch, offset:offset+note.duration] = 1 if binarize else self.config.velocity_bins[note.velocity]
        return roll

    def to_tokens(self, add_instrument_token=True):
        res = []
        bars = self.organize()
        for bar in bars:
            res += ['Bar']
            for pos in bar:
                res += ['Beat' + str(pos)]
                for note in bar[pos].notes:
                    res += note.to_tokens(include_metrics=False)
                    if add_instrument_token:
                        res += ['NoteInstrument_'+str(self.program)]
        return res

    def to_tuples(self, add_instrument_token=True):
        res = []
        for note in self.notes:
            res += [note.to_tuple()]
        res = np.array(res)
        if add_instrument_token: 
            inst_token = np.ones(shape=(res.shape[0], 1)).astype(int) * self.program
            res = np.concatenate([res, inst_token[:, None]], axis=1)
        return res

    def __repr__(self):
        return f'Track(inst_family={self.inst_family}, program={self.program}, name={self.name})'
    
    def __len__(self):
        return len(self.notes)

    def __getitem__(self, index):
        return self.notes[index]

    def __eq__(self, other):
        if isinstance(other, Track):
            return self.program == other.program and self.config == other.config and len(self) == len(other) and set(self.notes) == set(other.notes)   
        return False